"""Validate the one approved H200 allocation without widening other runs."""
import json
from pathlib import Path

from fixed_batch_gate import COMPARISONS, INPUTS, RNG_STATES, checked_comparison, require_file


RUN_ID = "NC1_seed2"
JOB_ID = "17777249"
SOURCE = "ab46b150fa34b5a5b457cd4062cd4c5ad830d964"
BASE = "cc7234ad7ff39a8f32de3af903606723a16f0648"


def validate_h200_gate(path, root, expected_commit):
    path = Path(path).resolve()
    original_path = root / "runs/verification_full_update/verification.json"
    original = json.loads(original_path.read_text())
    report = json.loads(path.read_text())
    if (report.get("passed") is not True or report.get("status") != "passed"
            or report.get("source_commit") != expected_commit
            or report.get("reference_commit") != BASE
            or report.get("gpu_name") != "NVIDIA H200 NVL"
            or report.get("source_verified_clean_before") is not True
            or report.get("source_verified_clean_after") is not True
            or report.get("deterministic_algorithms_enabled") is not False
            or report.get("deterministic_settings_unchanged") is not True
            or not report.get("numeric_settings_before")
            or report.get("numeric_settings_before") != report.get("numeric_settings_after")
            or report.get("numeric_settings_before") != original.get("numeric_settings_before")):
        raise ValueError("H200 fixed-batch check has missing/failed source, hardware or settings evidence")
    for key in ("timeout_seconds", "elapsed_seconds"):
        value = report.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 600:
            raise ValueError("H200 verification exceeded its approved ten-minute cap")
    if (report.get("original_gate") != str(original_path)
            or report.get("origin_batch") != original.get("capture", {}).get("saved_batch_path")):
        raise ValueError("H200 verification must reuse the original saved training batch")
    for key in ("origin_batch", "origin_capture_config"):
        if not isinstance(report.get(key), str):
            raise ValueError("Missing H200 saved-input evidence: " + key)
        require_file(report[key], key)
    result = report.get("gpu", {})
    update = result.get("full_update", {})
    if (result.get("passed") is not True or update.get("passed") is not True
            or update.get("epochs") != 5 or update.get("minibatches") != 20
            or update.get("actor_steps") != 20 or update.get("critic_steps") != 20
            or result.get("tolerance") != {"atol": 1e-7, "rtol": 1e-5}):
        raise ValueError("H200 gate must verify the full twenty-minibatch update at the agreed tolerance")
    for category in ("same_saved_batch", "same_initial_model_state", "same_initial_optimizer_states", "same_initial_rng_state"):
        if update.get(category) is not True:
            raise ValueError("Missing identical-input H200 check: " + category)
    for category in INPUTS:
        checked_comparison(update.get("initial_comparisons", {}).get(category, {}), "H200 initial " + category, True)
    for category in COMPARISONS:
        checked_comparison(update.get("comparisons", {}).get(category, {}), "H200 " + category,
                           category in ("permutations", "rng_after_update"))
    minibatches = update.get("minibatch_comparisons", [])
    if len(minibatches) != 20:
        raise ValueError("Missing H200 individual minibatch comparisons")
    for index, minibatch in enumerate(minibatches):
        if minibatch.get("minibatch") != index or minibatch.get("epoch") != index // 4:
            raise ValueError("Invalid H200 minibatch comparison schedule")
        for category in ("loss_components", "actor_gradients", "critic_gradients"):
            checked_comparison(minibatch.get(category, {}), "H200 minibatch%d %s" % (index, category), False)
    sampling = result.get("sampling", {})
    if (sampling.get("passed") is not True or sampling.get("returned_chain_states") != 11
            or sampling.get("observed_transition_states") != 20):
        raise ValueError("H200 sampling must compare every denoising transition")
    for category in ("chain", "trajectories", "all_20_transition_states"):
        checked_comparison(sampling.get("comparisons", {}).get(category, {}), "H200 sampling " + category, False)
    isolation = result.get("rng_isolation", {})
    if (isolation.get("passed") is not True or set(isolation.get("states", {})) != set(RNG_STATES)
            or any(isolation["states"][key] is not True for key in RNG_STATES)
            or isolation.get("cuda_generator_count") != 1):
        raise ValueError("H200 diagnostics must preserve all required global RNG states")
    for filename in ("pristine_update_evidence.pt", "edited_update_evidence.pt",
                     "sampling_evidence.pt", "diagnostic_rng_evidence.pt"):
        require_file(path.parent / "gpu" / filename, "H200 gpu/" + filename)
    return {"path": str(path), "passed": True, "source_commit": expected_commit,
            "criterion": "H200_full_update_fixed_batch_gpu_float_tolerance"}


def validate_hardware(gpu_query, args, environment, root):
    """Return exception provenance for the sole authorized job, otherwise None."""
    root = Path(root).resolve()
    lines = gpu_query.strip().splitlines()
    if len(lines) != 1:
        raise ValueError("Each run requires exactly one visible GPU")
    name = lines[0].split(",", 1)[0].strip()
    if "RTX A6000" in name:
        return None  # Preserve the originally approved allocation for every other run.
    if name != "NVIDIA H200 NVL" or args.run_id != RUN_ID or environment.get("SLURM_JOB_ID") != JOB_ID:
        raise ValueError("Each run requires RTX A6000 except the approved NC1 seed-2 H200 job")
    path = root / "provenance/h200_seed2_authorization.json"
    approved = json.loads(path.read_text())
    gate_path = root / "runs/verification_h200/verification.json"
    original_path = root / "runs/verification_full_update/verification.json"
    required = {"approved": True, "run_id": RUN_ID, "job_id": JOB_ID, "source_commit": SOURCE,
                "from_gpu": "NVIDIA RTX A6000", "to_gpu": "NVIDIA H200", "slurm_gres": "gpu:h200nvl:1",
                "gpus": 1, "cpus": 40, "memory_gib": 64, "runtime_cap_seconds": 10800,
                "verification_allocation_cap_seconds": 600,
                "h200_gate_path": str(gate_path), "existing_fixed_batch_gate": str(original_path),
                "deterministic_settings_changed": False, "other_runs_unchanged": True,
                "scientific_source_changes": False}
    if any(approved.get(key) != value for key, value in required.items()):
        raise ValueError("H200 authorization does not match the approved single-job scope")
    if (args.expected_commit != SOURCE or args.condition != "NC1" or args.seed != 2 or args.rows != 140
            or args.timeout_seconds != 10800 or Path(args.fixed_batch_gate).resolve() != original_path
            or environment.get("SLURM_CPUS_PER_TASK") != "40"
            or environment.get("SLURM_MEM_PER_NODE") != "65536"):
        raise ValueError("H200 run differs from the approved source, configuration or resource limits")
    gate = validate_h200_gate(gate_path, root, args.expected_commit)
    return {"authorization_path": str(path), "run_id": RUN_ID, "slurm_job_id": JOB_ID,
            "original_gpu_type": "NVIDIA RTX A6000", "actual_gpu_type": name,
            "same_pending_job_retargeted": True, "h200_fixed_batch_gate": gate}
