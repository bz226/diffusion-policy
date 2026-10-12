#!/usr/bin/env python
"""Registered actor parameter vectors and distances; no rollouts or updates.

Import load_checkpoint_vectors for the approved path-dot analysis. The CLI
creates only a new distance/provenance record and never writes checkpoints.
Checkpoint loading and arithmetic run on CPU; do not execute on the login host.
"""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch


STUDY = Path(__file__).resolve().parents[1]
PARENT = STUDY.parents[2]
SOURCE = PARENT / "results/stage3_gradmean/dppo"
MAPPING = STUDY / "provenance/checkpoints.json"
POINTS = ("P0", "D1", "D2", "N1", "N2")
CONFIG_RELATIVE = "cfg/gym/finetune/halfcheetah-v2/ft_ppo_diffusion_mlp.yaml"
EXPECTED_PARAMETER_COUNT = 568392


def _registered_actor(source):
    """Use the real actor constructor; preserve CPU RNG and create no env."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    source = Path(source).resolve()
    if not (source / CONFIG_RELATIVE).is_file():
        raise ValueError("Missing HalfCheetah source configuration: " + str(source))
    sys.path.insert(0, str(source))
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda expression: eval(
            expression, {"__builtins__": {}}, {}))
    cfg = OmegaConf.load(source / CONFIG_RELATIVE)
    actor_cfg = OmegaConf.to_container(cfg.model.actor, resolve=True)
    module_name = actor_cfg["_target_"].rsplit(".", 1)[0]
    module = importlib.import_module(module_name)
    if source not in Path(module.__file__).resolve().parents:
        raise RuntimeError("Actor module was already imported from a different source tree")
    with torch.random.fork_rng(devices=[]):
        actor = instantiate(actor_cfg).cpu()
    names = [name for name, _ in actor.named_parameters()]
    specs = [{"name": name, "shape": list(parameter.shape),
              "dtype": str(parameter.dtype), "numel": parameter.numel()}
             for name, parameter in actor.named_parameters()]
    if len(names) != len(set(names)) or sum(x["numel"] for x in specs) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("Registered actor parameter geometry differs from Addendum G")
    if any(parameter.dtype != torch.float32 for parameter in actor.parameters()):
        raise ValueError("The registered actor must use native float32 parameters")
    return actor, specs, actor_cfg


def _check_gradient_order(specs, meta_path):
    meta = json.loads(Path(meta_path).read_text())
    saved = meta.get("parameter_specs")
    if not isinstance(saved, list):
        raise ValueError("Gradient metadata is missing parameter_specs")
    geometry = lambda rows: [(row["name"], list(row["shape"])) for row in rows]
    if geometry(saved) != geometry(specs):
        raise ValueError("Registered actor order differs from gradient metadata: " + str(meta_path))
    if meta.get("parameter_count") != EXPECTED_PARAMETER_COUNT:
        raise ValueError("Gradient metadata parameter count differs")


def load_checkpoint_vectors(mapping_path=MAPPING, source_path=SOURCE, gradient_meta=None):
    """Return ({point: float64 vector}, metadata) in actor_ft registration order.

    P0 mirrors DiffusionModel's EMA network load followed by VPG's actor_ft
    deepcopy. RL points mirror VPG's final model.actor_ft state load. Values
    remain exactly those saved as float32; conversion precedes subtraction.
    """
    mapping_path, source_path = Path(mapping_path).resolve(), Path(source_path).resolve()
    mapping = json.loads(mapping_path.read_text())
    if set(mapping) != set(POINTS):
        raise ValueError("Checkpoint mapping must contain exactly P0,D1,D2,N1,N2")
    actor, specs, actor_cfg = _registered_actor(source_path)
    if gradient_meta is not None:
        _check_gradient_order(specs, gradient_meta)
    vectors, sources = {}, {}
    for index, point in enumerate(POINTS):
        record = mapping[point]
        if record.get("index") != index:
            raise ValueError("Checkpoint point index mismatch: " + point)
        checkpoint_path = Path(record["checkpoint"]).resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        container, prefix = ("ema", "network.") if point == "P0" else ("model", "actor_ft.")
        if container not in checkpoint:
            raise ValueError(point + " checkpoint is missing expected " + container + " state")
        state = {name[len(prefix):]: tensor for name, tensor in checkpoint[container].items()
                 if name.startswith(prefix)}
        if set(state) != set(actor.state_dict()):
            raise ValueError(point + " checkpoint actor state keys differ from the registered model")
        for name, parameter in actor.named_parameters():
            tensor = state[name]
            if (not torch.is_tensor(tensor) or tensor.dtype != torch.float32
                    or tensor.shape != parameter.shape or not bool(torch.isfinite(tensor).all())):
                raise ValueError(point + " invalid actor parameter " + name)
        actor.load_state_dict(state, strict=True)
        vector = torch.cat([parameter.detach().reshape(-1).double()
                            for parameter in actor.parameters()]).numpy().copy()
        if vector.shape != (EXPECTED_PARAMETER_COUNT,) or not np.isfinite(vector).all():
            raise ValueError(point + " invalid flattened actor vector")
        vectors[point] = vector
        sources[point] = {"checkpoint": str(checkpoint_path), "state_container": container,
                          "parameter_prefix": prefix, "index": index}
        del checkpoint, state
    metadata = {"mapping": str(mapping_path), "source": str(source_path),
                "actor_config": actor_cfg, "parameter_specs": specs,
                "parameter_count": EXPECTED_PARAMETER_COUNT, "sources": sources,
                "flatten_order": "registered actor_ft.named_parameters(); each tensor row-major",
                "arithmetic_dtype": "float64 converted from exact saved float32 parameters",
                "gradient_meta_checked": str(Path(gradient_meta).resolve()) if gradient_meta else None}
    return vectors, metadata


def distance_summary(vectors):
    """Euclidean distances with all differences formed in float64."""
    if set(vectors) != set(POINTS):
        raise ValueError("All five checkpoint vectors are required")
    distances = {a: {b: float(np.linalg.norm(vectors[a] - vectors[b])) for b in POINTS}
                 for a in POINTS}
    return {"distance_from_P0": {point: distances[point]["P0"] for point in POINTS},
            "path_increment_distance": {"D2_minus_D1": distances["D2"]["D1"],
                                        "N2_minus_N1": distances["N2"]["N1"]},
            "pairwise_euclidean": distances,
            "units": "Euclidean distance of native actor_ft parameters; no coordinate scaling"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh record directory inside Addendum G")
    parser.add_argument("--mapping", type=Path, default=MAPPING)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--gradient-meta", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if STUDY not in output.parents or output.exists():
        raise ValueError("Use a fresh output directory inside Addendum G")
    output.mkdir(parents=True)
    started = time.monotonic()
    record = {"status": "running", "command": sys.argv, "cwd": str(Path.cwd()),
              "parent_commit": subprocess.check_output(
                  ["git", "-C", str(PARENT), "rev-parse", "HEAD"], text=True).strip(),
              "torch_version": torch.__version__, "numpy_version": np.__version__}
    (output / "checkpoint_distances_source.py").write_bytes(Path(__file__).read_bytes())
    (output / "source.diff").write_bytes(subprocess.check_output([
        "git", "-C", str(PARENT), "diff", "HEAD", "--", str(args.source.resolve())]))
    try:
        vectors, metadata = load_checkpoint_vectors(args.mapping, args.source, args.gradient_meta)
        record.update(metadata)
        record.update(distance_summary(vectors))
        record["status"] = "complete"
    except BaseException as error:
        record.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        (output / "checkpoint_distances.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print("CHECKPOINT_DISTANCES " + json.dumps({"status": record["status"],
          "output": str(output / "checkpoint_distances.json")}), flush=True)


if __name__ == "__main__":
    main()
