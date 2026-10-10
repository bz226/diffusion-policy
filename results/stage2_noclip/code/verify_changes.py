#!/usr/bin/env python
"""Bounded, fixed-data Stage 2 verification; never launches an environment.

Run only after BOTH pristine probes have finished and the two authorized source
files have been edited. CUDA mode includes the production-sized 200,000-pair
check and has a hard 30-minute process alarm. CPU mode checks small fixed-model
batches and a synthetic vector-environment fixture, not scientific training.
All persistent output goes to a new, explicit directory inside this study.
"""

import argparse
import copy
import importlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import types


STUDY = Path(__file__).resolve().parents[1]
PIN = "cc7234ad7ff39a8f32de3af903606723a16f0648"
ALLOWED = {
    "model/diffusion/diffusion_ppo.py",
    "agent/finetune/train_ppo_diffusion_agent.py",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def tree_equal(a, b, name="value"):
    if torch.is_tensor(a):
        require(torch.is_tensor(b) and torch.equal(a, b), name + " differs")
    elif isinstance(a, dict):
        require(a.keys() == b.keys(), name + " keys differ")
        for key in a:
            tree_equal(a[key], b[key], name + "." + str(key))
    elif isinstance(a, (tuple, list)):
        require(type(a) is type(b) and len(a) == len(b), name + " shape differs")
        for index, (x, y) in enumerate(zip(a, b)):
            tree_equal(x, y, name + "[" + str(index) + "]")
    else:
        require(a == b, name + " differs: %r != %r" % (a, b))


def finite(value, label):
    if torch.is_tensor(value):
        require(bool(torch.isfinite(value).all()), label + " nonfinite")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite(item, label + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite(item, label + "[" + str(index) + "]")
    elif isinstance(value, (int, float, np.number)):
        require(math.isfinite(float(value)), label + " nonfinite")


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state().clone(),
        "cuda": [x.clone() for x in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
    }


def rng_equal(a, b):
    require(a["python"] == b["python"], "Python RNG changed")
    require(a["numpy"][0] == b["numpy"][0], "NumPy RNG kind changed")
    require(np.array_equal(a["numpy"][1], b["numpy"][1]), "NumPy RNG changed")
    require(a["numpy"][2:] == b["numpy"][2:], "NumPy RNG cursor changed")
    tree_equal(a["torch"], b["torch"], "Torch CPU RNG")
    tree_equal(a["cuda"], b["cuda"], "Torch CUDA RNG")


def construct(cls, device, checkpoint=None, **flags):
    from model.diffusion.mlp_diffusion import DiffusionMLP
    from model.common.critic import CriticObs

    return cls(
        actor=DiffusionMLP(
            action_dim=6, horizon_steps=4, cond_dim=17, time_dim=16,
            mlp_dims=[512, 512, 512], activation_type="ReLU", residual_style=True,
        ),
        critic=CriticObs(
            cond_dim=17, mlp_dims=[256, 256, 256], activation_type="Mish",
            residual_style=True,
        ),
        gamma_denoising=0.99, clip_ploss_coef=0.01,
        clip_ploss_coef_base=0.01, clip_ploss_coef_rate=3,
        randn_clip_value=3, min_sampling_denoising_std=0.1,
        min_logprob_denoising_std=0.1, ft_denoising_steps=10,
        horizon_steps=4, obs_dim=17, action_dim=6, denoising_steps=20,
        device=device, network_path=str(checkpoint) if checkpoint else None,
        **flags,
    )


def default_equivalence(new_cls, old_cls, device):
    old = construct(old_cls, device)
    new = construct(new_cls, device)
    new.load_state_dict(old.state_dict())
    require(new.clamp_logprob is True, "default clamp flag changed")
    require(new.logprob_reduce == "mean", "default reduction changed")
    generator = torch.Generator(device=device).manual_seed(107)
    n = 128
    obs = {"state": torch.randn(n, 1, 17, generator=generator, device=device)}
    prev = torch.randn(n, 4, 6, generator=generator, device=device) * 0.3
    next_ = prev + torch.randn(n, 4, 6, generator=generator, device=device) * 0.1
    indices = torch.arange(n, device=device) % 10
    returns = torch.randn(n, generator=generator, device=device)
    values = torch.randn(n, generator=generator, device=device)
    advantages = torch.randn(n, generator=generator, device=device)
    with torch.no_grad():
        raw_old = old.get_logprobs_subsample(obs, prev, next_, indices)
        raw_old = raw_old + torch.linspace(-0.08, 0.08, n, device=device)[:, None, None]
    old_optim = [torch.optim.AdamW(old.actor_ft.parameters(), lr=1e-4, weight_decay=0),
                 torch.optim.AdamW(old.critic.parameters(), lr=1e-3, weight_decay=0)]
    new_optim = [torch.optim.AdamW(new.actor_ft.parameters(), lr=1e-4, weight_decay=0),
                 torch.optim.AdamW(new.critic.parameters(), lr=1e-3, weight_decay=0)]
    outputs = []
    for model, optimizers in ((old, old_optim), (new, new_optim)):
        result = model.loss(obs, prev, next_, indices, returns, values,
                            advantages.clone(), raw_old.clone(), reward_horizon=4)
        finite(result, "default loss")
        outputs.append(result)
        (result[0] + 0 * result[1] + 0.5 * result[2] + 0 * result[6]).backward()
    tree_equal(outputs[0], outputs[1], "pinned/default losses")
    for (name_a, param_a), (name_b, param_b) in zip(old.named_parameters(), new.named_parameters()):
        require(name_a == name_b, "parameter names changed")
        tree_equal(param_a.grad, param_b.grad, "gradient " + name_a)
    for optimizer in old_optim + new_optim:
        optimizer.step()
    tree_equal(old.state_dict(), new.state_dict(), "default optimizer update")
    for index in range(2):
        tree_equal(old_optim[index].state_dict(), new_optim[index].state_dict(),
                   "default optimizer state")
    return {"pairs": n, "losses_exact": True, "gradients_exact": True,
            "parameters_and_optimizer_states_exact": True}


def flag_behavior(new_cls):
    try:
        construct(new_cls, "cpu", logprob_reduce="invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid reduction was accepted")

    class ScalarCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, obs):
            return self.value.expand(len(obs["state"]), 1)

    results = {}
    for clamp in (True, False):
        for reduce in ("mean", "sum"):
            model = new_cls.__new__(new_cls)
            torch.nn.Module.__init__(model)
            model.device = "cpu"
            model.clamp_logprob, model.logprob_reduce = clamp, reduce
            model.norm_adv = True
            model.gamma_denoising, model.ft_denoising_steps = 0.99, 10
            model.clip_ploss_coef = model.clip_ploss_coef_base = 1e6
            model.clip_ploss_coef_rate = 3
            model.clip_vloss_coef = None
            model.clip_advantage_lower_quantile = 0
            model.clip_advantage_upper_quantile = 1
            model.critic = ScalarCritic()
            raw = torch.linspace(-7.0, 3.0, 16 * 24).reshape(16, 4, 6)
            model.test_logits = torch.nn.Parameter(raw.clone())

            def get_logprobs(self, *args, **kwargs):
                return self.test_logits, torch.ones_like(self.test_logits)

            model.get_logprobs_subsample = types.MethodType(get_logprobs, model)
            old = raw + torch.linspace(-0.03, 0.03, 16)[:, None, None]
            advantages = torch.linspace(-1, 1, 16)
            result = model.loss({"state": torch.zeros(16, 1, 17)}, raw, raw,
                                torch.arange(16) % 10, torch.ones(16), torch.zeros(16),
                                advantages, old, reward_horizon=4)
            new_np, old_np = raw.numpy().astype(np.float64), old.numpy().astype(np.float64)
            if clamp:
                new_np, old_np = np.clip(new_np, -5, 2), np.clip(old_np, -5, 2)
            reduction = np.mean if reduce == "mean" else np.sum
            expected = float(np.exp(reduction(new_np, axis=(1, 2)) -
                                    reduction(old_np, axis=(1, 2))).mean())
            require(math.isclose(result[5], expected, rel_tol=3e-6, abs_tol=3e-6),
                    "flag ratio does not implement requested clamp/reduction")
            result[0].backward()
            saturated = (raw < -5) | (raw > 2)
            gradients = model.test_logits.grad[saturated]
            require(bool((gradients == 0).all()) if clamp else bool((gradients != 0).any()),
                    "clamp flag does not control saturated-coordinate gradients")
            results[str(clamp) + "/" + reduce] = {"ratio": result[5], "reference": expected}
    require(results["True/mean"]["ratio"] != results["False/mean"]["ratio"],
            "clamp fixture did not distinguish behavior")
    require(results["False/mean"]["ratio"] != results["False/sum"]["ratio"],
            "reduction fixture did not distinguish behavior")
    return {"invalid_reduction_rejected": True, "flags": results}


def diagnostic_checks(agent_cls):
    n, k = 20000, 10
    count = n * k
    flat = torch.arange(count * 24, dtype=torch.int64).reshape(count, 4, 6)
    old = ((flat % 101).float() / 10 - 6).reshape(n, k, 4, 6)
    post = old.reshape(count, 4, 6) + ((flat % 19).float() - 9) * 0.0002

    class FakeModel:
        ft_denoising_steps = 10
        clamp_logprob = True
        logprob_reduce = "mean"
        training = True

        def get_logprobs_subsample(self, cond, prev, next_, indices, **kwargs):
            pair = cond["state"][:, 0, 0].long() * k + indices
            return post[pair]

    agent = agent_cls.__new__(agent_cls)
    agent.model = FakeModel()
    agent.device = "cpu"
    agent.n_envs, agent.n_steps = 40, 500
    agent.reward_horizon = 4
    agent.diagnostic_generator = torch.Generator(device="cpu").manual_seed(19)
    obs = {"state": torch.arange(n, dtype=torch.float32)[:, None, None].expand(n, 1, 17)}
    chains = torch.zeros(n, 11, 4, 6)
    state = agent.diagnostic_generator.get_state().clone()
    reference_generator = torch.Generator(device="cpu")
    reference_generator.set_state(state)
    sample = torch.randperm(count, generator=reference_generator)[:20000]
    before = rng_state()
    actual = agent._post_update_diagnostics(obs, chains, old)
    rng_equal(before, rng_state())
    require(agent.model.training is True, "diagnostics changed model mode")
    expected_new = post[sample].numpy().astype(np.float64)
    expected_old = old.reshape(count, 4, 6)[sample].numpy().astype(np.float64)
    d = expected_new.sum(axis=(1, 2)) - expected_old.sum(axis=(1, 2))
    expected = {
        "kl_true_per_action": float(k * np.mean(np.expm1(d) - d)),
        "logratio_p99": float(np.quantile(np.abs(d), 0.99)),
        "clamp_hit_frac": float(np.mean((expected_new < -5) | (expected_new > 2))),
    }
    for key, value in expected.items():
        require(math.isclose(actual[key], value, rel_tol=2e-12, abs_tol=2e-12),
                key + " independent float64 arithmetic differs")
    tree_equal(agent.diagnostic_generator.get_state(), reference_generator.get_state(),
               "diagnostic generator sampling")
    agent.model.clamp_logprob = False
    agent.model.logprob_reduce = "sum"
    agent.diagnostic_generator.set_state(state)
    before = rng_state()
    repeated = agent._post_update_diagnostics(obs, chains, old)
    rng_equal(before, rng_state())
    tree_equal(actual, repeated, "diagnostic independence from objective flags")
    return {"sample_pairs": 20000, "coordinate_denominator": 480000,
            "global_rng_unchanged": True, "flag_independent": True,
            "actual": actual, "independent_reference": expected}


def counting_optimizer(base_cls):
    class CountingOptimizer(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = 0
            self.zero_calls = 0
            self.before_step = None

        def zero_grad(self, *args, **kwargs):
            self.zero_calls += 1
            return super().zero_grad(*args, **kwargs)

        def step(self, *args, **kwargs):
            if self.before_step is not None:
                self.before_step()
            self.calls += 1
            return super().step(*args, **kwargs)
    return CountingOptimizer


def synthetic_loop(agent_cls, new_cls, output):
    """Exercise the actual run control flow with deterministic fake env/data."""
    from collections import namedtuple

    Sample = namedtuple("Sample", "trajectories chains")

    class Critic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.01))

        def forward(self, obs):
            return self.weight * (obs["state"][:, 0, 0:1] / 20000 + 1)

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor_ft = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.actor_ft.weight.fill_(0.125)
            self.critic = Critic()
            self.ft_denoising_steps = 10
            self.device = "cpu"
            self.norm_adv, self.clamp_logprob = True, False
            self.logprob_reduce = "mean"
            self.clip_ploss_coef = self.clip_ploss_coef_base = 1e6
            self.clip_ploss_coef_rate = 3
            self.clip_vloss_coef = None
            self.gamma_denoising = 0.99
            self.clip_advantage_lower_quantile = 0
            self.clip_advantage_upper_quantile = 1
            self.learn_eta = False
            self.calls = []
            self.analytic_chunk_gradients = []

        def forward(self, cond, **kwargs):
            n = len(cond["state"])
            return Sample(torch.zeros(n, 4, 6), torch.zeros(n, 11, 4, 6))

        def get_logprobs(self, cond, chains, **kwargs):
            feature = cond["state"][:, 0, 0] / 20000 + 1
            result = self.actor_ft.weight.reshape(1, 1, 1) * feature[:, None, None]
            return result.expand(len(chains), 4, 6).repeat_interleave(10, dim=0)

        def get_logprobs_subsample(self, cond, prev, next_, indices, get_ent=False, **kwargs):
            feature = cond["state"][:, 0, 0] / 20000 + 1
            result = self.actor_ft.weight.reshape(1, 1, 1) * feature[:, None, None]
            result = result.expand(len(indices), 4, 6)
            return (result, torch.ones_like(result)) if get_ent else result

        def loss(self, obs, prev, next_, indices, *args, **kwargs):
            self.calls.append((obs["state"][:, 0, 0].long() * 10 + indices).clone())
            # Each coordinate is scalar_actor * feature(obs). At ratio=1 the
            # policy derivative is -mean(normalized_advantage * discount *
            # feature). A nonconstant feature avoids a near-zero residual.
            advantage = args[2].detach().numpy().astype(np.float64)
            normalized = (advantage - advantage.mean()) / (advantage.std(ddof=1) + 1e-8)
            discount = np.power(0.99, 9 - indices.numpy()).astype(np.float32)
            feature = (obs["state"][:, 0, 0] / 20000 + 1).numpy()
            self.analytic_chunk_gradients.append(float(-np.mean(normalized * discount * feature)))
            return new_cls.loss(self, obs, prev, next_, indices, *args, **kwargs)

        def step(self):
            pass

        def get_min_sampling_denoising_std(self):
            return 0.1

    class Environment:
        def __init__(self):
            self.step_index = 0

        def observation(self):
            obs = np.zeros((40, 1, 17), dtype=np.float32)
            obs[:, 0, 0] = (self.step_index % 500) * 40 + np.arange(40)
            return {"state": obs}

        def reset(self):
            self.step_index = 0
            return self.observation()

        def step(self, action):
            self.step_index += 1
            done = np.full(40, self.step_index % 250 == 0)
            return self.observation(), np.full(40, 4.0), done, np.zeros(40, dtype=bool), [{}] * 40

    class Scheduler:
        def step(self):
            pass

    agent = agent_cls.__new__(agent_cls)
    for key, value in {
        "itr": 0, "n_train_itr": 2, "n_envs": 40, "n_steps": 500,
        "n_cond_step": 1, "obs_dim": 17, "horizon_steps": 4, "action_dim": 6,
        "act_steps": 4, "reward_horizon": 4, "device": "cpu", "val_freq": 10,
        "force_train": False, "reset_at_iteration": False, "render_freq": 1,
        "render_video": False, "n_render": 0, "render_dir": str(output),
        "save_full_observations": False, "save_trajs": False,
        "furniture_sparse_reward": False, "best_reward_threshold_for_success": 3,
        "logprob_batch_size": 10000, "reward_scale_running": False,
        "reward_scale_const": 1.0, "gamma": 0.99, "gae_lambda": 0.95,
        "batch_size": 50000, "update_epochs": 5, "target_kl": None,
        "use_bc_loss": False, "bc_loss_coeff": 0, "ent_coef": 0, "vf_coef": 0.5,
        "learn_eta": False, "n_critic_warmup_itr": 0, "max_grad_norm": None,
        "traj_plotter": None, "save_model_freq": 35, "log_freq": 1,
        "use_wandb": False, "actor_single_step": True,
        "result_path": str(output / "synthetic_fixture_result.pkl"),
    }.items():
        setattr(agent, key, value)
    agent.model = Policy()
    agent.venv = Environment()
    agent.reset_env_all = lambda options_venv=None: agent.venv.reset()
    agent.save_model = lambda: None
    agent.actor_lr_scheduler = Scheduler()
    agent.critic_lr_scheduler = Scheduler()
    agent.diagnostic_generator = torch.Generator(device="cpu").manual_seed(0)
    optimizer_cls = counting_optimizer(torch.optim.AdamW)
    agent.actor_optimizer = optimizer_cls(agent.model.actor_ft.parameters(), lr=1e-4, weight_decay=0)
    agent.critic_optimizer = optimizer_cls(agent.model.critic.parameters(), lr=1e-3, weight_decay=0)
    actor_before = copy.deepcopy(agent.model.actor_ft.state_dict())
    agent.critic_optimizer.before_step = lambda: tree_equal(
        actor_before, agent.model.actor_ft.state_dict(), "actor changed during critic updates")
    gradient_evidence = {}

    def before_actor_step():
        require(agent.critic_optimizer.calls == 20, "actor step did not follow all 20 critic steps")
        expected = math.fsum(agent.model.analytic_chunk_gradients[-4:]) / 4
        actual = agent.model.actor_ft.weight.grad.item()
        require(abs(expected) > 1e-3, "analytic fixture gradient is too small to check weighting")
        require(math.isclose(actual, expected, rel_tol=5e-5, abs_tol=5e-8),
                "accumulated actor gradient differs from weighted analytic fixture")
        gradient_evidence.update(actual=actual, analytic_reference=expected)

    agent.actor_optimizer.before_step = before_actor_step
    agent.run()
    require(agent.actor_optimizer.calls == 1, "expected exactly one actor step")
    require(agent.actor_optimizer.zero_calls == 1, "expected exactly one actor zero_grad")
    require(agent.critic_optimizer.calls == 20, "expected exactly 20 critic steps")
    require(len(agent.model.calls) == 24, "expected 20 critic and four actor loss calls")
    coverage = torch.cat(agent.model.calls[20:])
    tree_equal(torch.sort(coverage).values, torch.arange(200000), "actor pair coverage")
    tree_equal(coverage, torch.cat(agent.model.calls[16:20]), "last critic permutation reuse")
    import pickle
    with open(agent.result_path, "rb") as handle:
        rows = pickle.load(handle)
    row = rows[1]
    require(row["actor_step_ratio"] == 1.0 and row["actor_step_clipfrac"] == 0.0,
            "synthetic actor pre-step invariant failed")
    require(row["actor_optimizer_steps"] == 1 and row["critic_optimizer_steps"] == 20,
            "recorded optimizer counts differ from observed counts")
    finite(row, "synthetic saved row")
    return {"synthetic_only": True, "actor_steps": 1, "critic_steps": 20,
            "coverage": 200000, "reused_last_permutation": True,
            "actor_unchanged_during_critic": True, "ratio": row["actor_step_ratio"],
            "clipfrac": row["actor_step_clipfrac"], "actor_zero_grad_calls": 1,
            "accumulated_gradient": gradient_evidence}


def full_batch_check(agent_cls, new_cls, device, checkpoint):
    require(device.startswith("cuda"), "production-size check requires authorized CUDA allocation")
    require(checkpoint is not None and checkpoint.is_file(), "released checkpoint missing")
    model = construct(new_cls, device, checkpoint, clamp_logprob=False, logprob_reduce="mean")
    model.clip_ploss_coef = model.clip_ploss_coef_base = 1e6
    model.randn_clip_value = 100
    model.train()
    generator = torch.Generator(device=device).manual_seed(203)
    obs = {"state": torch.randn(20000, 1, 17, generator=generator, device=device) * 0.5}
    with torch.no_grad():
        chains = model(cond=obs, deterministic=False, return_chain=True).chains
        old = torch.cat([
            model.get_logprobs({"state": obs["state"][start:start + 10000]},
                               chains[start:start + 10000]).reshape(10000, 10, 4, 6)
            for start in (0, 10000)
        ], dim=0)
        values = model.critic(obs).flatten()
    advantages = torch.randn(20000, generator=generator, device=device)
    returns = values + advantages
    indices = torch.randperm(200000, generator=generator, device=device)
    agent = agent_cls.__new__(agent_cls)
    agent.model, agent.device = model, device
    agent.n_envs, agent.n_steps, agent.batch_size = 40, 500, 50000
    agent.itr, agent.n_critic_warmup_itr = 1, 0
    agent.max_grad_norm = None
    agent.use_bc_loss, agent.bc_loss_coeff, agent.ent_coef = False, 0, 0
    agent.learn_eta, agent.reward_horizon = False, 4
    agent.diagnostic_generator = torch.Generator(device="cpu").manual_seed(0)
    optimizer_cls = counting_optimizer(torch.optim.AdamW)
    agent.actor_optimizer = optimizer_cls(model.actor_ft.parameters(), lr=1e-4, weight_decay=0)
    critic_before = copy.deepcopy(model.critic.state_dict())
    raw_before = old.clone()
    result = agent._single_actor_step(obs, chains, returns, values, advantages, old, indices)
    require(agent.actor_optimizer.calls == 1, "real batch did not make one actor step")
    require(result["actor_step_ratio"] == 1.0, "real 100k/50k shape ratio is not exactly one")
    require(result["actor_step_clipfrac"] == 0.0, "real batch clipfrac is not exactly zero")
    tree_equal(old, raw_before, "cached old logprobs changed")
    tree_equal(model.critic.state_dict(), critic_before, "actor step changed critic")
    before = rng_state()
    diagnostics = agent._post_update_diagnostics(obs, chains, old)
    rng_equal(before, rng_state())
    finite(result, "real actor result")
    finite(diagnostics, "real post-update diagnostics")
    finite(model.state_dict(), "real post-update parameters")
    return {"pairs": 200000, "old_forward_pairs": 100000, "actor_chunk_pairs": 50000,
            "actor_steps": 1, "actor_metrics": result, "diagnostics": diagnostics,
            "old_cache_unchanged": True, "critic_unchanged": True,
            "global_rng_unchanged_by_diagnostics": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=STUDY / "dppo")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    require(0 < args.timeout_seconds <= 1800, "verification cap exceeds 30 minutes")
    require(args.device == "cpu" or args.device.startswith("cuda"), "unsupported device")
    output = args.output.resolve()
    require(STUDY in output.parents, "verification output must remain inside Stage 2")
    require(not output.exists(), "refusing existing verification output")
    output.mkdir(parents=True)
    report = {"status": "running", "command": sys.argv, "started_utc": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tests": {}, "device": args.device,
        "scientific_environment_runs": 0, "timeout_seconds": args.timeout_seconds}
    start = time.monotonic()

    def save():
        with open(output / "verification.json", "w") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)

    def timeout(signum, frame):
        raise TimeoutError("fixed-batch verification reached approved process cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.timeout_seconds)
    save()
    try:
        repo = args.repo.resolve()
        require(repo.is_dir(), "Stage 2 source missing")
        command = lambda *parts: subprocess.check_output(
            ["git", "-C", str(repo), *parts], text=True).strip()
        require(command("rev-parse", PIN) == PIN, "pinned revision unavailable")
        changed = set(command("diff", "--name-only", PIN).splitlines())
        require(changed == ALLOWED, "expected only the two approved scientific source edits")
        report["source"] = {"repo": str(repo), "head": command("rev-parse", "HEAD"),
                            "reference": PIN, "changed_paths": sorted(changed)}
        with open(output / "source.diff", "x") as handle:
            handle.write(command("diff", PIN) + "\n")
        sys.path.insert(0, str(repo))
        global torch, np
        import torch
        import numpy as np
        # CPU fixtures run in both modes. This is process-local verification
        # plumbing; never export a threading setting to scientific runs.
        torch.set_num_threads(1)
        torch.manual_seed(611)
        model_module = importlib.import_module("model.diffusion.diffusion_ppo")
        agent_module = importlib.import_module("agent.finetune.train_ppo_diffusion_agent")
        for module in (model_module, agent_module):
            require(repo in Path(module.__file__).resolve().parents,
                    "scientific import routed outside Stage 2")
        old_module = types.ModuleType("stage2_pinned_ppo_reference")
        pinned_source = command("show", PIN + ":model/diffusion/diffusion_ppo.py")
        exec(compile(pinned_source, "<pinned diffusion_ppo.py>", "exec"), old_module.__dict__)
        new_cls, old_cls = model_module.PPODiffusion, old_module.PPODiffusion
        agent_cls = agent_module.TrainPPODiffusionAgent
        cases = [
            ("default_equivalence", lambda: default_equivalence(new_cls, old_cls, args.device)),
            ("flag_behavior", lambda: flag_behavior(new_cls)),
            ("diagnostic_checks", lambda: diagnostic_checks(agent_cls)),
            ("synthetic_control_flow", lambda: synthetic_loop(agent_cls, new_cls, output)),
        ]
        if args.device.startswith("cuda"):
            require(torch.cuda.is_available(), "CUDA allocation unavailable")
            report["gpu"] = torch.cuda.get_device_name(torch.device(args.device))
            cases.append(("full_batch_shape_invariant", lambda: full_batch_check(
                agent_cls, new_cls, args.device, args.checkpoint)))
        else:
            report["full_batch_shape_invariant"] = "not_run_requires_authorized_GPU_check"
        for name, test in cases:
            report["active_test"] = name
            save()
            print("VERIFY START " + name, flush=True)
            phase_start = time.monotonic()
            result = test()
            report["tests"][name] = {"status": "passed", "seconds": time.monotonic() - phase_start,
                                     "findings": result}
            save()
            print("VERIFY PASS " + name, flush=True)
        report["status"] = "passed" if args.device.startswith("cuda") else "cpu_checks_passed_gpu_pending"
        report.pop("active_test", None)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        signal.alarm(0)
        report["elapsed_seconds"] = time.monotonic() - start
        report["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()
    print("VERIFY COMPLETE " + report["status"], flush=True)


if __name__ == "__main__":
    main()
