"""Shared fixed-batch gate helpers, inherited from the accepted Stage-2 gate."""
import ast
import copy
import math
import os
import random
import numpy as np
import torch

ATOL, RTOL = 1e-7, 1e-5

def require(value, message):
    if not value:
        raise AssertionError(message)

def freeze(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: freeze(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(freeze(item) for item in value)
    return copy.deepcopy(value)


def transfer(value, device):
    if torch.is_tensor(value):
        return value.to(device).clone()
    if isinstance(value, dict):
        return {key: transfer(item, device) for key, item in value.items()}
    return copy.deepcopy(value)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [x.clone() for x in torch.cuda.get_rng_state_all()]}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def bits_equal(a, b):
    if not torch.is_tensor(a) or not torch.is_tensor(b):
        return False
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                       b.detach().cpu().contiguous().reshape(-1).view(torch.uint8))


def compare(reference, edited, bitwise, name="value"):
    """Relative error is |new-old|/max(|old|,float64 tiny), no hidden floor."""
    result = {"passed": True, "bitwise_equal": True, "max_abs": 0.0, "max_rel": 0.0,
              "elements": 0, "tensors": 0, "first_failure": None}

    def fail(label):
        result["passed"] = False
        if result["first_failure"] is None:
            result["first_failure"] = label

    def walk(a, b, label):
        if torch.is_tensor(a):
            if not torch.is_tensor(b) or a.dtype != b.dtype or a.shape != b.shape:
                fail(label + ": tensor dtype/shape")
                return
            result["tensors"] += 1
            result["elements"] += a.numel()
            exact = bits_equal(a, b)
            result["bitwise_equal"] &= exact
            x, y = a.detach().cpu().double(), b.detach().cpu().double()
            if x.numel():
                finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
                if not finite:
                    fail(label + ": nonfinite")
                    return
                delta = (y - x).abs()
                relative = delta / x.abs().clamp(min=torch.finfo(torch.float64).tiny)
                result["max_abs"] = max(result["max_abs"], delta.max().item())
                result["max_rel"] = max(result["max_rel"], relative.max().item())
                accepted = exact if bitwise or not a.is_floating_point() else bool(
                    (delta <= ATOL + RTOL * x.abs()).all())
                if not accepted:
                    fail(label + ": numerical difference")
        elif isinstance(a, np.ndarray):
            walk(torch.from_numpy(a.copy()), torch.from_numpy(b.copy()), label)
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                fail(label + ": dictionary keys")
                return
            for key in a:
                walk(a[key], b[key], label + "." + str(key))
        elif isinstance(a, (tuple, list)):
            if type(a) is not type(b) or len(a) != len(b):
                fail(label + ": sequence structure")
                return
            for index, (x, y) in enumerate(zip(a, b)):
                walk(x, y, label + "." + str(index))
        elif isinstance(a, (float, np.floating)):
            walk(torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64), label)
        elif a != b:
            result["bitwise_equal"] = False
            fail(label + ": nonnumeric value")

    walk(reference, edited, name)
    if math.isinf(result["max_rel"]):
        result["max_rel"] = "inf"
    return result


def rng_comparison(before, after):
    states = {key: compare(before[key], after[key], True, key)["passed"] for key in before}
    return {"passed": all(states.values()), "states": states,
            "cuda_generator_count": len(before["torch_cuda"])}


def numeric_settings():
    return {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "torch_num_threads": torch.get_num_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS")}


def assignment_to(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == name for target in node.targets)
