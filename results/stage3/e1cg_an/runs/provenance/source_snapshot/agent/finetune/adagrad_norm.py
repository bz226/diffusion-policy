"""AdaGrad-Norm with one float64 accumulator for the entire actor.

The gradient is left unchanged. Each step first updates b with the squared
L2 norm of all parameter gradients, then subtracts (lr / b) times the gradient.
There is no momentum, weight decay, per-coordinate scaling or clipping.
"""
import math

import torch


class AdaGradNorm(torch.optim.Optimizer):
    def __init__(self, params, lr, b0):
        if b0 is None or not math.isfinite(float(b0)) or float(b0) <= 0:
            raise ValueError("adagrad_norm_b0 must be explicitly set, finite and positive")
        if not math.isfinite(float(lr)) or float(lr) <= 0:
            raise ValueError("AdaGrad-Norm lr must be finite and positive")
        super().__init__(params, {"lr": float(lr)})
        if len(self.param_groups) != 1 or not self.param_groups[0]["params"]:
            raise ValueError("AdaGrad-Norm requires one nonempty actor parameter group")
        self.b0 = float(b0)
        self.b = torch.tensor(self.b0, dtype=torch.float64,
                              device=self.param_groups[0]["params"][0].device)
        self.actor_updates = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        group = self.param_groups[0]
        squared_norm = torch.zeros((), dtype=torch.float64, device=self.b.device)
        for parameter in group["params"]:
            if parameter.grad is not None:
                if parameter.grad.is_sparse:
                    raise RuntimeError("AdaGrad-Norm requires dense actor gradients")
                gradient = parameter.grad.detach().to(dtype=torch.float64)
                squared_norm.add_(torch.sum(gradient * gradient))
        next_b = torch.sqrt(self.b * self.b + squared_norm)
        if not bool(torch.isfinite(next_b)) or float(next_b.item()) <= 0:
            raise FloatingPointError("nonfinite AdaGrad-Norm accumulator")
        lr = float(group["lr"])
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("AdaGrad-Norm lr must remain finite and positive")
        self.b.copy_(next_b)
        scale = -lr / self.b.item()
        for parameter in group["params"]:
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=scale)
        self.actor_updates += 1
        return loss

    def metrics(self):
        """Read-only scalar values for the effective current step size."""
        value = self.b.item()
        return {"adagrad_b": value,
                "adagrad_effective_lr": float(self.param_groups[0]["lr"]) / value,
                "adagrad_actor_updates": self.actor_updates}

    def state_dict(self):
        payload = super().state_dict()
        # Keep global state separate from per-parameter optimizer state: torch's
        # default loader otherwise casts floating state to the parameter dtype.
        payload["adagrad_norm"] = {"version": 1, "b": self.b.detach().cpu().clone(),
                                    "b0": self.b0, "actor_updates": self.actor_updates}
        return payload

    def load_state_dict(self, state_dict):
        metadata = state_dict.get("adagrad_norm")
        if metadata is None or metadata.get("version") != 1:
            raise ValueError("optimizer state lacks supported AdaGrad-Norm scalar state")
        value = metadata.get("b")
        if (not torch.is_tensor(value) or value.dtype != torch.float64
                or value.numel() != 1 or not bool(torch.isfinite(value).all())
                or float(value.item()) <= 0):
            raise ValueError("saved AdaGrad-Norm b must be a positive float64 scalar")
        b0 = float(metadata["b0"])
        updates = metadata["actor_updates"]
        if not math.isfinite(b0) or b0 <= 0 or not isinstance(updates, int) or updates < 0:
            raise ValueError("invalid saved AdaGrad-Norm initial state or update count")
        base_state = {key: value for key, value in state_dict.items() if key != "adagrad_norm"}
        super().load_state_dict(base_state)
        if len(self.param_groups) != 1:
            raise ValueError("saved AdaGrad-Norm state must have one actor parameter group")
        self.b = value.detach().reshape(()).to(
            device=self.param_groups[0]["params"][0].device, dtype=torch.float64).clone()
        self.b0 = b0
        self.actor_updates = updates
