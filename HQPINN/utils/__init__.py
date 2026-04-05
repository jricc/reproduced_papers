"""
Shared training utilities for HQPINN benchmarks.

These helpers cover three recurring pieces of the reproduction:
- deterministic time grids for DHO,
- random collocation/boundary sampling for SEE,
- checkpoint and logging utilities reused across all benchmark wrappers.
"""

import os
from typing import Callable, Optional

import torch
import torch.nn as nn

from ..config import (
    DEVICE,
    DHO_N_SAMPLES,
    DTYPE,
    SEE_N_BC,
    SEE_N_F,
    SEE_N_IC,
    SEE_T_MAX,
    SEE_T_MIN,
    SEE_X_MAX,
    SEE_X_MIN,
)


def make_time_grid():
    """
    Return the DHO training grid on t in (0, 1].

    The first grid point at t=0 is excluded because the initial conditions are
    enforced analytically through dedicated loss terms in `core_dho.py`.
    """
    return torch.linspace(0.0, 1.0, DHO_N_SAMPLES, dtype=DTYPE, device=DEVICE)[
        1:
    ].reshape(-1, 1)


def make_optimizer(model, lr):
    """Create the Adam optimizer used by the benchmark training loops."""
    return torch.optim.Adam(model.parameters(), lr=lr)


def sample_ic_points():
    """
    Sample SEE initial-condition points on the line t=0.

    For the Sec. 3.1 smooth Euler problem, the initial state is a function of x,
    not a single scalar constraint. The code therefore samples points along the
    full initial line and uses them in the boundary/data term of the PINN loss.
    """
    x_ic = torch.rand(SEE_N_IC, 1, dtype=DTYPE, device=DEVICE)
    x_ic = SEE_X_MIN + (SEE_X_MAX - SEE_X_MIN) * x_ic
    t_ic = torch.zeros_like(x_ic)
    return x_ic, t_ic


def sample_bc_points():
    """
    Sample paired periodic-boundary points for SEE.

    The Sec. 3.1 benchmark imposes periodicity in x. Each sampled time is
    therefore turned into one left/right pair so the loss can penalize the
    mismatch U(x_min, t) - U(x_max, t).
    """
    t_bc = torch.rand(SEE_N_BC, 1, dtype=DTYPE, device=DEVICE)
    t_bc = SEE_T_MIN + (SEE_T_MAX - SEE_T_MIN) * t_bc
    x_left = torch.full_like(t_bc, SEE_X_MIN)
    x_right = torch.full_like(t_bc, SEE_X_MAX)
    return x_left, x_right, t_bc


def sample_collocation_points():
    """
    Sample interior collocation points for the SEE physics residual.

    These are the points where the code evaluates the Euler residual through
    automatic differentiation, exactly in the PINN sense of the paper's
    physics term.
    """
    x_f = torch.rand(SEE_N_F, 1, dtype=DTYPE, device=DEVICE)
    x_f = SEE_X_MIN + (SEE_X_MAX - SEE_X_MIN) * x_f
    t_f = torch.rand(SEE_N_F, 1, dtype=DTYPE, device=DEVICE)
    t_f = SEE_T_MIN + (SEE_T_MAX - SEE_T_MIN) * t_f
    return x_f, t_f


def count_trainable_params(model: nn.Module) -> int:
    """Count trainable parameters for the benchmark summary tables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_training_info(n_epochs, elapsed, final_loss, loss_ic, loss_bc, loss_f, rows):
    """Append one training snapshot to the in-memory CSV buffer and console."""
    print(
        f"Epoch {n_epochs:5d} | elapsed={elapsed:.2f}s  "
        f"L={final_loss:.3e} | "
        f"IC={loss_ic.item():.3e} | "
        f"BC={loss_bc.item():.3e} | "
        f"F={loss_f.item():.3e}"
    )

    rows.append(
        [
            n_epochs,
            f"{elapsed:.2f}",
            f"{final_loss:.3e}",
            f"{loss_ic.item():.3e}",
            f"{loss_bc.item():.3e}",
            f"{loss_f.item():.3e}",
        ]
    )


def load_model(
    ckpt_path: str, model_ctor: Callable[..., nn.Module], processor=None
) -> nn.Module:
    """
    Rebuild a model architecture, load a checkpoint, and return it in eval mode.

    The optional `processor` is only used for Merlin-based remote inference.
    """
    state = torch.load(ckpt_path, map_location="cpu")
    model = model_ctor(processor=processor)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model from: {ckpt_path} remote={processor is not None}")
    return model


def get_latest_checkpoint(ckpt_dir: str, case_prefix: str) -> Optional[str]:
    """Return the lexicographically latest checkpoint for one experiment case."""
    if not os.path.isdir(ckpt_dir):
        print(f"No checkpoint directory found at {ckpt_dir}")
        return None

    files = [
        f
        for f in os.listdir(ckpt_dir)
        if f.startswith(f"{case_prefix}_") and f.endswith(".pt")
    ]
    if not files:
        print(f"No checkpoints matching {case_prefix}_*.pt in {ckpt_dir}")
        return None

    files.sort()
    latest = files[-1]
    ckpt_path = os.path.join(ckpt_dir, latest)
    print(f"Latest checkpoint found: {ckpt_path}")
    return ckpt_path


def load_latest_model_local(
    ckpt_dir: str,
    case_prefix: str,
    model_ctor: Callable[[], nn.Module],
) -> Optional[nn.Module]:
    """Convenience wrapper for local checkpoint loading in interactive use."""
    ckpt_path = get_latest_checkpoint(ckpt_dir, case_prefix)
    if ckpt_path is None:
        return None
    return load_model(ckpt_path, model_ctor)
