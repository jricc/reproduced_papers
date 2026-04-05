"""
Core training and evaluation utilities for the DHO benchmark.

This module implements the Appendix A.2 damped harmonic oscillator setting used
as the simplest HQPINN validation problem. Unlike the Euler cases, DHO uses a
scalar ODE, a scalar target u(t), and full-batch training on a small 1D grid.
"""

import os
import csv
from datetime import datetime
import sys
from typing import Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad

import matplotlib

# Keep batch exports headless, but do not disable inline notebook rendering.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ...config import (
    M,
    MU,
    K,
    LAMBDA1,
    LAMBDA2,
    DTYPE,
    DEVICE,
)

DHO_SUMMARY_COLUMNS = [
    "run_id",
    "Model",
    "Size",
    "epoch",
    "elapsed time (s)",
    "Trainable parameters",
    "Loss",
    "IC_u",
    "IC_du",
    "PDE",
    "Relative L2 error",
]


# ============================================================
#  Damped oscillator (dho)
# ============================================================


def omega(mu: float = MU, k: float = K) -> float:
    """Return the underdamped angular frequency appearing in the closed form."""
    return np.sqrt(k - (mu / 2.0) ** 2)


def u_exact(t_array: np.ndarray, mu: float = MU, k: float = K) -> np.ndarray:
    """Exact DHO solution used for quantitative evaluation and plots."""
    w = omega(mu, k)
    return np.exp(-mu * t_array / 2.0) * (
        np.cos(w * t_array) + (mu / (2.0 * w)) * np.sin(w * t_array)
    )


def evaluate_dho_error(model: nn.Module, t_eval: torch.Tensor) -> float:
    """Relative L2 error on the provided time grid."""
    with torch.no_grad():
        u_pred = model(t_eval).detach().cpu().numpy().reshape(-1)
    u_ref = u_exact(t_eval.detach().cpu().numpy().reshape(-1))
    num = np.sqrt(np.mean((u_pred - u_ref) ** 2))
    den = np.sqrt(np.mean(u_ref**2))
    return float(num / den)


def derivative(u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute du/dt using PyTorch autograd."""
    return grad(
        outputs=u,
        inputs=t,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]


def second_derivative(u: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute d²u/dt² using PyTorch autograd."""
    du_dt = derivative(u, t)
    return derivative(du_dt, t)


def oscillator_loss(
    model: nn.Module,
    t: torch.Tensor,
    m: float = M,
    mu: float = MU,
    k: float = K,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the three components of the DHO PINN loss:

      1. Initial condition on u(0):  (u(0) - 1)^2
      2. Initial condition on u'(0): (u'(0))^2
      3. ODE residual: mean((m u'' + mu u' + k u)^2)

    Returns
    -------
    loss_ic_u : torch.Tensor
        Initial condition loss on u(0).
    loss_ic_du : torch.Tensor
        Initial condition loss on u'(0).
    loss_f : torch.Tensor
        PDE residual loss.
    """
    # Work on a fresh differentiable copy so autograd can build the temporal
    # derivatives required by the physics term at every optimization step.
    t = t.clone().detach().requires_grad_(True)

    # Evaluate the trial solution and its first/second time derivatives.
    u = model(t)
    du = derivative(u, t)
    d2u = second_derivative(u, t)

    # Physics residual of the damped oscillator equation.
    f = m * d2u + mu * du + k * u

    # Initial conditions are enforced explicitly at t=0 instead of relying on
    # the sampled training grid, which excludes the endpoint.
    t0 = torch.zeros((1, 1), dtype=DTYPE, device=DEVICE).requires_grad_(True)
    u0 = model(t0)
    du0 = derivative(u0, t0)

    loss_ic_u = (u0 - 1.0) ** 2  # u(0) = 1
    loss_ic_du = du0**2  # u'(0) = 0
    loss_f = torch.mean(f**2)

    return (
        loss_ic_u.squeeze(),
        loss_ic_du.squeeze(),
        loss_f.squeeze(),
    )


def train_oscillator_pinn(
    model: nn.Module,
    t_train: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    plot_every: int,
    out_dir: str,
    model_label: str,
    run_id: str,
    lambda1: float = LAMBDA1,
    lambda2: float = LAMBDA2,
    loss_fn: Callable[
        [nn.Module, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ] = oscillator_loss,
) -> None:
    """
    Train one DHO model and persist the loss trace plus diagnostic plots.

    DHO is cheap enough that this implementation keeps the optimization fully
    deterministic and full-batch: every epoch uses the same time grid.
    """

    os.makedirs(out_dir, exist_ok=True)

    png_path = os.path.join(out_dir, f"dho-{model_label}_{run_id}.png")
    csv_path = os.path.join(out_dir, f"dho-{model_label}_{run_id}.csv")

    start = datetime.now()
    rows = []
    # Intermediate plots are saved at fixed milestones so training trajectories
    # can be compared across branch types with the same optimizer settings.
    snapshot_epochs = {600, 1200}

    def save_prediction_png(epoch: int, elapsed_s: float) -> str:
        with torch.no_grad():
            t_np = t_train.squeeze().cpu().numpy()
            u_pred = model(t_train).cpu().numpy().flatten()
            u_ex = u_exact(t_np)

        epoch_png_path = os.path.join(
            out_dir, f"dho-{model_label}_{run_id}_epoch-{epoch}.png"
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t_np, u_pred, label=f"PINN ({model_label})")
        ax.plot(t_np, u_ex, "--", label="Exact")
        ax.legend()
        ax.set_xlabel("t")
        ax.set_ylabel("u(t)")
        ax.set_title(f"{elapsed_s:.2f}s - {epoch} epochs")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(epoch_png_path, bbox_inches="tight")
        plt.close(fig)
        return epoch_png_path

    # Full-batch training loop for the Appendix A.2 oscillator.
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        lic_u, lic_du, lf = loss_fn(model, t_train)
        loss = lic_u + lambda1 * lic_du + lambda2 * lf

        loss.backward()
        optimizer.step()
        elapsed = (datetime.now() - start).total_seconds()

        if epoch % plot_every == 0:
            print(f"Epoch {epoch:4d} | Elapsed: {elapsed:.2f}seconds")
            print(
                f"  Loss={loss.item():.4e} | "
                f"IC_u={lic_u:.4e} | IC_du={lic_du:.4e} | PDE={lf:.4e}"
            )

            # Persist the decomposed loss so the summary tables can report how
            # much error comes from the IC terms versus the physics residual.
            rows.append(
                [
                    epoch,
                    f"{elapsed:.2f}",
                    f"{loss.item():.4e}",
                    f"{lic_u:.4e}",
                    f"{lic_du:.4e}",
                    f"{lf:.4e}",
                ]
            )

        if epoch in snapshot_epochs:
            epoch_png_path = save_prediction_png(epoch=epoch, elapsed_s=elapsed)
            print(f"PNG snapshot saved to: {epoch_png_path}")

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "elapsed time (s)",
                "Loss",
                "IC_u",
                "IC_du",
                "PDE",
            ]
        )
        writer.writerows(rows)

    stop = datetime.now()
    elapsed = (stop - start).total_seconds()

    # Save the final prediction profile using the same helper as the snapshots.
    final_png_path = save_prediction_png(epoch=n_epochs - 1, elapsed_s=elapsed)
    if final_png_path != png_path:
        os.replace(final_png_path, png_path)

    print(f"\nCSV saved to: {csv_path}")
    print(f"PNG saved to: {png_path}")


def load_training_row_for_run_id(
    out_dir: str,
    model_label: str,
    run_id: str,
) -> dict[str, str] | None:
    """Return the last row from the detailed CSV for a given model/run_id pair."""
    csv_path = os.path.join(out_dir, f"dho-{model_label}_{run_id}.csv")
    if not os.path.isfile(csv_path):
        return None

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    return rows[-1]


def get_run_id_from_checkpoint(ckpt_path: str, case_prefix: str) -> str | None:
    """Extract the run_id encoded in a checkpoint filename."""
    ckpt_name = os.path.basename(ckpt_path)
    ckpt_prefix = f"{case_prefix}_"
    ckpt_suffix = ".pt"
    if not (ckpt_name.startswith(ckpt_prefix) and ckpt_name.endswith(ckpt_suffix)):
        return None

    run_id = ckpt_name[len(ckpt_prefix) : -len(ckpt_suffix)]
    return run_id or None


def append_summary_row(summary_path: str, row: dict[str, object]) -> bool:
    """
    Append one normalized DHO summary row, writing the header on first use.

    Returns True when the same `(run_id, Model, Size)` triplet was already
    present before the append, and False otherwise.
    """
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    write_header = (
        not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0
    )

    is_duplicate = False
    if not write_header:
        with open(summary_path, newline="") as f:
            existing_rows = list(csv.DictReader(f))
        row_key = (
            str(row.get("run_id", "")),
            str(row.get("Model", "")),
            str(row.get("Size", "")),
        )
        for existing in existing_rows:
            existing_key = (
                existing.get("run_id", ""),
                existing.get("Model", ""),
                existing.get("Size", ""),
            )
            if existing_key == row_key:
                is_duplicate = True
                break

    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DHO_SUMMARY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in DHO_SUMMARY_COLUMNS})
    return is_duplicate
