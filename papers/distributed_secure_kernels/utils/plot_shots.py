"""Fig. 4 counterpart: Digits kernel-SVM accuracy vs number of shots.

Usage: ``.venv/bin/python papers/distributed_secure_kernels/utils/plot_shots.py <shots_original sweep dir>``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt  # noqa: E402

from distributed_secure_kernels.utils.plot_style import (  # noqa: E402
    EXACT,
    INK,
    MUTED,
    OURS,
    PAPER,
    REFERENCE,
    save,
    style_axes,
)

SHOTS = (128, 256, 512, 1024)
PAPER_FIG4 = (0.71, 0.71, 0.74, 0.83)  # Fig. 4 labels, identical to the authors' logs


def main(sweep_dir: Path) -> None:
    summary = json.loads((sweep_dir / "summary.json").read_text())
    rows = {(r["method"], r["model"]): r["accuracy"] for r in summary["rows"]}
    ours = [rows[(f"quantum_distributed_{s}_shots", "kernel_svm")] for s in SHOTS]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(
        SHOTS,
        [o["seed_mean"] for o in ours],
        yerr=[o["seed_std"] for o in ours],
        marker="o",
        markersize=8,
        color=OURS,
        linewidth=2,
        capsize=3,
        label="ours (mean ± std over 5 seeds)",
    )
    ax.plot(
        SHOTS,
        PAPER_FIG4,
        marker="D",
        markersize=7,
        color=PAPER,
        linewidth=2,
        label="paper Fig. 4",
    )
    ax.axhline(
        rows[("exact_squared_overlap", "kernel_svm")]["seed_mean"],
        color=EXACT,
        linestyle="-.",
        linewidth=1.5,
        marker="s",
        markevery=[0],
        label="exact kernel (infinite shots)",
    )
    ax.axhline(
        rows[("classical_linear", "kernel_svm")]["seed_mean"],
        color=REFERENCE,
        linestyle="--",
        linewidth=1.5,
        label="classical linear SVM",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(SHOTS, [str(s) for s in SHOTS])
    ax.set_xlabel("shots per kernel entry", color=MUTED)
    ax.set_ylabel("5-fold CV accuracy (kernel SVM)", color=MUTED)
    ax.legend(
        fontsize=8,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
    )
    style_axes(ax)
    ax.set_title(
        "With the authors' preprocessing, accuracy changes little with shots",
        color=INK,
        loc="left",
        fontsize=11,
    )
    fig.text(
        0.01,
        -0.04,
        "Digits, 100-sample stratified subset, 10 classes; paper-accurate protocol, exact simulation, 5 seeds.",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout()
    for path in save(fig, "shots_comparison", sweep_dir):
        print(path)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
