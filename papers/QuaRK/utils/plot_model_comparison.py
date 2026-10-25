"""Normalized test MSE of every model of the main comparison, one panel per paper task.

Each dot is one seed; the bar marks the mean over seeds (as in the tables). Values are
normalized by each seed's test-label variance, which leaves the order of the models
within a seed unchanged. Values below ``FLOOR`` (exact fits of the linear ridge) are
drawn at the floor with a left-pointing marker.

Usage (from the repository root)::

    python papers/QuaRK/utils/plot_model_comparison.py \\
        papers/QuaRK/outdir/sweep_main_comparison --label "<scope>"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from plot_style import (
    PAPER_TASKS,
    TASK_LABELS,
    TEXT_SECONDARY,
    apply_style,
    load_sweep,
    save,
    selected,
)

MODELS = (
    ("quark", "QuaRK (paper settings)", "quantum"),
    ("quark_exact", "QuaRK, exact expectations", "quantum"),
    ("quark_shadows", "QuaRK, random-Pauli shadows", "quantum"),
    ("raw_window_kernel", "Matérn kernel, raw window", "classical"),
    ("raw_window_linear", "Linear ridge, raw window", "classical"),
    ("esn_default", "Echo state network, default", "classical"),
    ("esn_tuned", "Echo state network, tuned", "classical"),
)
FAMILY_COLORS = {"quantum": "#2a78d6", "classical": "#eb6834"}
FAMILY_LABELS = {"quantum": "QuaRK variants", "classical": "Classical baselines (ours)"}
FLOOR = 1e-14


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    apply_style()
    runs = load_sweep(args.sweep_dir)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    rows = np.arange(len(MODELS))[::-1]
    for ax, task in zip(axes, PAPER_TASKS):
        for row, (candidate, _name, family) in zip(rows, MODELS):
            values = np.array(
                [
                    selected(r["metrics"], task, "normalized_mse_test")
                    for r in runs
                    if r["candidate"] == candidate
                ]
            )
            color = FAMILY_COLORS[family]
            shown = values >= FLOOR
            ax.scatter(
                values[shown],
                np.full(shown.sum(), row),
                s=24,
                color=color,
                alpha=0.55,
                zorder=3,
            )
            hidden = int((~shown).sum())
            if hidden:
                ax.scatter([FLOOR], [row], s=30, marker="<", color=color, zorder=3)
                ax.annotate(
                    f"{hidden} seeds" if hidden > 1 else "1 seed",
                    (FLOOR, row),
                    xytext=(0, 7),
                    textcoords="offset points",
                    fontsize=7,
                    color=TEXT_SECONDARY,
                )
            ax.plot(
                [max(values.mean(), FLOOR)] * 2,
                [row - 0.32, row + 0.32],
                color=color,
                lw=2.5,
                zorder=4,
            )
        ax.set_xscale("log")
        ax.set_xlim(FLOOR / 3, 3)
        ax.set_title(TASK_LABELS[task], fontsize=9, loc="left")
        ax.set_yticks(rows, [name for _c, name, _f in MODELS], fontsize=8.5)
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.7, len(MODELS) - 0.3)
        ax.set_xlabel("Normalized test MSE (log scale)")
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            color=FAMILY_COLORS[f],
            label=FAMILY_LABELS[f],
        )
        for f in FAMILY_COLORS
    ]
    handles += [
        Line2D([], [], color=TEXT_SECONDARY, lw=2.5, label="Mean of 5 seeds"),
        Line2D(
            [],
            [],
            marker="<",
            ls="none",
            color=TEXT_SECONDARY,
            label="Below 10⁻¹⁴ (exact fit)",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(
        "Normalized test MSE (MSE / variance of the test labels), ridge chosen on "
        f"validation data · {args.label}",
        x=0.01,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    saved = save(fig, args.sweep_dir, "model_comparison")
    print("\n".join(str(p) for p in saved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
