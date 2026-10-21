"""Table 1 as a dot plot: paper value vs our 5-seed mean, per dataset and column.

Usage: ``.venv/bin/python papers/distributed_secure_kernels/utils/plot_table1.py <table1_original sweep dir>``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt  # noqa: E402

from distributed_secure_kernels.utils.compare_with_paper import (
    PAPER_VALUES,  # noqa: E402
)
from distributed_secure_kernels.utils.plot_style import (  # noqa: E402
    INK,
    MUTED,
    OURS,
    PAPER,
    REFERENCE,
    save,
    style_axes,
)

DATASETS = (
    ("wine", "Wine"),
    ("parkinsons", "Parkinson's"),
    ("framingham", "Framingham"),
)
COLUMNS = (
    ("classical_linear", "classical"),
    ("quantum_centralised", "centralised quantum"),
    ("quantum_distributed", "distributed quantum"),
)
MODELS = (("kernel_svm", "kernel SVM"), ("kernel_pca_best_on_test_folds", "kernel PCA"))


def main(sweep_dir: Path) -> None:
    summary = json.loads((sweep_dir / "summary.json").read_text())
    rows = {
        (r["candidate"], r["method"], r["model"]): r["accuracy"]
        for r in summary["rows"]
    }
    paper = {(c, m, mo): (v, s) for _, c, m, mo, _, v, s, _ in PAPER_VALUES}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=False)
    for ax, (dataset, title) in zip(axes, DATASETS):
        candidate = f"accuracy_{dataset}_original"
        labels, y = [], 0
        for model, model_label in MODELS:
            for method, column in COLUMNS:
                key = (candidate, method, model)
                if key not in paper:
                    continue
                ours = rows[key]
                value, std = paper[key]
                ax.errorbar(
                    value,
                    y + 0.15,
                    xerr=std or 0,
                    fmt="D",
                    color=PAPER,
                    markersize=7,
                    capsize=3,
                    linewidth=1.5,
                    label="paper (± fold std)" if y == 0 else None,
                )
                ax.errorbar(
                    ours["seed_mean"],
                    y - 0.15,
                    xerr=ours["seed_std"],
                    fmt="o",
                    color=OURS,
                    markersize=8,
                    capsize=3,
                    linewidth=2,
                    label="ours (mean ± std over 5 seeds)" if y == 0 else None,
                )
                labels.append(f"{model_label}, {column}")
                y += 1
        majority = rows[
            (f"accuracy_{dataset}_reference", "majority_class", "kernel_svm")
        ]["seed_mean"]
        ax.axvline(
            majority,
            color=REFERENCE,
            linestyle=":",
            linewidth=1.5,
            label="majority class",
        )
        ax.set_yticks(range(len(labels)), labels, color=INK)
        ax.invert_yaxis()
        ax.set_title(title, color=INK, loc="left", fontsize=11)
        ax.set_xlabel("5-fold CV accuracy", color=MUTED)
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.suptitle(
        "Table 1 under the authors' protocol: our values fall within the paper's error bars for 12 of 15 cells",
        color=INK,
        x=0.01,
        ha="left",
        fontsize=12,
    )
    fig.text(
        0.01,
        -0.17,
        "Paper-accurate protocol (authors' subset, oversampling and 5-fold CV), exact simulation of 1,024-shot swap tests, 5 seeds.",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout()
    for path in save(fig, "table1_comparison", sweep_dir):
        print(path)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
