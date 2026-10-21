"""Fair comparison figure from ``results/fair_comparison.json`` (corrected protocol, kernel SVM).

Usage: ``.venv/bin/python papers/distributed_secure_kernels/utils/plot_fair_comparison.py <report json> <output dir>``
The figure is saved in ``<output dir>`` and copied to ``results/``.
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

ORDER = [
    ("quantum_distributed", "distributed quantum", OURS, "o"),
    ("quantum_centralised", "centralised quantum", OURS, "D"),
    ("merlin", "MerLin photonic (adaptation)", OURS, "s"),
    ("exact_squared_overlap", "exact squared overlap", EXACT, "o"),
    ("exact_cosine", "exact cosine", EXACT, "D"),
    ("classical_linear", "linear SVM (paper baseline)", PAPER, "o"),
    ("tuned_rbf", "tuned RBF SVM (fair)", PAPER, "D"),
    ("tuned_poly", "tuned polynomial SVM (fair)", PAPER, "s"),
    ("majority_class", "majority class", REFERENCE, "|"),
]


def main(report_path: Path, out_dir: Path) -> None:
    report = json.loads(report_path.read_text())
    rows = {
        (s["dataset"], s["system"]): s
        for s in report["systems"]
        if s["model"] == "kernel_svm" and "accuracy" in s
    }
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), sharey=True)
    for col, (dataset, title) in enumerate(
        (("wine", "Wine"), ("parkinsons", "Parkinson's"), ("framingham", "Framingham"))
    ):
        for row, (metric, label) in enumerate(
            (("accuracy", "accuracy"), ("balanced_accuracy", "balanced accuracy"))
        ):
            ax = axes[row, col]
            for y, (key, _name, color, marker) in enumerate(ORDER):
                entry = rows.get((dataset, key))
                if entry is None:
                    ax.text(
                        0.02,
                        y,
                        "not run (identical kernel)",
                        color=MUTED,
                        fontsize=7,
                        va="center",
                        transform=ax.get_yaxis_transform(),
                    )
                    continue
                ax.errorbar(
                    entry[metric],
                    y,
                    xerr=entry[f"{metric}_seed_std"],
                    fmt=marker,
                    color=color,
                    markersize=7,
                    capsize=3,
                    linewidth=1.5,
                )
            ax.set_yticks(
                range(len(ORDER)),
                [name for _, name, _, _ in ORDER],
                color=INK,
                fontsize=8,
            )
            ax.invert_yaxis()
            ax.set_xlabel(f"5-fold CV {label}", color=MUTED)
            if row == 0:
                ax.set_title(title, color=INK, loc="left", fontsize=11)
            style_axes(ax)
    fig.suptitle(
        "Under a leak-free protocol, tuned classical SVMs beat the quantum kernel; distributed = centralised = MerLin",
        color=INK,
        x=0.01,
        ha="left",
        fontsize=12,
    )
    fig.text(
        0.01,
        -0.02,
        "Corrected protocol (oversampling and selection inside training folds), kernel SVM, mean ± std over 5 seeds; "
        "blue: paper method and MerLin adaptation (simulated), aqua: exact kernels, orange: classical baselines, gray: majority class. "
        "Framingham: high accuracy = majority-class prediction (majority rate 0.847).",
        color=MUTED,
        fontsize=8,
        wrap=True,
    )
    fig.tight_layout()
    for path in save(fig, "fair_comparison", out_dir):
        print(path)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
