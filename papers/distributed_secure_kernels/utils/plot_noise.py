"""Fig. 3 counterpart: kernel-SVM accuracy vs depolarizing rate, ours vs the authors' logs.

Usage::

    .venv/bin/python papers/distributed_secure_kernels/utils/plot_noise.py \
        <noise_original sweep dir> [<noise_decomposed_initialize sweep dir>]

The optional second sweep adds the diagnostic variant with noisy, decomposed
state preparation (older-Qiskit hypothesis).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt  # noqa: E402

from distributed_secure_kernels.utils.plot_style import (  # noqa: E402
    INK,
    MUTED,
    OURS,
    PAPER,
    REFERENCE,
    save,
    style_axes,
)

LEVELS = ("no noise", "0.1 %", "1 %")
# Authors' logged kernel-SVM accuracies (Table 1 distributed column for no noise;
# heart_dataset_kernel_1060336.txt / park_dataset_kernel_1055216.txt for noise).
AUTHORS = {
    "framingham": (0.6340, 0.5807, 0.5814),
    "parkinsons": (0.7983, 0.7873, 0.6330),
}
MAJORITY = {"framingham": 0.6249, "parkinsons": 0.6276}


def _rows(sweep_dir: Path) -> dict:
    summary = json.loads((sweep_dir / "summary.json").read_text())
    return {
        (r["candidate"], r["method"], r["model"]): r["accuracy"]
        for r in summary["rows"]
    }


def main(sweep_dir: Path, variant_dir: Path | None = None) -> None:
    rows = _rows(sweep_dir)
    variant = _rows(variant_dir) if variant_dir is not None else {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (dataset, title) in zip(
        axes, (("framingham", "Framingham"), ("parkinsons", "Parkinson's"))
    ):
        candidate = f"noise_{dataset}_original"
        x = range(3)
        ours = [
            rows[(candidate, f"quantum_distributed_noise_level_{level}", "kernel_svm")]
            for level in x
        ]
        ax.errorbar(
            x,
            [o["seed_mean"] for o in ours],
            yerr=[o["seed_std"] for o in ours],
            marker="o",
            markersize=8,
            color=OURS,
            linewidth=2,
            capsize=3,
            label="ours, noiseless state preparation (Qiskit 2.5)",
        )
        if variant:
            name = f"noise_{dataset}_decomposed_initialize"
            noisy_prep = [ours[0]] + [
                variant[
                    (
                        name,
                        f"quantum_distributed_noise_level_{level}_noisy_initialize",
                        "kernel_svm",
                    )
                ]
                for level in (1, 2)
            ]
            ax.errorbar(
                x,
                [o["seed_mean"] for o in noisy_prep],
                yerr=[o["seed_std"] for o in noisy_prep],
                marker="s",
                markersize=7,
                color=OURS,
                linewidth=1.5,
                linestyle="--",
                capsize=3,
                label="ours, noisy state preparation (older-Qiskit hypothesis)",
            )
        ax.plot(
            x,
            AUTHORS[dataset],
            marker="D",
            markersize=7,
            color=PAPER,
            linewidth=2,
            label="authors' logs",
        )
        classical = rows[(candidate, "classical_linear", "kernel_svm")]["seed_mean"]
        ax.axhline(
            classical,
            color=REFERENCE,
            linestyle="--",
            linewidth=1.5,
            label="classical linear SVM (ours)",
        )
        ax.axhline(
            MAJORITY[dataset],
            color=REFERENCE,
            linestyle=":",
            linewidth=1.5,
            label="majority class",
        )
        ax.set_xticks(list(x), LEVELS)
        ax.set_xlabel("depolarizing error rate per gate", color=MUTED)
        ax.set_title(title, color=INK, loc="left", fontsize=11)
        style_axes(ax)
    axes[0].set_ylabel("5-fold CV accuracy (kernel SVM)", color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.12),
    )
    title = (
        "Noisy state preparation brings the Parkinson's curve close to the authors'; their Framingham curve stays unexplained"
        if variant
        else "Noise lowers accuracy in both, but much less in our simulation than in the authors' logs"
    )
    fig.suptitle(title, color=INK, x=0.01, ha="left", fontsize=11)
    fig.text(
        0.01,
        -0.22,
        "Paper-accurate protocol; authors' noise model (depolarizing on u1/u2/u3/cx); exact noisy probabilities + 1,024 shots, 5 seeds.",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout()
    for path in save(fig, "noise_comparison", sweep_dir):
        print(path)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None)
