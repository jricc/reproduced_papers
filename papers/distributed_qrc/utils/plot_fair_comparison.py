"""Plot the phase-05 fair comparison from ``results/fair_comparison.json``.

Per architecture and configuration, seed-mean test MAE (normalized; error bars =
sample std over seeds 0-2) of the eval-selected quantum/hybrid cell (R2), the
eval-selected fair classical cell, the eval-selected MerLin photonic cell
(adaptation), and the paper's baseline B0. The reservoir-free floors B1 (AR(4))
and B3 (product kernel on the raw lags) are horizontal lines. Log scale, because
B0 is 2-5x higher than the other systems. Every value is locally executed and
simulated; none is a paper-reported value.

Saves ``results/fair_comparison.png`` and a copy in every ``outdir/sweep_E*-CLASSICAL``.

Usage (from the repository root)::

    python papers/distributed_qrc/utils/plot_fair_comparison.py papers/distributed_qrc/outdir
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER_DIR = Path(__file__).resolve().parents[1]
ARCHS = {"srsr": "E1", "mrsr": "E2", "srmr": "E3", "mrmr": "E4"}
SELECTED = [
    ("quantum", "quantum/hybrid, eval-selected (R2)", "tab:blue"),
    ("fair_classical", "fair classical, eval-selected (B0-B4)", "tab:green"),
    ("merlin", "MerLin photonic, eval-selected (adaptation)", "tab:red"),
]


def main() -> int:
    outdir = Path(sys.argv[1])
    data = json.loads((PAPER_DIR / "results" / "fair_comparison.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, arch in zip(axes.flat, ARCHS):
        a = data["architectures"][arch]
        configs = list(a["eval_selected"])
        x = range(len(configs))
        for pool, label, color in SELECTED:
            values = [a["eval_selected"][c][pool]["test_mae"] for c in configs]
            ax.errorbar(
                x,
                [v["mean"] for v in values],
                yerr=[v["std"] for v in values],
                label=label,
                color=color,
                marker="o",
                markersize=3,
                capsize=2,
            )
        b0 = [a["per_configuration"][c]["classical|linear"] for c in configs]
        ax.errorbar(
            x,
            [v["mean"] for v in b0],
            yerr=[v["std"] for v in b0],
            label="paper baseline B0 (classical | linear)",
            color="gray",
            linestyle=":",
            marker="s",
            markersize=3,
            capsize=2,
        )
        for system, label, style in (
            (
                "raw_lags|quantum_kernel_ideal",
                "B3: product kernel on raw lags (no reservoir)",
                "--",
            ),
            ("raw_lags|linear", "B1: AR(4) linear ridge", "-."),
        ):
            value = a["per_configuration"][configs[0]][system]["mean"]
            ax.axhline(
                value, color="black", linestyle=style, linewidth=0.9, label=label
            )
        ax.set_yscale("log")
        ax.set_xticks(list(x), configs, rotation=30, fontsize=8)
        ax.set_title(f"{arch.upper()} ({ARCHS[arch]})")
        ax.set_xlabel("configuration (matched total neurons and readout instances)")
        ax.set_ylabel("test MAE (normalized, log scale)")
        ax.grid(alpha=0.3, which="both")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(
        "Phase 05 fair comparison: locally executed simulations, seeds 0-2, mean ± std; "
        "selection on the evaluation split\nMatching: same data, split, features/neurons, readout instances and λ protocol; MerLin is an adaptation",
        fontsize=10,
    )
    fig.tight_layout()
    targets = [PAPER_DIR / "results" / "fair_comparison.png"]
    targets += [
        outdir / f"sweep_{exp}-CLASSICAL" / "fair_comparison.png"
        for exp in ARCHS.values()
        if (outdir / f"sweep_{exp}-CLASSICAL").is_dir()
    ]
    for path in targets:
        fig.savefig(path, dpi=130)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
