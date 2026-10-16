"""Plot the phase-04 MerLin adaptation against the R2 gate-model cells.

Reads ``results/merlin_comparison.json`` (written by ``utils/merlin_comparison.py``)
and draws, per architecture, the seed-mean test MAE (normalized, error bars =
sample std over seeds 0-2) of the full-quantum and reservoir-only cells, with the
classical|linear baseline. Saves ``results/merlin_comparison.png`` and a copy in
each ``outdir/sweep_E*-MERLIN`` folder passed as the first argument's children.

Usage (from the repository root)::

    python papers/distributed_qrc/utils/plot_merlin_comparison.py papers/distributed_qrc/outdir
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
SERIES = [
    ("a_full_quantum", "r2", "gate: Q-ideal | kernel-ideal (R2)", "tab:blue", "-"),
    ("a_full_quantum", "merlin", "photonic | photonic kernel (MerLin)", "tab:red", "-"),
    ("c_reservoir_linear", "r2", "gate: Q-ideal | linear (R2)", "tab:blue", "--"),
    ("c_reservoir_linear", "merlin", "photonic | linear (MerLin)", "tab:red", "--"),
]


def main() -> int:
    outdir = Path(sys.argv[1])
    data = json.loads((PAPER_DIR / "results" / "merlin_comparison.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    for ax, arch in zip(axes.flat, ARCHS):
        a = data[arch]
        ax.set_title(f"{arch.upper()} ({ARCHS[arch]})")
        if a["status"] != "COMPLETED":
            ax.text(0.5, 0.5, "not available", ha="center", transform=ax.transAxes)
            continue
        for comparison, side, label, color, style in SERIES:
            rows = a[comparison]["configurations"]
            configs = list(rows)
            ax.errorbar(
                range(len(configs)),
                [rows[c][f"{side}_mean"] for c in configs],
                yerr=[rows[c][f"{side}_std"] for c in configs],
                label=label,
                color=color,
                linestyle=style,
                marker="o",
                markersize=3,
                capsize=2,
            )
        ax.set_xticks(range(len(configs)), configs, rotation=30, fontsize=8)
        ax.set_xlabel("configuration")
        ax.set_ylabel("test MAE (normalized)")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(
        "MerLin photonic adaptation vs gate-model reproduction (R2): simulation, "
        "seeds 0-2, mean ± std\nAdaptation, not the paper's method; noise cells excluded",
        fontsize=10,
    )
    fig.tight_layout()
    targets = [PAPER_DIR / "results" / "merlin_comparison.png"]
    targets += [
        outdir / f"sweep_{exp}-MERLIN" / "merlin_comparison.png"
        for exp in ARCHS.values()
        if (outdir / f"sweep_{exp}-MERLIN").is_dir()
    ]
    for path in targets:
        fig.savefig(path, dpi=130)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
