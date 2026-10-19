"""Headline-statistics figure: paper vs reproduction (average and peak error reductions, E7).

Reads ``results/data/headline_statistics.json`` (from ``utils/headline_statistics.py``)
and writes ``results/headline_reductions.png``. Bars: the paper's reported rule
applied to the paper's tables and to each of our seeds (mean +/- std).

Usage (from the repository root)::

    python papers/distributed_qrc/utils/plot_headline.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PAPER_DIR = Path(__file__).resolve().parents[1]
ARCHITECTURES = ("srsr", "mrsr", "srmr", "mrmr")


def main() -> int:
    data = json.loads(
        (PAPER_DIR / "results" / "data" / "headline_statistics.json").read_text()
    )
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
    panels = [
        ("mae", "average"),
        ("mae", "peak"),
        ("rmse", "average"),
        ("rmse", "peak"),
    ]
    x = np.arange(len(ARCHITECTURES))
    for ax, (metric, kind) in zip(axes, panels):
        paper = [100 * data["paper"][a][metric][kind] for a in ARCHITECTURES]
        ours = [
            100 * data["ours"][a]["paper_rule"][metric][f"{kind}_mean"]
            for a in ARCHITECTURES
        ]
        ours_std = [
            100 * data["ours"][a]["paper_rule"][metric][f"{kind}_std"]
            for a in ARCHITECTURES
        ]
        ax.bar(x - 0.2, paper, 0.4, color="#BBBBBB", label="paper (as printed)")
        ax.bar(
            x + 0.2,
            ours,
            0.4,
            yerr=ours_std,
            capsize=3,
            color="#4C72B0",
            label="ours (mean ± std, 3 seeds)",
        )
        ax.set_xticks(x, [a.upper() for a in ARCHITECTURES])
        ax.set_title(f"{metric.upper()} reduction, {kind}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("reduction vs classical reservoir + linear ridge (%)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Headline error reductions (paper abstract and Tables IV-VII), paper rule (best quantum/hybrid cell per configuration by test R²); "
        "full data, 3 seeds, full-n kernels",
        fontsize=10,
    )
    fig.tight_layout()
    out = PAPER_DIR / "results" / "headline_reductions.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {os.path.relpath(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
