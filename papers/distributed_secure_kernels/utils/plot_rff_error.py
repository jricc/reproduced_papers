"""Fig. 5 counterpart: RFF relative Frobenius error vs D (RBF and Laplacian; d = 10 and d = 100,000).

Usage: ``.venv/bin/python papers/distributed_secure_kernels/utils/plot_rff_error.py <d=10 run dir> <d=100000 run dir>``
The figure is saved in the first run directory and in ``results/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from distributed_secure_kernels.utils.plot_style import (
    INK,
    MUTED,
    REFERENCE,
    save,
    style_axes,
)  # noqa: E402

# Ordinal blue ramp (validated reference palette, steps 250 -> 700).
RAMP = (
    "#86b6ef",
    "#5598e7",
    "#2a78d6",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
    "#0a2a55",
    "#081f40",
)


def main(run_dirs: list[Path]) -> None:
    errors = {}
    for run_dir in run_dirs:
        for kind, per_dim in json.loads((run_dir / "metrics.json").read_text())[
            "errors"
        ].items():
            errors.setdefault(kind, {}).update(per_dim)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for col, (kind, symbol) in enumerate((("rbf", "σ"), ("laplacian", "α"))):
        for row, dimension in enumerate(("10", "100000")):
            ax = axes[row, col]
            per_param = errors[kind][dimension]
            for index, (param, per_d) in enumerate(per_param.items()):
                ds = sorted(int(d) for d in per_d)
                means = [np.mean(per_d[str(d)]) for d in ds]
                ax.plot(
                    ds,
                    means,
                    marker="o",
                    markersize=4,
                    linewidth=1.8,
                    color=RAMP[index % len(RAMP)],
                    label=f"{symbol} = {param}",
                )
            reference = means[0] * np.sqrt(ds[0] / np.array(ds))
            ax.plot(
                ds,
                reference,
                color=REFERENCE,
                linestyle="--",
                linewidth=1.2,
                label="∝ 1/√D",
            )
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_title(
                f"{'RBF' if kind == 'rbf' else 'Laplacian'} kernel, d = {int(dimension):,}",
                color=INK,
                loc="left",
                fontsize=10,
            )
            style_axes(ax)
            if row == 1:
                ax.set_xlabel(
                    "random Fourier features D (qubits = log2(2D))", color=MUTED
                )
            if col == 0:
                ax.set_ylabel("relative Frobenius error", color=MUTED)
            if row == 0:
                ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle(
        "The RFF error falls as 1/√D; at d = 100,000 the bandwidth no longer matters (as in Fig. 5)",
        color=INK,
        x=0.01,
        ha="left",
        fontsize=12,
    )
    fig.text(
        0.01,
        -0.02,
        "100 standard-normal samples, mean of 5 repetitions; paper-accurate settings of the authors' script.",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout()
    for path in save(fig, "rff_error", run_dirs[0]):
        print(path)


if __name__ == "__main__":
    main([Path(arg) for arg in sys.argv[1:]])
