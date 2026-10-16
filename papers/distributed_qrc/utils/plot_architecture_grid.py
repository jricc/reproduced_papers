"""Per-architecture MAE / RMSE figure (analogue of the paper's Figs. 4-7).

For every configuration of one architecture, plots the test MAE and RMSE of the
9 cells: our mean +/- std over seeds (filled markers with error bars) and the
paper's single-run value (open markers). Colour = readout, marker = reservoir.

Reads ``<sweep_dir>/summary.json``, ``<sweep_dir>/sweep_status.json`` and
``results/paper_tables.json``; writes ``<sweep_dir>/<id>_grid.png`` and
``results/<id>_grid.png``.

Usage (from the repository root)::

    python papers/distributed_qrc/utils/plot_architecture_grid.py papers/distributed_qrc/outdir/sweep_E1
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PAPER_DIR = Path(__file__).resolve().parents[1]
READOUT_COLOURS = {
    "linear": "#4C72B0",
    "quantum_kernel_ideal": "#DD8452",
    "quantum_kernel_noisy": "#C44E52",
}
RESERVOIR_MARKERS = {"classical": "s", "quantum_ideal": "o", "quantum_noisy": "^"}
TITLES = {
    "srsr": "SRSR (E1)",
    "mrsr": "MRSR (E2)",
    "srmr": "SRMR (E3)",
    "mrmr": "MRMR (E4)",
}


def main() -> int:
    sweep_dir = Path(sys.argv[1])
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{sweep_dir} is {status['status']}, not COMPLETED")
    summary = json.loads((sweep_dir / "summary.json").read_text())
    metrics = json.loads((sweep_dir / status["runs"][0]["metrics_path"]).read_text())
    architecture = metrics["architecture"]
    paper = {
        (c["configuration"], c["reservoir_type"], c["readout_type"]): c
        for c in json.loads((PAPER_DIR / "results" / "paper_tables.json").read_text())[
            "cells"
        ]
        if c["architecture"] == architecture
    }
    cells = summary["cells"]["default"]

    def total_neurons(label: str) -> tuple[int, str]:
        if "x" in label:
            count, size = label.split("x")
            return int(count) * int(size), label
        return int(label.split("/")[0]), label

    configs = sorted(
        dict.fromkeys(key.split("|")[0] for key in cells), key=total_neurons
    )
    n_seeds = next(iter(cells.values()))["n_seeds"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    for ax, metric in zip(axes, ("mae", "rmse")):
        for r_index, reservoir in enumerate(RESERVOIR_MARKERS):
            for o_index, readout in enumerate(READOUT_COLOURS):
                offset = (r_index * 3 + o_index - 4) * 0.08
                xs = [i + offset for i in range(len(configs))]
                ours = [
                    cells[f"{c}|{reservoir}|{readout}"][f"test.{metric}"]
                    for c in configs
                ]
                ax.errorbar(
                    xs,
                    [o["mean"] for o in ours],
                    yerr=[o["std"] for o in ours],
                    fmt=RESERVOIR_MARKERS[reservoir],
                    color=READOUT_COLOURS[readout],
                    ms=5,
                    capsize=2,
                    lw=1,
                    label=f"ours: {reservoir} + {readout}",
                )
                ax.scatter(
                    xs,
                    [paper[(c, reservoir, readout)][metric] for c in configs],
                    marker=RESERVOIR_MARKERS[reservoir],
                    facecolors="none",
                    edgecolors=READOUT_COLOURS[readout],
                    s=40,
                    lw=1,
                )
        ax.set_xticks(range(len(configs)), configs)
        ax.set_xlabel(
            "configuration (neurons, or reservoirs x neurons, or neurons/readouts)"
        )
        ax.set_ylabel(f"test {metric.upper()} (normalized load)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7, ncol=1, loc="upper right")
    fig.suptitle(
        f"{TITLES[architecture]}: filled = this reproduction (mean ± std, {n_seeds} seeds, full data), "
        "open = paper Appendix C (single run)",
        fontsize=10,
    )
    fig.tight_layout()
    name = f"{summary['sweep_id']}_grid.png"
    for target in (sweep_dir / name, PAPER_DIR / "results" / name):
        fig.savefig(target, dpi=150)
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
