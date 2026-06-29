"""Plot the q-sweep summary from a run directory.

Produces, for a given ``--run-dir`` (containing summary.csv):
  * f1_vs_q.png        minority-class F1 vs qubit count, per classifier
  * auc_vs_q.png       test AUC vs qubit count, per classifier
  * effrank_vs_q.png   kernel effective rank vs qubit count (concentration story)
  * f1_auc_bar_q<Q>.png  paired F1-vs-AUC bars at a chosen q (the artifact figure)

Figures are saved both into the run directory and into ``results/``.
Usage:
    python utils/plot_summary.py --run-dir outdir/run_XXXX --highlight-q 11
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LABELS = {
    "qsvm": "QSVM (BSP, C=1)",
    "linear_c1": "linear SVM C=1 (Tier-1)",
    "rbf_c1": "RBF C=1 (Tier-2 default)",
    "rbf_rank_matched": "RBF rank-matched (Tier-2)",
    "linear_balanced": "linear balanced (fair)",
    "linear_tuned": "linear tuned+balanced (fair)",
}
ORDER = ["qsvm", "linear_c1", "rbf_c1", "rbf_rank_matched", "linear_balanced", "linear_tuned"]


def _save(fig, name, run_dir, results_dir):
    for d in (run_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _line(df, metric, ylabel, title, name, run_dir, results_dir, hline=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in ORDER:
        sub = df[df.method == m].sort_values("q")
        if sub.empty:
            continue
        ax.errorbar(sub["q"], sub[f"{metric}_mean"], yerr=sub.get(f"{metric}_std"),
                    marker="o", capsize=2, label=LABELS.get(m, m))
    if hline is not None:
        ax.axhline(hline, ls="--", c="gray", lw=1, label="chance / majority")
    ax.set_xlabel("qubits / PCA dimension q")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, name, run_dir, results_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--highlight-q", type=int, default=11)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    results_dir = Path(args.results_dir)
    df = pd.read_csv(run_dir / "summary.csv")

    _line(df, "f1", "minority-class F1", "Minority-class F1 vs q (substitute embeddings)",
          "f1_vs_q.png", run_dir, results_dir, hline=0.0)
    _line(df, "auc", "test AUC", "Test AUC vs q  (0.5 = no discrimination)",
          "auc_vs_q.png", run_dir, results_dir, hline=0.5)
    _line(df, "eff_rank", "kernel effective rank", "Kernel effective rank vs q",
          "effrank_vs_q.png", run_dir, results_dir)

    # Artifact figure: at highlight q, paired F1 and AUC bars
    q = args.highlight_q
    sub = df[df.q == q].set_index("method")
    methods = [m for m in ORDER if m in sub.index]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    import numpy as np
    x = np.arange(len(methods))
    w = 0.38
    ax.bar(x - w / 2, [sub.loc[m, "f1_mean"] for m in methods], w, label="minority F1",
           yerr=[sub.loc[m, "f1_std"] for m in methods], capsize=2)
    ax.bar(x + w / 2, [sub.loc[m, "auc_mean"] for m in methods], w, label="test AUC",
           yerr=[sub.loc[m, "auc_std"] for m in methods], capsize=2)
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(m, m) for m in methods], rotation=30, ha="right", fontsize=8)
    ax.set_title(f"q={q}: minority-F1 separates but AUC does not (artifact)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    _save(fig, f"f1_auc_bar_q{q}.png", run_dir, results_dir)
    print(f"Saved figures to {run_dir} and {results_dir}")


if __name__ == "__main__":
    main()
