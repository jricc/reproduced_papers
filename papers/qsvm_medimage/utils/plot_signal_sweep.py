"""Plot the decisive fairness figure from a signal-sweep CSV.

Two panels vs signal strength: (left) minority-class F1, (right) test AUC, for the
untuned QSVM, the untuned linear SVM (C=1), and a fair class-weighted linear SVM.

The story: in the weak-signal regime the untuned linear SVM collapses (F1=0) while QSVM
does not — but their AUC curves coincide (same discrimination), and the fair class-weighted
linear baseline matches/beats QSVM on both metrics.  The F1 "advantage" is a thresholding
artifact, not quantum discrimination.

Usage:
    python utils/plot_signal_sweep.py --csv results/signal_sweep_q11.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SHOW = {
    "qsvm": "QSVM (C=1)",
    "linear_c1": "linear C=1 (paper Tier-1 baseline)",
    "linear_balanced": "linear balanced (fair baseline)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    out = Path(args.out) if args.out else Path(args.csv).with_name(
        Path(args.csv).stem + "_fig.png")

    fig, (axf, axa) = plt.subplots(1, 2, figsize=(12, 4.5))
    for m, lab in SHOW.items():
        sub = df[df.method == m].sort_values("signal")
        axf.errorbar(sub["signal"], sub["f1_mean"], yerr=sub.get("f1_std"),
                     marker="o", capsize=2, label=lab)
        axa.errorbar(sub["signal"], sub["auc_mean"], yerr=sub.get("auc_std"),
                     marker="o", capsize=2, label=lab)
    q = int(df["q"].iloc[0])
    axf.set_title(f"Minority-class F1 vs signal (q={q})")
    axf.set_xlabel("task signal strength")
    axf.set_ylabel("minority F1")
    axf.grid(alpha=0.3)
    axf.legend(fontsize=8)
    axa.axhline(0.5, ls="--", c="gray", lw=1, label="chance")
    axa.set_title(f"Test AUC vs signal (q={q})")
    axa.set_xlabel("task signal strength")
    axa.set_ylabel("test AUC")
    axa.grid(alpha=0.3)
    axa.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
