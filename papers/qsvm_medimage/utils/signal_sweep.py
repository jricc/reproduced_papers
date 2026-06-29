"""Signal-strength sweep at fixed q: the decisive fairness experiment.

Maps minority-F1 AND AUC for each classifier as the task goes from unpredictable
(signal=0, the insurance regime) to genuinely learnable (signal large).  The key question:
in any regime where the untuned QSVM beats the untuned linear SVM (C=1) on minority-F1,
does it also beat it on **AUC**?  If F1 separates but AUC does not, the "advantage" is a
threshold / regularization artifact rather than genuine quantum discrimination.

Usage:
    python utils/signal_sweep.py --q 11 --seeds 0,1,2,3,4 \
        --signals 0,0.25,0.5,1,2 --out results/signal_sweep_q11.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root (for runtime_lib)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project dir (for lib)
from lib.data import make_synthetic_embeddings  # noqa: E402
from lib.svm_pipeline import run_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, default=11)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--signals", default="0,0.25,0.5,1,2")
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--ambient-dim", type=int, default=768)
    ap.add_argument("--minority-fraction", type=float, default=0.28)
    ap.add_argument("--out", default="results/signal_sweep.csv")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    signals = [float(s) for s in args.signals.split(",")]
    rows = []
    for sig in signals:
        for seed in seeds:
            X, y = make_synthetic_embeddings(
                n_samples=args.n_samples, ambient_dim=args.ambient_dim,
                minority_fraction=args.minority_fraction, signal=sig, seed=seed)
            for r in run_one(X, y, q=args.q, seed=seed):
                r["signal"] = sig
                rows.append(r)
        print(f"signal={sig} done")

    # aggregate
    agg = {}
    for r in rows:
        agg.setdefault((r["signal"], r["method"]), []).append(r)
    out_rows = []
    for (sig, method), rs in sorted(agg.items()):
        out_rows.append({
            "signal": sig, "method": method, "q": args.q, "n_seeds": len(rs),
            "f1_mean": float(np.mean([x["f1"] for x in rs])),
            "f1_std": float(np.std([x["f1"] for x in rs])),
            "auc_mean": float(np.nanmean([x["auc"] for x in rs])),
            "auc_std": float(np.nanstd([x["auc"] for x in rs])),
            "recall_mean": float(np.mean([x["recall"] for x in rs])),
            "accuracy_mean": float(np.mean([x["accuracy"] for x in rs])),
            "collapse_rate": float(np.mean([x["f1"] < 0.05 for x in rs])),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWrote {out}")
    # pretty print F1 vs AUC for qsvm and linear_c1
    import pandas as pd
    df = pd.DataFrame(out_rows)
    for m in ["qsvm", "linear_c1", "linear_balanced"]:
        print(f"\n{m}:")
        print(df[df.method == m][["signal", "f1_mean", "auc_mean", "recall_mean", "collapse_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
