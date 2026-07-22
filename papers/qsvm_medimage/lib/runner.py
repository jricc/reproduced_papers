"""Entry point: sweep classifiers across qubit counts and seeds, save results.

Reproduces the qubit-sweep / Tier-1 / Tier-2 comparisons of arXiv:2604.24597 on
substitute foundation-model embeddings (see ``lib/data.py``).  Each (method, q, seed)
row records minority-class F1, recall, accuracy, AUC, majority-class accuracy and the
kernel effective rank.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

import numpy as np

from .data import get_dataset
from .svm_pipeline import run_one


def train_and_evaluate(cfg, run_dir: Path) -> None:
    log = logging.getLogger(__name__)
    exp = cfg.get("experiment", {})
    q_list = exp.get("q_list", [2, 4, 6, 8, 10, 11, 12])
    seeds = exp.get("seeds", [0, 1, 2, 3, 4])
    circuit = exp.get("circuit", "bsp")
    reps = int(exp.get("reps", 1))
    classifiers = exp.get("classifiers")

    all_rows = []
    source = "synthetic"
    t0 = time.time()

    kernel_normalization = exp.get(
        "kernel_normalization",
        "trace",
    )

    for seed in seeds:
        (X, y), source = get_dataset(cfg, seed)
        log.info(
            "seed=%d source=%s X=%s pos_rate=%.3f",
            seed,
            source,
            X.shape,
            float(np.mean(y)),
        )
        for q in q_list:
            if q > X.shape[1]:
                continue
            t_q = time.time()

            rows = run_one(
                X,
                y,
                q,
                seed,
                circuit=circuit,
                reps=reps,
                classifiers=classifiers,
                kernel_normalization=kernel_normalization,
            )

            # rows = run_one(X, y, q, seed, circuit=circuit, reps=reps,
            #                classifiers=classifiers)
            for r in rows:
                r["data_source"] = source
            all_rows.extend(rows)
            f1s = {r["method"]: round(r["f1"], 3) for r in rows}
            log.info("  q=%2d (%.1fs) minority-F1: %s", q, time.time() - t_q, f1s)

    # --- persist raw rows -------------------------------------------------------------
    fields = sorted({k for r in all_rows for k in r})
    csv_path = run_dir / "results_long.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    # --- aggregate mean +/- std over seeds, per (method, q) ---------------------------
    agg = {}
    for r in all_rows:
        agg.setdefault((r["method"], r["q"]), []).append(r)
    summary = []
    metric_keys = [
        "f1",
        "recall",
        "precision",
        "accuracy",
        "auc",
        "majority_acc",
        "eff_rank",
        "predicted_minority_count",
        "true_minority_count",
        "collapse",
        "zero_f1",
    ]
    for (method, q), rs in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        row = {"method": method, "q": q, "n_seeds": len(rs)}
        for mk in metric_keys:
            vals = np.array([x[mk] for x in rs], dtype=float)
            finite = vals[np.isfinite(vals)]
            if len(finite) == 0:
                row[f"{mk}_mean"] = float("nan")
                row[f"{mk}_std"] = float("nan")
            else:
                row[f"{mk}_mean"] = float(np.mean(finite))
                row[f"{mk}_std"] = float(np.std(finite))
        summary.append(row)
    sfields = ["method", "q", "n_seeds"] + [
        f"{mk}_{s}" for mk in metric_keys for s in ("mean", "std")
    ]
    with (run_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        w.writerows(summary)

    # --- collapse rates (fraction of seeds with majority-only / zero-F1 predictions) ---
    collapse = {}
    for r in all_rows:
        key = (r["method"], r["q"])
        collapse.setdefault(key, []).append(bool(r.get("collapse", False)))
    collapse_rows = [
        {
            "method": m,
            "q": q,
            "collapse_rate": float(np.mean(v)),
            "n_seeds": len(v),
        }
        for (m, q), v in sorted(collapse.items())
    ]
    with (run_dir / "collapse_rates.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "q", "collapse_rate", "n_seeds"])
        w.writeheader()
        w.writerows(collapse_rows)

    meta = {
        "n_rows": len(all_rows),
        "q_list": q_list,
        "seeds": seeds,
        "circuit": circuit,
        "reps": reps,
        "data_source": source,
        "wall_clock_s": round(time.time() - t0, 1),
        "kernel_normalization": kernel_normalization,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("Done in %.1fs. Rows=%d -> %s", time.time() - t0, len(all_rows), run_dir)
