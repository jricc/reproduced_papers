#!/usr/bin/env python3
"""Photonic (MerLin / linear-optical) variants of the core surrogate artifacts.

The paper's QSVM uses a qubit gate-based fidelity kernel. This driver recomputes
the central claims with the *photonic* fidelity kernel from ``lib.photonic_kernel``
(a two-photon linear-optical translation of the BSP feature map) so the photonic
and qubit kernels can be compared side by side on the same surrogate data.

It emits three photonic artifacts under ``results/`` (prefix ``photonic_*``):

1. ``photonic_table1``      -- Tier-1 shape (Table 1): mean F1 for the collapsing
                               linear SVM vs the qubit QSVM vs the photonic QSVM,
                               per (model, q).
2. ``photonic_figure4``     -- class-sorted photonic Gram matrix heatmap (Figure 4).
3. ``photonic_effrank``     -- photonic-kernel effective rank vs q (Figure 2 / Table 5
                               eigenspectrum claim).

Photonic SLOS simulation is O(N^2); train/test are subsampled (``--train-cap`` /
``--test-cap``) and this is stated in every output. These are illustrative photonic
counterparts, not paper-number reproductions (the real embeddings are gated).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = PROJECT_ROOT.parents[1]
for root in (PROJECT_ROOT, REPRO_ROOT, PROJECT_ROOT / "utils"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from lib.photonic_kernel import photonic_fidelity_kernels  # noqa: E402
from lib.quantum_kernel import fidelity_kernel  # noqa: E402
from lib.svm_pipeline import preprocess, split_indices  # noqa: E402
from synthetic_surrogate_table1 import (  # noqa: E402
    SyntheticSpec,
    compute_metrics,
    decision_scores,
    load_dataset,
    parse_ints,
)

# Photonic q-sweep per model (kept modest: two-photon SLOS over q modes).
PHOTONIC_CONFIGS: tuple[tuple[str, int], ...] = (
    ("medsiglip-448", 4),
    ("medsiglip-448", 6),
    ("medsiglip-448", 8),
    ("medsiglip-448", 10),
    ("rad-dino", 4),
    ("rad-dino", 6),
    ("rad-dino", 8),
    ("vit-patch32-cls", 4),
    ("vit-patch32-cls", 6),
    ("vit-patch32-cls", 8),
)

MODEL_SEED_OFFSETS = {
    "medsiglip-448": 0,
    "rad-dino": 10_000,
    "vit-patch32-cls": 20_000,
}

PAPER_POINTERS = {
    "table1": "https://arxiv.org/html/2604.24597v1#S4.T1",
    "figure4": "https://arxiv.org/html/2604.24597v1#S4.F4",
    "effrank": "https://arxiv.org/html/2604.24597v1#S4.F2",
}


def subsample(idx: np.ndarray, y: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Stratified cap of an index array so photonic SLOS stays affordable."""
    if cap <= 0 or len(idx) <= cap:
        return idx
    rng = np.random.default_rng(seed)
    labels = y[idx]
    keep: list[int] = []
    classes, counts = np.unique(labels, return_counts=True)
    for cls, cnt in zip(classes, counts):
        cls_idx = idx[labels == cls]
        take = max(1, int(round(cap * cnt / len(idx))))
        take = min(take, len(cls_idx))
        keep.extend(rng.choice(cls_idx, size=take, replace=False).tolist())
    return np.array(sorted(keep))


def effective_rank(kernel: np.ndarray) -> float:
    """exp(von Neumann entropy) of the trace-normalized kernel spectrum."""
    eig = np.linalg.eigvalsh(kernel)
    eig = eig[eig > 1e-12]
    if eig.size == 0:
        return 0.0
    p = eig / eig.sum()
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def prepare_split(
    *, source, model, q, seed, data_root, synthetic, train_cap, test_cap
):
    X, y = load_dataset(
        source=source,
        model=model,
        seed=seed + MODEL_SEED_OFFSETS.get(model, 0) if source == "synthetic" else seed,
        data_root=data_root,
        synthetic=synthetic,
    )
    idx_train, _idx_val, idx_test = split_indices(y, seed=seed)
    idx_train = subsample(idx_train, y, train_cap, seed)
    idx_test = subsample(idx_test, y, test_cap, seed + 1)
    X_train, _Xv, X_test, evr = preprocess(X[idx_train], X[idx_train], X[idx_test], q)
    return X_train, X_test, y[idx_train], y[idx_test], evr


def score_precomputed(K_train, K_test, y_train, y_test, seed):
    svc = SVC(kernel="precomputed", C=1.0, random_state=seed)
    svc.fit(K_train, y_train)
    y_pred = svc.predict(K_test)
    scores = decision_scores(svc, K_test)
    return compute_metrics(y_test, y_pred, scores)


def score_linear(X_train, X_test, y_train, y_test, seed):
    svc = SVC(kernel="linear", C=1.0, random_state=seed)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    scores = decision_scores(svc, X_test)
    return compute_metrics(y_test, y_pred, scores)


# --------------------------------------------------------------------------- #
# Artifact 1: photonic Table 1 (Tier-1 shape)                                 #
# --------------------------------------------------------------------------- #
def build_table1(args, synthetic, seeds, meta):
    long_rows: list[dict] = []
    for model, q in PHOTONIC_CONFIGS:
        for seed in seeds:
            X_train, X_test, y_train, y_test, evr = prepare_split(
                source=args.source, model=model, q=q, seed=seed,
                data_root=args.data_root, synthetic=synthetic,
                train_cap=args.train_cap, test_cap=args.test_cap,
            )
            base = {"model": model, "q": q, "seed": seed,
                    "train_samples": int(len(y_train)),
                    "test_samples": int(len(y_test)),
                    "explained_variance_ratio": float(evr)}

            lin = score_linear(X_train, X_test, y_train, y_test, seed)
            long_rows.append({**base, "method": "linear_c1", **lin})

            Kq_tr = fidelity_kernel(X_train)
            Kq_te = fidelity_kernel(X_test, X_train)
            q_metrics = score_precomputed(Kq_tr, Kq_te, y_train, y_test, seed)
            long_rows.append({**base, "method": "qsvm_qubit", **q_metrics})

            Kp_tr, Kp_te = photonic_fidelity_kernels(
                X_train, X_test, q, n_photons=args.n_photons, seed=seed)
            p_metrics = score_precomputed(Kp_tr, Kp_te, y_train, y_test, seed)
            long_rows.append({**base, "method": "qsvm_photonic", **p_metrics})
            print(f"[table1] {model} q={q} seed={seed} "
                  f"lin_f1={lin['f1']:.3f} qubit_f1={q_metrics['f1']:.3f} "
                  f"phot_f1={p_metrics['f1']:.3f}", flush=True)

    # summarize per (model, q)
    grouped = defaultdict(list)
    for r in long_rows:
        grouped[(r["model"], r["q"])].append(r)
    summary = []
    for model, q in PHOTONIC_CONFIGS:
        rows = grouped[(model, q)]
        def mean(method, metric):
            vals = [r[metric] for r in rows if r["method"] == method]
            return float(np.mean(vals)) if vals else float("nan")
        lin_f1 = mean("linear_c1", "f1")
        qub_f1 = mean("qsvm_qubit", "f1")
        pho_f1 = mean("qsvm_photonic", "f1")
        summary.append({
            "model": model, "q": q,
            "linear_c1_f1": lin_f1,
            "qsvm_qubit_f1": qub_f1,
            "qsvm_photonic_f1": pho_f1,
            "qsvm_qubit_auc": mean("qsvm_qubit", "auc"),
            "qsvm_photonic_auc": mean("qsvm_photonic", "auc"),
            "photonic_vs_linear_gain": pho_f1 - lin_f1,
            "photonic_vs_qubit_delta": pho_f1 - qub_f1,
        })

    prefix = "photonic_table1"
    write_csv(args.results_dir / f"{prefix}_long.csv", long_rows)
    write_csv(args.results_dir / f"{prefix}_summary.csv", summary)
    wins = sum(1 for r in summary if r["photonic_vs_linear_gain"] > 0)
    payload = {
        "artifact": prefix, "paper_table": "Table 1 (photonic variant)",
        "paper_pointer": PAPER_POINTERS["table1"],
        "aggregate": {
            "photonic_beats_linear": f"{wins}/{len(summary)}",
            "mean_photonic_vs_linear_gain":
                float(np.mean([r["photonic_vs_linear_gain"] for r in summary])),
            "mean_photonic_vs_qubit_delta":
                float(np.mean([r["photonic_vs_qubit_delta"] for r in summary])),
        },
        **meta,
    }
    (args.results_dir / f"{prefix}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_table1_md(args.results_dir / f"{prefix}.md", summary, payload)
    print(json.dumps(payload["aggregate"], indent=2))
    return payload


def write_table1_md(path, summary, payload):
    lines = [
        "# Photonic surrogate Table 1 (Tier-1 shape)",
        "",
        "Photonic (MerLin two-photon linear-optical) fidelity kernel vs the qubit "
        "QSVM and the collapsing linear SVM, on the labelled surrogate dataset. "
        "Not a reproduction of the paper numbers (real MIMIC-CXR embeddings are gated); "
        f"train/test subsampled to {payload['subsampling']['train_cap']}/"
        f"{payload['subsampling']['test_cap']} for photonic SLOS tractability.",
        "",
        f"Paper methodology pointer: {payload['paper_pointer']}",
        "",
        "| Model | q | Linear C=1 F1 | QSVM qubit F1 | QSVM photonic F1 | "
        "Photonic AUC | Photonic − Linear | Photonic − Qubit |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in summary:
        lines.append(
            f"| {r['model']} | {r['q']} | {r['linear_c1_f1']:.3f} | "
            f"{r['qsvm_qubit_f1']:.3f} | {r['qsvm_photonic_f1']:.3f} | "
            f"{r['qsvm_photonic_auc']:.3f} | {r['photonic_vs_linear_gain']:+.3f} | "
            f"{r['photonic_vs_qubit_delta']:+.3f} |")
    agg = payload["aggregate"]
    lines += [
        "",
        f"Photonic QSVM beats the collapsing linear SVM on minority-F1 in "
        f"**{agg['photonic_beats_linear']}** configs "
        f"(mean gain {agg['mean_photonic_vs_linear_gain']:+.3f}). "
        f"Mean photonic−qubit F1 delta: {agg['mean_photonic_vs_qubit_delta']:+.3f}.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Artifact 2: photonic Figure 4 (class-sorted Gram matrix heatmap)            #
# --------------------------------------------------------------------------- #
def build_figure4(args, synthetic, meta):
    model, q, seed = args.fig4_model, args.fig4_q, args.fig4_seed
    X, y = load_dataset(
        source=args.source, model=model,
        seed=seed + MODEL_SEED_OFFSETS.get(model, 0) if args.source == "synthetic" else seed,
        data_root=args.data_root, synthetic=synthetic,
    )
    idx_train, _v, _t = split_indices(y, seed=seed)
    idx_train = subsample(idx_train, y, args.fig4_samples, seed)
    # sort selected samples by class label for block structure
    order = np.argsort(y[idx_train], kind="stable")
    idx_sorted = idx_train[order]
    X_train, _Xv, _Xt, _evr = preprocess(
        X[idx_sorted], X[idx_sorted], X[idx_sorted[:2]], q)
    y_sorted = y[idx_sorted]
    Kp_tr, _ = photonic_fidelity_kernels(
        X_train, X_train[:2], q, n_photons=args.n_photons, seed=seed)
    K = Kp_tr
    boundary = int(np.searchsorted(y_sorted, y_sorted.max()))

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    vmax = float(np.quantile(K, 0.99))
    vmin = float(np.quantile(K, 0.01))
    im = ax.imshow(K, cmap="viridis", vmin=vmin, vmax=vmax)
    if 0 < boundary < len(y_sorted):
        ax.axhline(boundary - 0.5, color="white", lw=0.8, ls="--")
        ax.axvline(boundary - 0.5, color="white", lw=0.8, ls="--")
    ax.set_title(f"Photonic fidelity Gram matrix\n{model}, q={q}, "
                 f"{len(y_sorted)} samples (class-sorted)")
    ax.set_xlabel("sample (sorted by class)")
    ax.set_ylabel("sample (sorted by class)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="K(x, y)")
    fig.tight_layout()
    prefix = "photonic_figure4"
    fig.savefig(args.results_dir / f"{prefix}.png", dpi=130)
    plt.close(fig)
    np.savetxt(args.results_dir / f"{prefix}_kernel_matrix.csv", K, delimiter=",")

    payload = {
        "artifact": prefix, "paper_figure": "Figure 4 (photonic variant)",
        "paper_pointer": PAPER_POINTERS["figure4"],
        "model": model, "q": q, "seed": seed,
        "n_samples": int(len(y_sorted)),
        "minority_count": int((y_sorted == y_sorted.max()).sum()),
        "kernel_stats": {"min": float(K.min()), "max": float(K.max()),
                         "mean": float(K.mean()),
                         "offdiag_mean": float((K.sum() - np.trace(K)) /
                                               (K.size - len(K)))},
        **meta,
    }
    (args.results_dir / f"{prefix}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.results_dir / f"{prefix}.md").write_text(
        f"# Photonic surrogate Figure 4\n\n"
        f"Class-sorted photonic fidelity Gram matrix ({model}, q={q}, "
        f"{len(y_sorted)} samples). Off-diagonal block structure mirrors the "
        f"qubit Figure 4. Paper pointer: {payload['paper_pointer']}.\n\n"
        f"Kernel off-diagonal mean {payload['kernel_stats']['offdiag_mean']:.4f}, "
        f"max {payload['kernel_stats']['max']:.3f}.\n")
    print(f"[figure4] wrote {prefix}.png offdiag_mean="
          f"{payload['kernel_stats']['offdiag_mean']:.4f}")
    return payload


# --------------------------------------------------------------------------- #
# Artifact 3: photonic effective rank vs q                                     #
# --------------------------------------------------------------------------- #
def build_effrank(args, synthetic, seeds, meta):
    model = args.effrank_model
    qs = args.effrank_qs
    rows = []
    for q in qs:
        vals = []
        for seed in seeds:
            X, y = load_dataset(
                source=args.source, model=model,
                seed=seed + MODEL_SEED_OFFSETS.get(model, 0) if args.source == "synthetic" else seed,
                data_root=args.data_root, synthetic=synthetic,
            )
            idx_train, _v, _t = split_indices(y, seed=seed)
            idx_train = subsample(idx_train, y, args.effrank_samples, seed)
            X_train, _Xv, _Xt, _evr = preprocess(
                X[idx_train], X[idx_train], X[idx_train[:2]], q)
            Kp_tr, _ = photonic_fidelity_kernels(
                X_train, X_train[:2], q, n_photons=args.n_photons, seed=seed)
            vals.append(effective_rank(Kp_tr))
        rows.append({"q": q, "model": model,
                     "photonic_eff_rank_mean": float(np.mean(vals)),
                     "photonic_eff_rank_std": float(np.std(vals)),
                     "n_samples": int(len(idx_train))})
        print(f"[effrank] q={q} eff_rank={np.mean(vals):.2f}", flush=True)

    prefix = "photonic_effrank"
    write_csv(args.results_dir / f"{prefix}_summary.csv", rows)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.errorbar([r["q"] for r in rows],
                [r["photonic_eff_rank_mean"] for r in rows],
                yerr=[r["photonic_eff_rank_std"] for r in rows],
                marker="o", capsize=3)
    ax.set_xlabel("qubits / modes q")
    ax.set_ylabel("photonic kernel effective rank")
    ax.set_title(f"Photonic fidelity-kernel effective rank vs q\n{model}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.results_dir / f"{prefix}.png", dpi=130)
    plt.close(fig)

    payload = {
        "artifact": prefix, "paper_ref": "Figure 2 / Table 5 (photonic variant)",
        "paper_pointer": PAPER_POINTERS["effrank"],
        "model": model, "rows": rows, **meta,
    }
    (args.results_dir / f"{prefix}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.results_dir / f"{prefix}.md").write_text(
        f"# Photonic surrogate effective rank vs q\n\n"
        f"Effective rank (exp von-Neumann entropy) of the photonic fidelity kernel "
        f"as q grows ({model}). Mirrors the eigenspectrum claim (Figure 2 / Table 5). "
        f"Paper pointer: {payload['paper_pointer']}.\n\n"
        + "\n".join(f"- q={r['q']}: eff_rank "
                    f"{r['photonic_eff_rank_mean']:.2f} ± {r['photonic_eff_rank_std']:.2f}"
                    for r in rows) + "\n")
    print(f"[effrank] wrote {prefix}.png/.csv")
    return payload


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=("synthetic", "synthetic_file"),
                   default="synthetic_file")
    p.add_argument("--data-root", type=Path,
                   default=PROJECT_ROOT / "data" / "synthetic_qml_mimic_cxr_embeddings")
    p.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--n-photons", type=int, default=2)
    p.add_argument("--train-cap", type=int, default=220)
    p.add_argument("--test-cap", type=int, default=100)
    # synthetic in-memory fallback params
    p.add_argument("--n-samples", type=int, default=600)
    p.add_argument("--ambient-dim", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=30)
    p.add_argument("--minority-frac", type=float, default=0.28)
    p.add_argument("--signal", type=float, default=0.30)
    p.add_argument("--noise", type=float, default=1.0)
    # figure4 params
    p.add_argument("--fig4-model", default="medsiglip-448")
    p.add_argument("--fig4-q", type=int, default=6)
    p.add_argument("--fig4-seed", type=int, default=0)
    p.add_argument("--fig4-samples", type=int, default=200)
    # effrank params
    p.add_argument("--effrank-model", default="medsiglip-448")
    p.add_argument("--effrank-qs", default="4,6,8,10")
    p.add_argument("--effrank-samples", type=int, default=200)
    p.add_argument("--only", default="all",
                   help="comma list of {table1,figure4,effrank} or 'all'")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.effrank_qs = parse_ints(args.effrank_qs)
    seeds = parse_ints(args.seeds)
    synthetic = SyntheticSpec(
        n_samples=args.n_samples, ambient_dim=args.ambient_dim,
        latent_dim=args.latent_dim, minority_frac=args.minority_frac,
        signal=args.signal, noise=args.noise,
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "kernel": "photonic (MerLin two-photon linear-optical fidelity)",
        "n_photons": args.n_photons,
        "data": {
            "source": args.source,
            "synthetic_surrogate": True,
            "data_root": str(args.data_root) if args.source == "synthetic_file" else None,
            "seeds": seeds,
        },
        "subsampling": {"train_cap": args.train_cap, "test_cap": args.test_cap,
                        "reason": "photonic SLOS Gram matrix is O(N^2)"},
    }
    which = ({"table1", "figure4", "effrank"} if args.only == "all"
             else set(s.strip() for s in args.only.split(",")))
    if "table1" in which:
        build_table1(args, synthetic, seeds, meta)
    if "figure4" in which:
        build_figure4(args, synthetic, meta)
    if "effrank" in which:
        build_effrank(args, synthetic, seeds, meta)
    print("photonic artifacts done")


if __name__ == "__main__":
    main()
