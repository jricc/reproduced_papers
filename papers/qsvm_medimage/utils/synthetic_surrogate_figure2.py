#!/usr/bin/env python3
"""Generate a Figure 2-style linear-kernel eigenspectrum plot.

This is not a reproduction of the paper figure when ``--source synthetic`` is
used. It mirrors the Figure 2 diagnostic shape: MedSigLIP-448 at q=6, seed 0,
showing that the PCA-q linear kernel has only q positive eigenvalues.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = PROJECT_ROOT.parents[1]
for root in (PROJECT_ROOT, REPRO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from synthetic_surrogate_table1 import SyntheticSpec, load_dataset
from lib.quantum_kernel import effective_rank
from lib.svm_pipeline import preprocess, split_indices


PAPER_FIGURE2_POINTER = "https://arxiv.org/html/2604.24597v1#S4.F2"


def kernel_eigenvalues(kernel: np.ndarray) -> np.ndarray:
    """Return non-negative kernel eigenvalues sorted descending."""
    eigenvalues = np.linalg.eigvalsh(kernel)
    return np.maximum(eigenvalues, 0.0)[::-1]


def normalized_eigenvalues(kernel: np.ndarray) -> np.ndarray:
    eigenvalues = kernel_eigenvalues(kernel)
    total = float(np.sum(eigenvalues))
    if total == 0.0:
        return eigenvalues
    return eigenvalues / total


def count_positive_eigenvalues(eigenvalues: np.ndarray, tol: float | None = None) -> int:
    if tol is None:
        max_eigenvalue = float(np.max(eigenvalues)) if len(eigenvalues) else 0.0
        tol = max(1e-12, 1e-10 * max_eigenvalue)
    return int(np.sum(eigenvalues > tol))


def eigenvalue_rows(raw_eigenvalues: np.ndarray) -> list[dict[str, object]]:
    total = float(np.sum(raw_eigenvalues))
    if total == 0.0:
        normalized = raw_eigenvalues
    else:
        normalized = raw_eigenvalues / total
    cumulative = np.cumsum(normalized)
    return [
        {
            "eigenvalue_index": index,
            "raw_eigenvalue": float(raw_eigenvalues[index]),
            "normalized_eigenvalue": float(normalized[index]),
            "eigenvalue_count": index + 1,
            "cumulative_variance": float(cumulative[index]),
        }
        for index in range(len(raw_eigenvalues))
    ]


def compute_figure2(
    *,
    source: str,
    model: str,
    q: int,
    seed: int,
    data_root: Path | None,
    synthetic: SyntheticSpec,
) -> dict[str, object]:
    """Compute the linear-kernel eigenspectrum used by Figure 2."""
    X, y = load_dataset(
        source=source,
        model=model,
        seed=seed,
        data_root=data_root,
        synthetic=synthetic,
    )
    idx_train, idx_val, idx_test = split_indices(y, seed=seed)
    X_train, _, _, explained_variance_ratio = preprocess(
        X[idx_train], X[idx_val], X[idx_test], q
    )

    linear_kernel = X_train @ X_train.T
    raw_eigenvalues = kernel_eigenvalues(linear_kernel)
    normalized = normalized_eigenvalues(linear_kernel)
    positive_rank = count_positive_eigenvalues(raw_eigenvalues)
    return {
        "source": source,
        "synthetic_surrogate": source != "real",
        "model": model,
        "q": q,
        "seed": seed,
        "train_samples": int(len(idx_train)),
        "pca_variance_percent": 100.0 * float(explained_variance_ratio),
        "positive_rank": positive_rank,
        "rank_upper_bound": q,
        "rank_validation": positive_rank <= q,
        "effective_rank": effective_rank(linear_kernel),
        "raw_eigenvalues": raw_eigenvalues,
        "normalized_eigenvalues": normalized,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path: Path, *, figure2: dict[str, object]) -> None:
    raw_eigenvalues = np.asarray(figure2["raw_eigenvalues"], dtype=float)
    normalized = np.asarray(figure2["normalized_eigenvalues"], dtype=float)
    eigenvalue_index = np.arange(len(raw_eigenvalues))
    eigenvalue_count = np.arange(len(normalized) + 1)
    cumulative_variance = np.concatenate([[0.0], np.cumsum(normalized)])
    positive_rank = int(figure2["positive_rank"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    plotted = np.where(raw_eigenvalues > 1e-12, raw_eigenvalues, np.nan)
    ax.semilogy(
        eigenvalue_index,
        plotted,
        marker="o",
        markersize=4,
        color="#1f77b4",
        linewidth=2.0,
        label=f"Linear (eff. rank = {figure2['effective_rank']:.1f})",
    )
    ax.set_xlabel("Eigenvalue Index (sorted descending)")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_xlim(0, 200)
    if len(raw_eigenvalues) and raw_eigenvalues[0] > 0.0:
        ax.set_ylim(max(1e-8, raw_eigenvalues[0] * 1e-8), raw_eigenvalues[0] * 1.3)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    ax2 = axes[1]
    ax2.plot(
        eigenvalue_count,
        cumulative_variance,
        color="#1f77b4",
        linewidth=2.0,
        label=f"{figure2['model']}, q={figure2['q']}",
    )
    ax2.axhline(0.90, color="gray", linestyle="--", linewidth=0.9, label="90%")
    ax2.axhline(0.95, color="gray", linestyle=":", linewidth=0.9, label="95%")
    ax2.set_xlabel("Number of Eigenvalues")
    ax2.set_ylabel("Cumulative Variance Explained")
    ax2.set_xlim(0, 200)
    ax2.set_ylim(0, 1.02)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path: Path, *, payload: dict[str, object]) -> None:
    summary = payload["summary"]
    image_name = Path(payload["paths"]["png"]).name
    lines = [
        "# Synthetic surrogate Figure 2 pipeline",
        "",
        "This artifact is a surrogate computation only. It does not reproduce the paper figure because the gated MIMIC-CXR embedding dataset is not available locally.",
        "",
        f"Paper methodology pointer: {PAPER_FIGURE2_POINTER}",
        "",
        "Figure 2 is a linear-kernel eigenspectrum diagnostic for MedSigLIP-448 at q=6. The left panel plots eigenvalues by index on a logarithmic y axis; the right panel plots cumulative eigenvalue variance by number of eigenvalues. Both x axes are shown from 0 to 200.",
        "",
        "Validation note: this script follows the Figure 2 caption and the identity `rank(X X^T) <= q` after PCA-q. A many-eigenvalue curve or a `Quantum` legend would describe a different kernel and would not validate the caption claim that the linear kernel has exactly q positive eigenvalues.",
        "",
        f"![Synthetic surrogate Figure 2]({image_name})",
        "",
        "Summary:",
        "",
        f"- model: `{summary['model']}`",
        f"- q: `{summary['q']}`",
        f"- seed: `{summary['seed']}`",
        f"- train samples: `{summary['train_samples']}`",
        f"- PCA variance percent: `{summary['pca_variance_percent']:.3f}`",
        f"- positive rank: `{summary['positive_rank']}`",
        f"- rank validation (`positive_rank <= q`): `{summary['rank_validation']}`",
        f"- effective rank: `{summary['effective_rank']:.3f}`",
        "",
        "Data source metadata:",
        "",
        "```json",
        json.dumps(payload["data"], indent=2, sort_keys=True),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n")


def default_prefix(source: str) -> str:
    if source == "synthetic":
        return "synthetic_surrogate_figure2"
    if source == "synthetic_file":
        return "synthetic_file_figure2"
    return "real_figure2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "synthetic_file", "real"), default="synthetic")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--model", default="medsiglip-448")
    parser.add_argument("--q", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--ambient-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=30)
    parser.add_argument("--minority-frac", type=float, default=0.20)
    parser.add_argument("--signal", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synthetic = SyntheticSpec(
        n_samples=args.n_samples,
        ambient_dim=args.ambient_dim,
        latent_dim=args.latent_dim,
        minority_frac=args.minority_frac,
        signal=args.signal,
        noise=args.noise,
    )
    figure2 = compute_figure2(
        source=args.source,
        model=args.model,
        q=args.q,
        seed=args.seed,
        data_root=args.data_root,
        synthetic=synthetic,
    )

    prefix = args.output_prefix or default_prefix(args.source)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.results_dir / f"{prefix}.png"
    csv_path = args.results_dir / f"{prefix}_eigenvalues.csv"
    json_path = args.results_dir / f"{prefix}.json"
    md_path = args.results_dir / f"{prefix}.md"

    rows = eigenvalue_rows(np.asarray(figure2["raw_eigenvalues"], dtype=float))
    write_csv(csv_path, rows)
    save_plot(png_path, figure2=figure2)

    summary = {
        key: value
        for key, value in figure2.items()
        if key not in {"raw_eigenvalues", "normalized_eigenvalues"}
    }
    payload: dict[str, object] = {
        "artifact": prefix,
        "paper_figure": "Figure 2",
        "paper_pointer": PAPER_FIGURE2_POINTER,
        "paths": {
            "png": str(png_path),
            "eigenvalues_csv": str(csv_path),
            "json": str(json_path),
            "markdown": str(md_path),
        },
        "data": {
            "source": args.source,
            "synthetic_surrogate": args.source != "real",
            "synthetic_spec": asdict(synthetic) if args.source == "synthetic" else None,
            "data_root": str(args.data_root) if args.data_root else None,
            "split": "80/10/10 stratified via lib.svm_pipeline.split_indices",
        },
        "summary": summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(md_path, payload=payload)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
