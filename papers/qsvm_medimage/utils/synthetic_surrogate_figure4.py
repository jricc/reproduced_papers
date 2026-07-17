#!/usr/bin/env python3
"""Generate a Figure 4-style quantum-kernel heatmap.

This is not a reproduction of the paper figure when ``--source synthetic`` is
used. It mirrors the Figure 4 diagnostic shape: a MedSigLIP-448 q=6 quantum
kernel matrix heatmap on 200 training samples, trace-normalized and sorted by
class label so the off-diagonal block structure at the class boundary is
visible (paper Figure 4 caption: "Quantum kernel matrix K_Q (trace-normalized)
for MedSigLIP-448 at q=6, 200 training samples sorted by class label").
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
from lib.quantum_kernel import fidelity_kernel
from lib.svm_pipeline import preprocess, split_indices


PAPER_FIGURE4_POINTER = "https://arxiv.org/html/2604.24597v1#S4.F4"


def select_samples_sorted_by_class(
    X_train: np.ndarray, y_train: np.ndarray, sample_count: int
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Select the first ``sample_count`` training samples, then sort them by class.

    The paper's Figure 4 shows the quantum kernel on "200 training samples sorted
    by class label", which is what produces the off-diagonal block structure. We
    take the first ``sample_count`` samples (deterministic given the split) and
    then apply a *stable* sort by label so within-class order is preserved. The
    returned ``class_counts`` gives the number of samples per class in sorted
    order, so the class boundary can be drawn on the heatmap.
    """
    n = min(sample_count, len(X_train))
    X_subset = X_train[:n]
    y_subset = y_train[:n]
    order = np.argsort(y_subset, kind="stable")
    X_sorted = X_subset[order]
    y_sorted = y_subset[order]
    labels, counts = np.unique(y_sorted, return_counts=True)
    return X_sorted, y_sorted, [int(c) for c in counts]


def normalize_kernel_for_plot(kernel: np.ndarray, normalization: str) -> np.ndarray:
    """Apply the requested heatmap normalization."""
    if normalization == "none":
        return kernel
    if normalization == "trace":
        trace = float(np.trace(kernel))
        if trace <= 0.0:
            return kernel
        return kernel / trace
    raise ValueError(f"unknown kernel normalization: {normalization}")


def kernel_summary(kernel: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(kernel)),
        "max": float(np.max(kernel)),
        "mean": float(np.mean(kernel)),
        "std": float(np.std(kernel)),
        "trace": float(np.trace(kernel)),
    }


def compute_figure4(
    *,
    source: str,
    model: str,
    q: int,
    seed: int,
    data_root: Path | None,
    synthetic: SyntheticSpec,
    sample_count: int,
    kernel_normalization: str,
) -> dict[str, object]:
    """Compute the quantum-kernel matrix used by Figure 4."""
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
    y_train = y[idx_train]
    X_subset, y_subset, class_counts = select_samples_sorted_by_class(
        X_train, y_train, sample_count
    )
    raw_kernel = fidelity_kernel(X_subset)
    plot_kernel = normalize_kernel_for_plot(raw_kernel, kernel_normalization)
    # Boundary index between the first (majority-label) block and the next class,
    # used to draw the class divider on the heatmap.
    class_boundary = class_counts[0] if len(class_counts) > 1 else None
    return {
        "source": source,
        "synthetic_surrogate": source != "real",
        "model": model,
        "q": q,
        "seed": seed,
        "train_samples": int(len(idx_train)),
        "sample_count": int(len(X_subset)),
        "sorted_by_class": True,
        "class_counts": class_counts,
        "class_boundary": class_boundary,
        "pca_variance_percent": 100.0 * float(explained_variance_ratio),
        "kernel_normalization": kernel_normalization,
        "raw_kernel_summary": kernel_summary(raw_kernel),
        "plot_kernel_summary": kernel_summary(plot_kernel),
        "plot_kernel": plot_kernel,
    }


def write_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(matrix.tolist())


def save_plot(
    path: Path,
    *,
    figure4: dict[str, object],
    vmin: float | None,
    vmax: float | None,
) -> tuple[float, float]:
    matrix = np.asarray(figure4["plot_kernel"], dtype=float)
    # Auto-range the color scale to the data when not explicitly set. This matters
    # for trace normalization, where values are ~1/N and a fixed [-1, 1] scale
    # would wash the heatmap out completely.
    if vmin is None:
        vmin = float(matrix.min())
    if vmax is None:
        vmax = float(matrix.max())
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax, interpolation="nearest", origin="upper")
    boundary = figure4.get("class_boundary")
    if boundary:
        line = float(boundary) - 0.5
        ax.axhline(line, color="black", linewidth=0.8, linestyle="--")
        ax.axvline(line, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Sample Index (sorted by class)")
    ax.set_ylabel("Sample Index (sorted by class)")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Kernel Value")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return vmin, vmax


def write_markdown(path: Path, *, payload: dict[str, object]) -> None:
    summary = payload["summary"]
    image_name = Path(payload["paths"]["png"]).name
    lines = [
        "# Synthetic surrogate Figure 4 pipeline",
        "",
        "This artifact is a surrogate computation only. It does not reproduce the paper figure because the gated MIMIC-CXR embedding dataset is not available locally.",
        "",
        f"Paper methodology pointer: {PAPER_FIGURE4_POINTER}",
        "",
        "Figure 4 is a trace-normalized quantum-kernel heatmap for MedSigLIP-448 at q=6 on 200 training samples sorted by class label. The off-diagonal block structure reflects the class boundary; the color scale is auto-ranged to the trace-normalized values.",
        "",
        f"![Synthetic surrogate Figure 4]({image_name})",
        "",
        "Summary:",
        "",
        f"- model: `{summary['model']}`",
        f"- q: `{summary['q']}`",
        f"- seed: `{summary['seed']}`",
        f"- sample count: `{summary['sample_count']}`",
        f"- sorted by class: `{summary.get('sorted_by_class', False)}`",
        f"- class counts (sorted): `{summary.get('class_counts')}`",
        f"- kernel normalization: `{summary['kernel_normalization']}`",
        f"- raw kernel mean: `{summary['raw_kernel_summary']['mean']:.6f}`",
        f"- raw kernel std: `{summary['raw_kernel_summary']['std']:.6f}`",
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
        return "synthetic_surrogate_figure4"
    if source == "synthetic_file":
        return "synthetic_file_figure4"
    return "real_figure4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "synthetic_file", "real"), default="synthetic")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--model", default="medsiglip-448")
    parser.add_argument("--q", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--kernel-normalization", choices=("none", "trace"), default="trace")
    parser.add_argument(
        "--vmin", type=float, default=None, help="Color-scale minimum (default: auto from data)."
    )
    parser.add_argument(
        "--vmax", type=float, default=None, help="Color-scale maximum (default: auto from data)."
    )
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
    figure4 = compute_figure4(
        source=args.source,
        model=args.model,
        q=args.q,
        seed=args.seed,
        data_root=args.data_root,
        synthetic=synthetic,
        sample_count=args.sample_count,
        kernel_normalization=args.kernel_normalization,
    )

    prefix = args.output_prefix or default_prefix(args.source)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.results_dir / f"{prefix}.png"
    csv_path = args.results_dir / f"{prefix}_kernel_matrix.csv"
    json_path = args.results_dir / f"{prefix}.json"
    md_path = args.results_dir / f"{prefix}.md"

    matrix = np.asarray(figure4["plot_kernel"], dtype=float)
    write_matrix_csv(csv_path, matrix)
    used_vmin, used_vmax = save_plot(png_path, figure4=figure4, vmin=args.vmin, vmax=args.vmax)

    summary = {key: value for key, value in figure4.items() if key != "plot_kernel"}
    payload: dict[str, object] = {
        "artifact": prefix,
        "paper_figure": "Figure 4",
        "paper_pointer": PAPER_FIGURE4_POINTER,
        "paths": {
            "png": str(png_path),
            "kernel_matrix_csv": str(csv_path),
            "json": str(json_path),
            "markdown": str(md_path),
        },
        "data": {
            "source": args.source,
            "synthetic_surrogate": args.source != "real",
            "synthetic_spec": asdict(synthetic) if args.source == "synthetic" else None,
            "data_root": str(args.data_root) if args.data_root else None,
            "split": "80/10/10 stratified via lib.svm_pipeline.split_indices",
            "color_scale": [used_vmin, used_vmax],
            "color_scale_auto": args.vmin is None or args.vmax is None,
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
