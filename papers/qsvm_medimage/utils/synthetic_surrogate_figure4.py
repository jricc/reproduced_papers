#!/usr/bin/env python3
"""Generate a Figure 4-style quantum-kernel heatmap.

This is not a reproduction of the paper figure when ``--source synthetic`` is
used. It mirrors the Figure 4 diagnostic shape: a MedSigLIP-448 q=11 quantum
kernel matrix heatmap on 200 training samples.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from synthetic_surrogate_table1 import SyntheticSpec, load_dataset
from lib.quantum_kernel import fidelity_kernel
from lib.svm_pipeline import preprocess, split_indices


PAPER_FIGURE4_POINTER = "https://arxiv.org/html/2604.24597v1#S4.F4"


def select_first_samples(X_train: np.ndarray, sample_count: int) -> np.ndarray:
    """Use the first training samples, matching the paper heatmap's sample-index view."""
    return X_train[: min(sample_count, len(X_train))]


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
    X_subset = select_first_samples(X_train, sample_count)
    raw_kernel = fidelity_kernel(X_subset)
    plot_kernel = normalize_kernel_for_plot(raw_kernel, kernel_normalization)
    return {
        "source": source,
        "synthetic_surrogate": source != "real",
        "model": model,
        "q": q,
        "seed": seed,
        "train_samples": int(len(idx_train)),
        "sample_count": int(len(X_subset)),
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


def save_plot(path: Path, *, figure4: dict[str, object], vmin: float, vmax: float) -> None:
    matrix = np.asarray(figure4["plot_kernel"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax, interpolation="nearest", origin="upper")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Sample Index")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Kernel Value")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


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
        "Figure 4 is a quantum-kernel heatmap for MedSigLIP-448 at q=11 on 200 training samples. The color scale is fixed to [-1, 1] to match the paper rendering.",
        "",
        f"![Synthetic surrogate Figure 4]({image_name})",
        "",
        "Summary:",
        "",
        f"- model: `{summary['model']}`",
        f"- q: `{summary['q']}`",
        f"- seed: `{summary['seed']}`",
        f"- sample count: `{summary['sample_count']}`",
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
    parser.add_argument("--q", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--kernel-normalization", choices=("none", "trace"), default="none")
    parser.add_argument("--vmin", type=float, default=-1.0)
    parser.add_argument("--vmax", type=float, default=1.0)
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
    save_plot(png_path, figure4=figure4, vmin=args.vmin, vmax=args.vmax)

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
            "color_scale": [args.vmin, args.vmax],
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
