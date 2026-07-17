#!/usr/bin/env python3
"""Generate a Figure 3-style quantum-vs-linear eigenspectrum plot.

This is not a reproduction of the paper figure when ``--source synthetic`` is
used. It mirrors the Figure 3 diagnostic shape: MedSigLIP-448 at q=4 and q=6,
comparing quantum fidelity-kernel eigenvalues with linear-kernel eigenvalues.
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

from synthetic_surrogate_figure2 import count_positive_eigenvalues, normalized_eigenvalues
from synthetic_surrogate_table1 import SyntheticSpec, load_dataset, parse_ints
from lib.quantum_kernel import effective_rank, fidelity_kernel
from lib.svm_pipeline import preprocess, split_indices


PAPER_FIGURE3_POINTER = "https://arxiv.org/html/2604.24597v1#S4.F3"

QUANTUM_COLORS = {4: "#2166ac", 6: "#4dac26"}
LINEAR_COLORS = {4: "#e66101", 6: "#d01c8b"}


def cumulative_with_zero(eigenvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalue_count = np.arange(len(eigenvalues) + 1)
    cumulative_variance = np.concatenate([[0.0], np.cumsum(eigenvalues)])
    return eigenvalue_count, cumulative_variance


def compute_kernel_spectrum(kernel: np.ndarray) -> dict[str, object]:
    """Return normalized eigenvalues and rank diagnostics for one kernel."""
    eigenvalues = normalized_eigenvalues(kernel)
    return {
        "eigenvalues": eigenvalues,
        "positive_rank": count_positive_eigenvalues(eigenvalues),
        "effective_rank": effective_rank(kernel),
    }


def compute_figure3_series(
    *,
    source: str,
    model: str,
    q: int,
    seed: int,
    data_root: Path | None,
    synthetic: SyntheticSpec,
) -> list[dict[str, object]]:
    """Compute quantum and linear eigenspectra for one q value."""
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

    quantum_spectrum = compute_kernel_spectrum(fidelity_kernel(X_train))
    linear_spectrum = compute_kernel_spectrum(X_train @ X_train.T)
    shared = {
        "source": source,
        "synthetic_surrogate": source != "real",
        "model": model,
        "q": q,
        "seed": seed,
        "train_samples": int(len(idx_train)),
        "pca_variance_percent": 100.0 * float(explained_variance_ratio),
    }
    return [
        {**shared, "kernel": "quantum", **quantum_spectrum},
        {**shared, "kernel": "linear", **linear_spectrum},
    ]


def eigenvalue_rows(series_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for series in series_rows:
        eigenvalues = np.asarray(series["eigenvalues"], dtype=float)
        cumulative_variance = np.cumsum(eigenvalues)
        for index, value in enumerate(eigenvalues):
            rows.append(
                {
                    "model": series["model"],
                    "q": series["q"],
                    "kernel": series["kernel"],
                    "eigenvalue_index": index,
                    "normalized_eigenvalue": float(value),
                    "eigenvalue_count": index + 1,
                    "cumulative_variance": float(cumulative_variance[index]),
                }
            )
    return rows


def plot_eigenvalue_decay(ax, *, series_rows: list[dict[str, object]]) -> None:
    q_values = sorted({int(row["q"]) for row in series_rows})
    plot_order = [("quantum", q) for q in q_values] + [("linear", q) for q in q_values]
    ordered_rows = [
        row
        for kernel, q in plot_order
        for row in series_rows
        if row["kernel"] == kernel and int(row["q"]) == q
    ]
    for series in ordered_rows:
        q = int(series["q"])
        eigenvalues = np.asarray(series["eigenvalues"], dtype=float)
        x = np.arange(len(eigenvalues))
        y = np.where(eigenvalues > 1e-14, eigenvalues, np.nan)
        if series["kernel"] == "quantum":
            color = QUANTUM_COLORS.get(q, "#2166ac")
            label = f"Quantum q={q} (eff.rank={series['effective_rank']:.1f})"
            ax.semilogy(x, y, color=color, linewidth=1.9, label=label)
        else:
            color = LINEAR_COLORS.get(q, "#e66101")
            label = f"Linear q={q} (eff.rank={series['effective_rank']:.1f})"
            ax.semilogy(x, y, color=color, linewidth=2.8, linestyle="--", label=label)
            rank = int(series["positive_rank"])
            if rank > 0:
                ax.scatter([rank - 1], [y[rank - 1]], marker="X", color=color, s=90, zorder=4)
                ax.axvline(rank, color=color, linestyle=":", linewidth=1.3, alpha=0.55, label=f"Rank boundary q={q}")

    ax.set_xlabel("Eigenvalue index (sorted descending)")
    ax.set_ylabel("Normalised eigenvalue (log scale)")
    ax.set_xlim(0, 80)
    ax.set_ylim(1e-12, 1.5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower left")


def plot_cumulative_variance(ax, *, series_rows: list[dict[str, object]]) -> None:
    q_values = sorted({int(row["q"]) for row in series_rows})
    plot_order = [("quantum", q) for q in q_values] + [("linear", q) for q in q_values]
    ordered_rows = [
        row
        for kernel, q in plot_order
        for row in series_rows
        if row["kernel"] == kernel and int(row["q"]) == q
    ]
    for series in ordered_rows:
        q = int(series["q"])
        eigenvalues = np.asarray(series["eigenvalues"], dtype=float)
        eigenvalue_count, cumulative_variance = cumulative_with_zero(eigenvalues)
        if series["kernel"] == "quantum":
            color = QUANTUM_COLORS.get(q, "#2166ac")
            ax.plot(eigenvalue_count, cumulative_variance, color=color, linewidth=1.9, label=f"Quantum q={q}")
        else:
            color = LINEAR_COLORS.get(q, "#e66101")
            rank = int(series["positive_rank"])
            ax.plot(
                eigenvalue_count[: rank + 1],
                cumulative_variance[: rank + 1],
                color=color,
                linewidth=2.8,
                linestyle="--",
                label=f"Linear q={q}",
            )
            if rank > 0:
                ax.scatter([rank], [cumulative_variance[rank]], marker="X", color=color, s=90, zorder=4)

    ax.axhline(0.99, color="gray", linestyle=":", linewidth=0.9, label="99% var")
    ax.axhline(0.90, color="gray", linestyle="--", linewidth=0.9, label="90% var")
    ax.set_xlabel("Number of eigenvalues")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower right")


def save_plot(path: Path, *, series_rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_eigenvalue_decay(axes[0], series_rows=series_rows)
    plot_cumulative_variance(axes[1], series_rows=series_rows)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, *, payload: dict[str, object]) -> None:
    image_name = Path(payload["paths"]["png"]).name
    lines = [
        "# Synthetic surrogate Figure 3 pipeline",
        "",
        "This artifact is a surrogate computation only. It does not reproduce the paper figure because the gated MIMIC-CXR embedding dataset is not available locally.",
        "",
        f"Paper methodology pointer: {PAPER_FIGURE3_POINTER}",
        "",
        "Figure 3 compares quantum and linear kernel eigenspectra for q=4 and q=6. The left panel shows normalized eigenvalues on a logarithmic scale; the right panel shows cumulative explained variance.",
        "",
        f"![Synthetic surrogate Figure 3]({image_name})",
        "",
        "Summary:",
        "",
    ]
    for summary in payload["summary_rows"]:
        lines.append(
            "- {kernel} q={q}: effective rank `{eff:.3f}`, positive rank `{rank}`".format(
                kernel=summary["kernel"],
                q=summary["q"],
                eff=summary["effective_rank"],
                rank=summary["positive_rank"],
            )
        )
    lines.extend(
        [
            "",
            "Data source metadata:",
            "",
            "```json",
            json.dumps(payload["data"], indent=2, sort_keys=True),
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def default_prefix(source: str) -> str:
    if source == "synthetic":
        return "synthetic_surrogate_figure3"
    if source == "synthetic_file":
        return "synthetic_file_figure3"
    return "real_figure3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "synthetic_file", "real"), default="synthetic")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--model", default="medsiglip-448")
    parser.add_argument("--q-values", default="4,6")
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
    q_values = parse_ints(args.q_values)
    synthetic = SyntheticSpec(
        n_samples=args.n_samples,
        ambient_dim=args.ambient_dim,
        latent_dim=args.latent_dim,
        minority_frac=args.minority_frac,
        signal=args.signal,
        noise=args.noise,
    )

    series_rows: list[dict[str, object]] = []
    for q in q_values:
        series_rows.extend(
            compute_figure3_series(
                source=args.source,
                model=args.model,
                q=q,
                seed=args.seed,
                data_root=args.data_root,
                synthetic=synthetic,
            )
        )

    prefix = args.output_prefix or default_prefix(args.source)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.results_dir / f"{prefix}.png"
    csv_path = args.results_dir / f"{prefix}_eigenvalues.csv"
    json_path = args.results_dir / f"{prefix}.json"
    md_path = args.results_dir / f"{prefix}.md"

    write_csv(csv_path, eigenvalue_rows(series_rows))
    save_plot(png_path, series_rows=series_rows)

    summary_rows = [{key: value for key, value in row.items() if key != "eigenvalues"} for row in series_rows]
    payload: dict[str, object] = {
        "artifact": prefix,
        "paper_figure": "Figure 3",
        "paper_pointer": PAPER_FIGURE3_POINTER,
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
            "q_values": q_values,
            "split": "80/10/10 stratified via lib.svm_pipeline.split_indices",
        },
        "summary_rows": summary_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(md_path, payload=payload)

    print(json.dumps(summary_rows, indent=2, sort_keys=True))
    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
