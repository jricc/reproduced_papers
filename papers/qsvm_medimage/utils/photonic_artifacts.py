#!/usr/bin/env python3
"""Generate photonic variants of selected kernel artifacts.

The paper evaluates a qubit fidelity kernel. This script compares that kernel
with a native linear-optical fidelity kernel implemented with MerLin.

The photonic feature map is not a gate-by-gate translation of the BSP circuit.
It uses:

1. one fixed mode-mixing interferometer;
2. one feature-dependent phase shift per optical mode;
3. a fixed two-photon input state by default.

Three artifacts are generated:

1. ``photonic_table1``
   Linear SVM, qubit QSVM, and photonic QSVM minority-class F1 and ROC-AUC.

2. ``photonic_figure4``
   Class-sorted photonic fidelity Gram matrix.

3. ``photonic_effrank``
   Photonic-kernel effective rank as a function of the mode count q.

Photonic kernels are evaluated on stratified subsets because Gram-matrix
construction scales quadratically with the number of samples.

Results obtained from synthetic data are controlled benchmark results. They do
not reproduce the numerical results or medical-data claims of the paper.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = PROJECT_ROOT.parents[1]
UTILS_ROOT = PROJECT_ROOT / "utils"

for root in (
    PROJECT_ROOT,
    REPRO_ROOT,
    UTILS_ROOT,
):
    root_string = str(root)

    if root_string not in sys.path:
        sys.path.insert(0, root_string)


import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from lib.photonic_kernel import (  # noqa: E402
    photonic_fidelity_kernels,
)

# Used only for the qubit comparison in photonic_table1.
from lib.quantum_kernel import (  # noqa: E402
    effective_rank,
    fidelity_kernel,
)
from lib.svm_pipeline import (  # noqa: E402
    normalize_train_test_kernels,
    preprocess,
    split_indices,
)
from synthetic_surrogate_table1 import (  # noqa: E402
    SyntheticSpec,
    compute_metrics,
    decision_scores,
    load_dataset,
    parse_ints,
)

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

MODEL_NAMES = (
    "medsiglip-448",
    "rad-dino",
    "vit-patch32-cls",
)

# These offsets are used only for in-memory synthetic generation. Materialized
# synthetic datasets already contain model-specific files and use the
# requested seed directly.
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

ARTIFACT_NAMES = {
    "table1",
    "figure4",
    "effrank",
}


def dataset_seed(
    *,
    source: str,
    model: str,
    seed: int,
) -> int:
    """Return the seed used to load or generate one model dataset."""
    if source == "synthetic":
        return seed + MODEL_SEED_OFFSETS.get(
            model,
            0,
        )

    return seed


def stratified_subsample(
    indices: np.ndarray,
    labels: np.ndarray,
    cap: int,
    seed: int,
) -> np.ndarray:
    """Return at most ``cap`` indices while preserving class proportions.

    The returned subset has exactly ``cap`` elements when the input contains
    more than ``cap`` elements.
    """
    indices = np.asarray(
        indices,
        dtype=int,
    )

    labels = np.asarray(
        labels,
        dtype=int,
    )

    if cap <= 0 or len(indices) <= cap:
        return np.sort(indices)

    selected_indices, _ = train_test_split(
        indices,
        train_size=cap,
        random_state=seed,
        stratify=labels[indices],
    )

    return np.sort(selected_indices)


def prepare_split(
    *,
    source: str,
    model: str,
    q: int,
    seed: int,
    data_root: Path | None,
    synthetic: SyntheticSpec,
    train_cap: int,
    test_cap: int,
):
    """Load, split, subsample, and preprocess one configuration.

    PCA and scalers are fitted only on the capped training subset. The
    validation set is transformed but is not used by the fixed-C comparisons.
    """
    X, y = load_dataset(
        source=source,
        model=model,
        seed=dataset_seed(
            source=source,
            model=model,
            seed=seed,
        ),
        data_root=data_root,
        synthetic=synthetic,
    )

    (
        training_indices,
        validation_indices,
        test_indices,
    ) = split_indices(
        y,
        seed=seed,
    )

    training_indices = stratified_subsample(
        training_indices,
        y,
        train_cap,
        seed,
    )

    test_indices = stratified_subsample(
        test_indices,
        y,
        test_cap,
        seed + 1,
    )

    (
        X_train,
        X_validation,
        X_test,
        explained_variance,
    ) = preprocess(
        X[training_indices],
        X[validation_indices],
        X[test_indices],
        q,
    )

    return {
        "X_train": X_train,
        "X_validation": X_validation,
        "X_test": X_test,
        "y_train": y[training_indices],
        "y_validation": y[validation_indices],
        "y_test": y[test_indices],
        "explained_variance": explained_variance,
        "training_indices": training_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
    }


def score_precomputed_kernel(
    *,
    K_train: np.ndarray,
    K_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    c: float = 1.0,
) -> dict[str, object]:
    """Train and evaluate an SVM with a precomputed kernel."""
    classifier = SVC(
        kernel="precomputed",
        C=c,
        random_state=seed,
    )

    classifier.fit(
        K_train,
        y_train,
    )

    predictions = classifier.predict(K_test)

    scores = decision_scores(
        classifier,
        K_test,
    )

    return compute_metrics(
        y_test,
        predictions,
        scores,
    )


def score_linear_svm(
    *,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    c: float = 1.0,
) -> dict[str, object]:
    """Train and evaluate the Tier-1 linear SVM baseline."""
    classifier = SVC(
        kernel="linear",
        C=c,
        random_state=seed,
    )

    classifier.fit(
        X_train,
        y_train,
    )

    predictions = classifier.predict(X_test)

    scores = decision_scores(
        classifier,
        X_test,
    )

    return compute_metrics(
        y_test,
        predictions,
        scores,
    )


def compute_qubit_kernels(
    *,
    X_train: np.ndarray,
    X_test: np.ndarray,
    normalization: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute and consistently normalize the manuscript qubit kernel."""
    K_train_raw = fidelity_kernel(X_train)

    K_test_raw = fidelity_kernel(
        X_test,
        X_train,
    )

    rank = effective_rank(K_train_raw,psd_tolerance=1e-4,)

    K_train, K_test = normalize_train_test_kernels(
        K_train_raw,
        K_test_raw,
        method=normalization,
    )

    return K_train, K_test, rank


def compute_photonic_kernels(
    *,
    X_train: np.ndarray,
    X_test: np.ndarray,
    q: int,
    n_photons: int,
    seed: int,
    normalization: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute and consistently normalize the photonic fidelity kernel."""
    K_train_raw, K_test_raw = photonic_fidelity_kernels(
        X_train,
        X_test,
        q,
        n_photons=n_photons,
        seed=seed,
    )

    rank = effective_rank(K_train_raw,psd_tolerance=1e-4,)

    K_train, K_test = normalize_train_test_kernels(
        K_train_raw,
        K_test_raw,
        method=normalization,
    )

    return K_train, K_test, rank


def mean_metric(
    rows: list[dict[str, object]],
    method: str,
    metric: str,
) -> float:
    """Return the finite mean of one metric for one method."""
    values = np.asarray(
        [row[metric] for row in rows if row["method"] == method],
        dtype=np.float64,
    )

    values = values[np.isfinite(values)]

    if values.size == 0:
        return float("nan")

    return float(np.mean(values))


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write rows using the union of all dictionary keys."""
    if not rows:
        raise ValueError(f"No rows to write to {path}.")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = sorted({key for row in rows for key in row})

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    payload: dict[str, object],
) -> None:
    """Write one JSON artifact."""
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def build_table1(
    args: argparse.Namespace,
    synthetic: SyntheticSpec,
    seeds: list[int],
    metadata: dict[str, object],
) -> dict[str, object]:
    """Generate the photonic Table 1-style comparison."""
    long_rows: list[dict[str, object]] = []

    for model, q in PHOTONIC_CONFIGS:
        for seed in seeds:
            split = prepare_split(
                source=args.source,
                model=model,
                q=q,
                seed=seed,
                data_root=args.data_root,
                synthetic=synthetic,
                train_cap=args.train_cap,
                test_cap=args.test_cap,
            )

            base_row = {
                "model": model,
                "q": q,
                "seed": seed,
                "train_samples": int(len(split["y_train"])),
                "validation_samples": int(len(split["y_validation"])),
                "test_samples": int(len(split["y_test"])),
                "train_positive_ratio": float(np.mean(split["y_train"])),
                "test_positive_ratio": float(np.mean(split["y_test"])),
                "pca_explained_variance": float(split["explained_variance"]),
                "kernel_normalization": (args.kernel_normalization),
            }

            linear_metrics = score_linear_svm(
                X_train=split["X_train"],
                X_test=split["X_test"],
                y_train=split["y_train"],
                y_test=split["y_test"],
                seed=seed,
            )

            long_rows.append(
                {
                    **base_row,
                    "method": "linear_c1",
                    "effective_rank": effective_rank(
                        split["X_train"] @ split["X_train"].T
                    ),
                    **linear_metrics,
                }
            )

            (
                K_qubit_train,
                K_qubit_test,
                qubit_rank,
            ) = compute_qubit_kernels(
                X_train=split["X_train"],
                X_test=split["X_test"],
                normalization=args.kernel_normalization,
            )

            qubit_metrics = score_precomputed_kernel(
                K_train=K_qubit_train,
                K_test=K_qubit_test,
                y_train=split["y_train"],
                y_test=split["y_test"],
                seed=seed,
            )

            long_rows.append(
                {
                    **base_row,
                    "method": "qsvm_qubit",
                    "effective_rank": qubit_rank,
                    **qubit_metrics,
                }
            )

            (
                K_photonic_train,
                K_photonic_test,
                photonic_rank,
            ) = compute_photonic_kernels(
                X_train=split["X_train"],
                X_test=split["X_test"],
                q=q,
                n_photons=args.n_photons,
                seed=seed,
                normalization=args.kernel_normalization,
            )

            photonic_metrics = score_precomputed_kernel(
                K_train=K_photonic_train,
                K_test=K_photonic_test,
                y_train=split["y_train"],
                y_test=split["y_test"],
                seed=seed,
            )

            long_rows.append(
                {
                    **base_row,
                    "method": "qsvm_photonic",
                    "effective_rank": photonic_rank,
                    **photonic_metrics,
                }
            )

            print(
                f"[table1] {model} q={q} seed={seed} "
                f"linear_f1={linear_metrics['f1']:.3f} "
                f"qubit_f1={qubit_metrics['f1']:.3f} "
                f"photonic_f1={photonic_metrics['f1']:.3f}",
                flush=True,
            )

    grouped: dict[
        tuple[str, int],
        list[dict[str, object]],
    ] = defaultdict(list)

    for row in long_rows:
        grouped[
            (
                str(row["model"]),
                int(row["q"]),
            )
        ].append(row)

    summary_rows: list[dict[str, object]] = []

    for model, q in PHOTONIC_CONFIGS:
        rows = grouped[(model, q)]

        linear_f1 = mean_metric(
            rows,
            "linear_c1",
            "f1",
        )

        qubit_f1 = mean_metric(
            rows,
            "qsvm_qubit",
            "f1",
        )

        photonic_f1 = mean_metric(
            rows,
            "qsvm_photonic",
            "f1",
        )

        summary_rows.append(
            {
                "model": model,
                "q": q,
                "seed_count": len(seeds),
                "linear_c1_f1": linear_f1,
                "linear_c1_auc": mean_metric(
                    rows,
                    "linear_c1",
                    "auc",
                ),
                "qsvm_qubit_f1": qubit_f1,
                "qsvm_qubit_auc": mean_metric(
                    rows,
                    "qsvm_qubit",
                    "auc",
                ),
                "qsvm_qubit_effective_rank": (
                    mean_metric(
                        rows,
                        "qsvm_qubit",
                        "effective_rank",
                    )
                ),
                "qsvm_photonic_f1": photonic_f1,
                "qsvm_photonic_auc": mean_metric(
                    rows,
                    "qsvm_photonic",
                    "auc",
                ),
                "qsvm_photonic_effective_rank": (
                    mean_metric(
                        rows,
                        "qsvm_photonic",
                        "effective_rank",
                    )
                ),
                "photonic_vs_linear_f1_delta": (photonic_f1 - linear_f1),
                "photonic_vs_qubit_f1_delta": (photonic_f1 - qubit_f1),
            }
        )

    prefix = "photonic_table1"

    long_path = args.results_dir / f"{prefix}_long.csv"

    summary_path = args.results_dir / f"{prefix}_summary.csv"

    write_csv(
        long_path,
        long_rows,
    )

    write_csv(
        summary_path,
        summary_rows,
    )

    strict_wins = sum(row["photonic_vs_linear_f1_delta"] > 0.0 for row in summary_rows)

    ties = sum(
        np.isclose(
            row["photonic_vs_linear_f1_delta"],
            0.0,
        )
        for row in summary_rows
    )

    aggregate = {
        "photonic_strictly_beats_linear": (f"{strict_wins}/{len(summary_rows)}"),
        "photonic_ties_linear": (f"{ties}/{len(summary_rows)}"),
        "mean_photonic_vs_linear_f1_delta": float(
            np.mean([row["photonic_vs_linear_f1_delta"] for row in summary_rows])
        ),
        "mean_photonic_vs_qubit_f1_delta": float(
            np.mean([row["photonic_vs_qubit_f1_delta"] for row in summary_rows])
        ),
    }

    payload = {
        "artifact": prefix,
        "paper_counterpart": "Table 1",
        "paper_pointer": PAPER_POINTERS["table1"],
        "interpretation": (
            "Native photonic-kernel comparison on capped data. "
            "Not a reproduction of paper values."
        ),
        "aggregate": aggregate,
        "paths": {
            "long_csv": str(long_path),
            "summary_csv": str(summary_path),
        },
        "summary_rows": summary_rows,
        **metadata,
    }

    write_json(
        args.results_dir / f"{prefix}.json",
        payload,
    )

    write_table1_markdown(
        args.results_dir / f"{prefix}.md",
        summary_rows,
        payload,
    )

    print(
        json.dumps(
            aggregate,
            indent=2,
        )
    )

    return payload


def write_table1_markdown(
    path: Path,
    summary_rows: list[dict[str, object]],
    payload: dict[str, object],
) -> None:
    """Write the human-readable photonic Table 1 summary."""
    lines = [
        "# Photonic Table 1-style comparison",
        "",
        (
            "This artifact compares the native photonic fidelity kernel with "
            "the manuscript qubit fidelity kernel and the linear C=1 SVM."
        ),
        "",
        (
            "Results use capped data for photonic simulation and do not "
            "reproduce the paper's numerical results."
        ),
        "",
        f"Paper methodology pointer: {payload['paper_pointer']}",
        "",
        (
            "| Model | q | Linear F1 | Qubit F1 | Photonic F1 | "
            "Photonic AUC | Photonic rank | Photonic vs linear |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in summary_rows:
        lines.append(
            f"| {row['model']} "
            f"| {row['q']} "
            f"| {row['linear_c1_f1']:.3f} "
            f"| {row['qsvm_qubit_f1']:.3f} "
            f"| {row['qsvm_photonic_f1']:.3f} "
            f"| {row['qsvm_photonic_auc']:.3f} "
            f"| {row['qsvm_photonic_effective_rank']:.2f} "
            f"| {row['photonic_vs_linear_f1_delta']:+.3f} |"
        )

    aggregate = payload["aggregate"]

    lines.extend(
        [
            "",
            (
                "Strict photonic wins over linear: "
                f"{aggregate['photonic_strictly_beats_linear']}."
            ),
            (f"Photonic ties with linear: {aggregate['photonic_ties_linear']}."),
            (
                "Mean photonic versus linear F1 delta: "
                f"{aggregate['mean_photonic_vs_linear_f1_delta']:+.3f}."
            ),
            (
                "Mean photonic versus qubit F1 delta: "
                f"{aggregate['mean_photonic_vs_qubit_f1_delta']:+.3f}."
            ),
        ]
    )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def build_figure4(
    args: argparse.Namespace,
    synthetic: SyntheticSpec,
    metadata: dict[str, object],
) -> dict[str, object]:
    """Generate a class-sorted photonic Gram-matrix heatmap."""
    model = args.fig4_model
    q = args.fig4_q
    seed = args.fig4_seed

    X, y = load_dataset(
        source=args.source,
        model=model,
        seed=dataset_seed(
            source=args.source,
            model=model,
            seed=seed,
        ),
        data_root=args.data_root,
        synthetic=synthetic,
    )

    (
        training_indices,
        validation_indices,
        test_indices,
    ) = split_indices(
        y,
        seed=seed,
    )

    selected_indices = stratified_subsample(
        training_indices,
        y,
        args.fig4_samples,
        seed,
    )

    # Fit preprocessing on the selected training subset before sorting.
    (
        X_selected,
        _,
        _,
        explained_variance,
    ) = preprocess(
        X[selected_indices],
        X[validation_indices],
        X[test_indices[:2]],
        q,
    )

    y_selected = y[selected_indices]

    order = np.argsort(
        y_selected,
        kind="stable",
    )

    X_sorted = X_selected[order]

    y_sorted = y_selected[order]

    K_train_raw, _ = photonic_fidelity_kernels(
        X_sorted,
        X_sorted[:2],
        q,
        n_photons=args.n_photons,
        seed=seed,
    )

    boundary = int(np.sum(y_sorted == 0))

    lower_color_limit = float(
        np.quantile(
            K_train_raw,
            0.01,
        )
    )

    upper_color_limit = float(
        np.quantile(
            K_train_raw,
            0.99,
        )
    )

    figure, axis = plt.subplots(figsize=(5.4, 4.6))

    image = axis.imshow(
        K_train_raw,
        cmap="viridis",
        vmin=lower_color_limit,
        vmax=upper_color_limit,
    )

    if 0 < boundary < len(y_sorted):
        axis.axhline(
            boundary - 0.5,
            color="white",
            linewidth=0.8,
            linestyle="--",
        )

        axis.axvline(
            boundary - 0.5,
            color="white",
            linewidth=0.8,
            linestyle="--",
        )

    axis.set_title(
        "Photonic fidelity Gram matrix\n"
        f"{model}, q={q}, "
        f"{len(y_sorted)} class-sorted samples"
    )

    axis.set_xlabel("Sample sorted by class")

    axis.set_ylabel("Sample sorted by class")

    figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
        label="Raw fidelity K(x, y)",
    )

    figure.tight_layout()

    prefix = "photonic_figure4"

    png_path = args.results_dir / f"{prefix}.png"

    matrix_path = args.results_dir / f"{prefix}_kernel_matrix.csv"

    figure.savefig(
        png_path,
        dpi=130,
    )

    plt.close(figure)

    np.savetxt(
        matrix_path,
        K_train_raw,
        delimiter=",",
    )

    off_diagonal_mask = ~np.eye(
        len(K_train_raw),
        dtype=bool,
    )

    off_diagonal_values = K_train_raw[off_diagonal_mask]

    payload = {
        "artifact": prefix,
        "paper_counterpart": "Figure 4",
        "paper_pointer": PAPER_POINTERS["figure4"],
        "model": model,
        "q": q,
        "seed": seed,
        "sample_count": int(len(y_sorted)),
        "class_0_count": int(np.sum(y_sorted == 0)),
        "class_1_count": int(np.sum(y_sorted == 1)),
        "class_boundary": boundary,
        "pca_explained_variance": float(explained_variance),
        "kernel_effective_rank": (effective_rank(K_train_raw,psd_tolerance=1e-4,)),
        "kernel_statistics": {
            "minimum": float(K_train_raw.min()),
            "maximum": float(K_train_raw.max()),
            "mean": float(K_train_raw.mean()),
            "off_diagonal_mean": float(off_diagonal_values.mean()),
            "off_diagonal_std": float(off_diagonal_values.std()),
            "trace": float(np.trace(K_train_raw)),
        },
        "paths": {
            "png": str(png_path),
            "kernel_csv": str(matrix_path),
        },
        **metadata,
    }

    write_json(
        args.results_dir / f"{prefix}.json",
        payload,
    )

    markdown = [
        "# Photonic Figure 4-style Gram matrix",
        "",
        (
            "Class-sorted raw photonic fidelity Gram matrix for "
            f"{model}, q={q}, using {len(y_sorted)} samples."
        ),
        "",
        (
            "This is a native photonic feature-map result on capped data, "
            "not a reproduction of the paper's numerical matrix."
        ),
        "",
        f"Paper methodology pointer: {payload['paper_pointer']}",
        "",
        (f"Photonic effective rank: {payload['kernel_effective_rank']:.3f}."),
        (
            "Off-diagonal mean: "
            f"{payload['kernel_statistics']['off_diagonal_mean']:.4f}."
        ),
    ]

    (args.results_dir / f"{prefix}.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )

    print(
        f"[figure4] wrote {png_path} "
        f"effective_rank="
        f"{payload['kernel_effective_rank']:.2f}"
    )

    return payload


def build_effective_rank(
    args: argparse.Namespace,
    synthetic: SyntheticSpec,
    seeds: list[int],
    metadata: dict[str, object],
) -> dict[str, object]:
    """Generate photonic effective rank as a function of q."""
    model = args.effrank_model
    rows: list[dict[str, object]] = []

    for q in args.effrank_qs:
        seed_values: list[float] = []
        sample_counts: list[int] = []
        positive_ratios: list[float] = []

        for seed in seeds:
            X, y = load_dataset(
                source=args.source,
                model=model,
                seed=dataset_seed(
                    source=args.source,
                    model=model,
                    seed=seed,
                ),
                data_root=args.data_root,
                synthetic=synthetic,
            )

            (
                training_indices,
                validation_indices,
                test_indices,
            ) = split_indices(
                y,
                seed=seed,
            )

            selected_indices = stratified_subsample(
                training_indices,
                y,
                args.effrank_samples,
                seed,
            )

            (
                X_train,
                _,
                _,
                _,
            ) = preprocess(
                X[selected_indices],
                X[validation_indices],
                X[test_indices[:2]],
                q,
            )

            K_train_raw, _ = photonic_fidelity_kernels(
                X_train,
                X_train[:2],
                q,
                n_photons=(args.n_photons),
                seed=seed,
            )

            seed_values.append(effective_rank(K_train_raw,psd_tolerance=1e-4,))

            sample_counts.append(len(selected_indices))

            positive_ratios.append(float(np.mean(y[selected_indices])))

        rows.append(
            {
                "model": model,
                "q": q,
                "seed_count": len(seeds),
                "sample_count": int(min(sample_counts)),
                "positive_ratio_mean": float(np.mean(positive_ratios)),
                "photonic_effective_rank_mean": float(np.mean(seed_values)),
                "photonic_effective_rank_std": float(np.std(seed_values)),
                "full_training_comparison": False,
            }
        )

        print(
            f"[effrank] q={q} effective_rank={np.mean(seed_values):.2f}",
            flush=True,
        )

    prefix = "photonic_effrank"

    summary_path = args.results_dir / f"{prefix}_summary.csv"

    png_path = args.results_dir / f"{prefix}.png"

    write_csv(
        summary_path,
        rows,
    )

    figure, axis = plt.subplots(figsize=(5.2, 3.8))

    axis.errorbar(
        [row["q"] for row in rows],
        [row["photonic_effective_rank_mean"] for row in rows],
        yerr=[row["photonic_effective_rank_std"] for row in rows],
        marker="o",
        capsize=3,
    )

    axis.set_xlabel("Optical modes q")

    axis.set_ylabel("Photonic-kernel effective rank")

    axis.set_title(f"Photonic fidelity-kernel effective rank\n{model}")

    axis.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        png_path,
        dpi=130,
    )

    plt.close(figure)

    payload = {
        "artifact": prefix,
        "paper_counterpart": ("Figure 2 and Table 5"),
        "paper_pointer": PAPER_POINTERS["effrank"],
        "model": model,
        "interpretation": (
            "Effective rank of a native photonic kernel on a capped, "
            "stratified subset. Not directly comparable with full-training "
            "paper values."
        ),
        "rows": rows,
        "paths": {
            "summary_csv": str(summary_path),
            "png": str(png_path),
        },
        **metadata,
    }

    write_json(
        args.results_dir / f"{prefix}.json",
        payload,
    )

    markdown_lines = [
        "# Photonic effective rank versus q",
        "",
        (
            "The values below are calculated on capped, stratified subsets. "
            "They are not directly comparable with effective ranks calculated "
            "on the paper's complete training set."
        ),
        "",
        f"Paper methodology pointer: {payload['paper_pointer']}",
        "",
    ]

    markdown_lines.extend(
        [
            (
                f"- q={row['q']}: "
                f"{row['photonic_effective_rank_mean']:.2f} "
                f"± {row['photonic_effective_rank_std']:.2f}"
            )
            for row in rows
        ]
    )

    (args.results_dir / f"{prefix}.md").write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )

    print(f"[effrank] wrote {png_path}")

    return payload


def parse_artifact_names(
    raw: str,
) -> set[str]:
    """Parse and validate artifact names."""
    if raw.strip().lower:
        return set(ARTIFACT_NAMES)

    names = {part.strip().lower() for part in raw.split(",") if part.strip()}

    if not names:
        raise argparse.ArgumentTypeError("At least one artifact is required.")

    unknown = names - ARTIFACT_NAMES

    if unknown:
        raise argparse.ArgumentTypeError(
            "Unknown artifacts: " + ", ".join(sorted(unknown))
        )

    return names


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--source",
        choices=(
            "synthetic",
            "synthetic_file",
        ),
        default="synthetic_file",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=(PROJECT_ROOT / "data" / "synthetic_qml_mimic_cxr_embeddings"),
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=(PROJECT_ROOT / "results"),
    )

    parser.add_argument(
        "--seeds",
        default="0,1,2",
    )

    parser.add_argument(
        "--n-photons",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--kernel-normalization",
        choices=(
            "trace",
            "none",
        ),
        default="trace",
    )

    parser.add_argument(
        "--train-cap",
        type=int,
        default=220,
    )

    parser.add_argument(
        "--test-cap",
        type=int,
        default=100,
    )

    # In-memory synthetic-data parameters.
    parser.add_argument(
        "--n-samples",
        type=int,
        default=600,
    )

    parser.add_argument(
        "--ambient-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--minority-frac",
        type=float,
        default=0.28,
    )

    parser.add_argument(
        "--signal",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=1.0,
    )

    # Figure 4 settings.
    parser.add_argument(
        "--fig4-model",
        choices=MODEL_NAMES,
        default="medsiglip-448",
    )

    parser.add_argument(
        "--fig4-q",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--fig4-seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--fig4-samples",
        type=int,
        default=200,
    )

    # Effective-rank settings.
    parser.add_argument(
        "--effrank-model",
        choices=MODEL_NAMES,
        default="medsiglip-448",
    )

    parser.add_argument(
        "--effrank-qs",
        default="4,6,8,10",
    )

    parser.add_argument(
        "--effrank-samples",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--only",
        type=parse_artifact_names,
        default=set(ARTIFACT_NAMES),
        help=("Comma-separated subset of table1, figure4, effrank, or all."),
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    """Validate argument consistency."""
    if args.n_photons <= 0:
        raise ValueError("--n-photons must be positive.")

    if args.train_cap <= 0:
        raise ValueError("--train-cap must be positive.")

    if args.test_cap <= 0:
        raise ValueError("--test-cap must be positive.")

    if args.fig4_q <= 0:
        raise ValueError("--fig4-q must be positive.")

    if args.fig4_samples <= 0:
        raise ValueError("--fig4-samples must be positive.")

    if args.effrank_samples <= 0:
        raise ValueError("--effrank-samples must be positive.")

    if any(q <= 0 for q in args.effrank_qs):
        raise ValueError("All effective-rank q values must be positive.")

    if args.source == "synthetic_file":
        if not args.data_root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {args.data_root}")

        index_path = args.data_root / "synthetic_dataset_index.json"

        if not index_path.is_file():
            raise FileNotFoundError(f"Synthetic dataset index not found: {index_path}")


def main() -> None:
    """Generate the selected photonic artifacts."""
    args = parse_args()

    args.effrank_qs = parse_ints(args.effrank_qs)

    seeds = parse_ints(args.seeds)

    validate_args(args)

    synthetic = SyntheticSpec(
        n_samples=args.n_samples,
        ambient_dim=args.ambient_dim,
        latent_dim=args.latent_dim,
        minority_frac=args.minority_frac,
        signal=args.signal,
        noise=args.noise,
    )

    args.results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "kernel": {
            "type": "native photonic fidelity kernel",
            "backend": "MerLin",
            "n_photons": args.n_photons,
            "gate_by_gate_bsp_translation": False,
            "kernel_normalization": (args.kernel_normalization),
        },
        "data": {
            "source": args.source,
            "synthetic_surrogate": True,
            "data_root": (
                str(args.data_root) if args.source == "synthetic_file" else None
            ),
            "seeds": seeds,
            "medical_semantics": False,
        },
        "subsampling": {
            "train_cap": args.train_cap,
            "test_cap": args.test_cap,
            "reason": (
                "Photonic Gram-matrix construction scales quadratically "
                "with the number of samples."
            ),
        },
    }

    if "table1" in args.only:
        build_table1(
            args,
            synthetic,
            seeds,
            metadata,
        )

    if "figure4" in args.only:
        build_figure4(
            args,
            synthetic,
            metadata,
        )

    if "effrank" in args.only:
        build_effective_rank(
            args,
            synthetic,
            seeds,
            metadata,
        )

    print("Photonic artifacts completed.")


if __name__ == "__main__":
    main()
