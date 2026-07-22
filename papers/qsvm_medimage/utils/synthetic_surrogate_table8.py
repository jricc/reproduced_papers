#!/usr/bin/env python3
"""Compute a Table 8-style comparison between 1-DOF and 3-DOF QSVMs.

The script follows the protocol described in the manuscript:

    q = 8
    reps = 1
    trace normalization
    C = 1
    seed = 0

The two feature maps are:

    1-DOF:
        one Ry rotation per qubit, followed by a CNOT ring.

    3-DOF:
        one Rz-Ry-Rz sequence per qubit, followed by the same CNOT ring.

The 1-DOF circuit receives q PCA components.

The 3-DOF circuit receives 3*q PCA components, providing three independent
input angles per qubit.

When synthetic data are used, this script reproduces the structure of the
Table 8 experiment, not the numerical results obtained from the inaccessible
MIMIC-CXR-derived embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = PROJECT_ROOT.parents[1]

for root in (PROJECT_ROOT, REPRO_ROOT):
    root_string = str(root)

    if root_string not in sys.path:
        sys.path.insert(0, root_string)


from lib.quantum_kernel import (
    _apply_cnot,
    _apply_single_qubit_gate,
    _ry,
    fidelity_kernel,
)
from lib.svm_pipeline import (
    normalize_train_test_kernels,
    preprocess,
    split_indices,
)
from synthetic_surrogate_table1 import (
    SyntheticSpec,
    compute_metrics,
    load_dataset,
)

PAPER_TABLE8_POINTER = "https://arxiv.org/html/2604.24597v1#S4.T8"

TABLE8_MODELS: tuple[str, ...] = (
    "medsiglip-448",
    "rad-dino",
    "vit-patch32-cls",
)

TABLE8_CIRCUITS: tuple[str, ...] = (
    "1-DOF",
    "3-DOF",
)

MODEL_DISPLAY = {
    "medsiglip-448": "MedSigLIP",
    "rad-dino": "RAD-DINO",
    "vit-patch32-cls": "ViT-p32",
}


def _rz(theta: float) -> np.ndarray:
    """Return the Rz gate used only by the 3-DOF ablation."""
    return np.array(
        [
            [
                np.exp(-0.5j * theta),
                0.0,
            ],
            [
                0.0,
                np.exp(0.5j * theta),
            ],
        ],
        dtype=np.complex128,
    )


def pca_dim_for_circuit(
    q: int,
    circuit: str,
) -> int:
    """Return the number of PCA components required by each circuit.

    The 1-DOF circuit uses one input angle per qubit.

    The 3-DOF circuit uses three independent input angles per qubit:

        first q values:
            first Rz layer;

        next q values:
            Ry layer;

        final q values:
            second Rz layer.
    """
    if q <= 0:
        raise ValueError("q must be positive.")

    if circuit == "1-DOF":
        return q

    if circuit == "3-DOF":
        return 3 * q

    raise ValueError(f"Unknown circuit: {circuit!r}.")


def _validate_3dof_features(
    x: np.ndarray,
    q: int,
    reps: int,
) -> np.ndarray:
    """Validate one input vector for the 3-DOF circuit."""
    features = np.asarray(
        x,
        dtype=np.float64,
    )

    if q <= 0:
        raise ValueError("q must be positive.")

    if reps < 1:
        raise ValueError("reps must be at least 1.")

    if features.ndim != 1:
        raise ValueError("A 3-DOF input must be a one-dimensional vector.")

    expected_dimension = 3 * q

    if features.size != expected_dimension:
        raise ValueError(
            "The 3-DOF circuit expects "
            f"{expected_dimension} features for q={q}, "
            f"but received {features.size}."
        )

    if not np.all(np.isfinite(features)):
        raise ValueError("The 3-DOF input contains non-finite values.")

    return features


def bsp_3dof_statevector(
    x: np.ndarray,
    *,
    q: int,
    reps: int = 1,
) -> np.ndarray:
    """Encode one sample with Rz-Ry-Rz rotations and a CNOT ring.

    For qubit d, the three angles are:

        x[d]
        x[d + q]
        x[d + 2*q]

    One repetition applies:

        Rz(x[d])
        Ry(x[d + q])
        Rz(x[d + 2*q])

    on every qubit, followed by the same CNOT ring used by the 1-DOF circuit.

    No Hadamard gate is used.
    """
    features = _validate_3dof_features(
        x,
        q,
        reps,
    )

    state = np.zeros(
        2**q,
        dtype=np.complex128,
    )
    state[0] = 1.0

    for _ in range(reps):
        # Apply three independent encoding rotations per qubit.
        for qubit in range(q):
            first_rz_angle = float(features[qubit])
            ry_angle = float(features[qubit + q])
            second_rz_angle = float(features[qubit + 2 * q])

            state = _apply_single_qubit_gate(
                state,
                _rz(first_rz_angle),
                qubit,
                q,
            )

            state = _apply_single_qubit_gate(
                state,
                _ry(ry_angle),
                qubit,
                q,
            )

            state = _apply_single_qubit_gate(
                state,
                _rz(second_rz_angle),
                qubit,
                q,
            )

        # Connect neighbouring qubits.
        for control in range(q - 1):
            state = _apply_cnot(
                state,
                control,
                control + 1,
                q,
            )

        # Close the chain into a ring.
        if q > 1:
            state = _apply_cnot(
                state,
                q - 1,
                0,
                q,
            )

    return state


def _validate_3dof_dataset(
    data: np.ndarray,
    q: int,
    name: str,
) -> np.ndarray:
    """Validate a matrix of 3*q-dimensional inputs."""
    samples = np.asarray(
        data,
        dtype=np.float64,
    )

    if samples.ndim != 2:
        raise ValueError(f"{name} must have shape (n_samples, 3*q).")

    if samples.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")

    expected_dimension = 3 * q

    if samples.shape[1] != expected_dimension:
        raise ValueError(
            f"{name} has {samples.shape[1]} features, "
            f"but the 3-DOF circuit requires "
            f"{expected_dimension} for q={q}."
        )

    if not np.all(np.isfinite(samples)):
        raise ValueError(f"{name} contains non-finite values.")

    return samples


def fidelity_kernel_3dof(
    data1: np.ndarray,
    *,
    q: int,
    reps: int,
    data2: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the fidelity kernel for the 3-DOF ablation.

    If data2 is omitted, return a square training Gram matrix.

    If data2 is provided, return a rectangular data1-versus-data2 matrix.
    """
    samples1 = _validate_3dof_dataset(
        data1,
        q,
        "data1",
    )

    states1 = np.stack(
        [
            bsp_3dof_statevector(
                row,
                q=q,
                reps=reps,
            )
            for row in samples1
        ]
    )

    if data2 is None:
        overlaps = states1.conj() @ states1.T
        kernel = np.abs(overlaps) ** 2

        # Remove small numerical asymmetries.
        kernel = 0.5 * (kernel + kernel.T)

        diagonal_error = float(np.max(np.abs(np.diag(kernel) - 1.0)))

        if diagonal_error > 1e-10:
            raise RuntimeError(
                "The 3-DOF encoded states are not normalized. "
                f"Maximum diagonal error: {diagonal_error:.3e}."
            )

        np.fill_diagonal(
            kernel,
            1.0,
        )

        return np.asarray(
            kernel,
            dtype=np.float64,
        )

    samples2 = _validate_3dof_dataset(
        data2,
        q,
        "data2",
    )

    states2 = np.stack(
        [
            bsp_3dof_statevector(
                row,
                q=q,
                reps=reps,
            )
            for row in samples2
        ]
    )

    overlaps = states1.conj() @ states2.T
    kernel = np.abs(overlaps) ** 2

    return np.asarray(
        kernel,
        dtype=np.float64,
    )


def score_qsvm_circuit(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    q: int,
    circuit: str,
    reps: int,
    c: float,
    seed: int,
    kernel_normalization: str,
) -> dict[str, object]:
    """Train and evaluate one fixed-C QSVM circuit."""
    if circuit == "1-DOF":
        training_kernel_raw = fidelity_kernel(
            X_train,
            reps=reps,
        )

        test_kernel_raw = fidelity_kernel(
            X_test,
            X_train,
            reps=reps,
        )

    elif circuit == "3-DOF":
        training_kernel_raw = fidelity_kernel_3dof(
            X_train,
            q=q,
            reps=reps,
        )

        test_kernel_raw = fidelity_kernel_3dof(
            X_test,
            q=q,
            reps=reps,
            data2=X_train,
        )

    else:
        raise ValueError(f"Unknown circuit: {circuit!r}.")

    training_kernel, test_kernel = normalize_train_test_kernels(
        training_kernel_raw,
        test_kernel_raw,
        method=kernel_normalization,
    )

    classifier = SVC(
        kernel="precomputed",
        C=c,
        random_state=seed,
    )

    classifier.fit(
        training_kernel,
        y_train,
    )

    predictions = classifier.predict(test_kernel)

    decision_scores = classifier.decision_function(test_kernel)

    return compute_metrics(
        y_test,
        predictions,
        decision_scores,
    )


def compute_table8_rows(
    *,
    source: str,
    model: str,
    q: int,
    reps: int,
    c: float,
    seed: int,
    data_root: Path | None,
    synthetic: SyntheticSpec,
    kernel_normalization: str,
) -> list[dict[str, object]]:
    """Compute the 1-DOF and 3-DOF rows for one embedding model."""
    X, y = load_dataset(
        source=source,
        model=model,
        seed=seed,
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

    y_train = y[training_indices]
    y_test = y[test_indices]

    rows: list[dict[str, object]] = []

    for circuit in TABLE8_CIRCUITS:
        pca_dimension = pca_dim_for_circuit(
            q,
            circuit,
        )

        (
            X_train,
            _,
            X_test,
            explained_variance_ratio,
        ) = preprocess(
            X[training_indices],
            X[validation_indices],
            X[test_indices],
            pca_dimension,
        )

        metrics = score_qsvm_circuit(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            q=q,
            circuit=circuit,
            reps=reps,
            c=c,
            seed=seed,
            kernel_normalization=kernel_normalization,
        )

        rows.append(
            {
                "source": source,
                "synthetic_surrogate": source != "real",
                "model": model,
                "model_display": MODEL_DISPLAY[model],
                "circuit": circuit,
                "q": q,
                "pca_dim": pca_dimension,
                "reps": reps,
                "C": c,
                "seed": seed,
                "kernel_normalization": kernel_normalization,
                "train_samples": int(len(y_train)),
                "test_samples": int(len(y_test)),
                "test_class_0": int(np.sum(y_test == 0)),
                "test_class_1": int(np.sum(y_test == 1)),
                "pca_variance_percent": (100.0 * float(explained_variance_ratio)),
                "accuracy": metrics["accuracy"],
                "auc": metrics["auc"],
                "f1": metrics["f1"],
            }
        )

    return rows


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write result rows to CSV."""
    if not rows:
        raise ValueError(f"No rows to write to {path}.")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def format_metric(
    value: float,
) -> str:
    """Format one metric with three decimal places."""
    return f"{value:.3f}"


def write_markdown(
    path: Path,
    *,
    payload: dict[str, object],
) -> None:
    """Write the human-readable Table 8 summary."""
    lines = [
        "# Synthetic surrogate Table 8 pipeline",
        "",
        (
            "This artifact follows the Table 8 protocol on the selected "
            "data source. Results obtained with synthetic data do not reproduce "
            "the numerical results obtained with the gated MIMIC-CXR-derived "
            "embeddings."
        ),
        "",
        (
            "The 1-DOF circuit uses Ry encoding and a CNOT ring. "
            "The 3-DOF ablation uses Rz-Ry-Rz encoding and the same CNOT ring."
        ),
        "",
        (
            "The 1-DOF circuit uses q PCA components. "
            "The 3-DOF circuit uses 3*q PCA components."
        ),
        "",
        f"Paper methodology pointer: {PAPER_TABLE8_POINTER}",
        "",
        "| Model | Circuit | PCA dim | Acc | AUC | F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    summary_rows = payload["summary_rows"]

    if not isinstance(summary_rows, list):
        raise TypeError("payload['summary_rows'] must be a list.")

    for row in summary_rows:
        if not isinstance(row, dict):
            raise TypeError("Each summary row must be a dictionary.")

        lines.append(
            "| {model} | {circuit} | {pca_dim} | {accuracy} | {auc} | {f1} |".format(
                model=row["model_display"],
                circuit=row["circuit"],
                pca_dim=row["pca_dim"],
                accuracy=format_metric(float(row["accuracy"])),
                auc=format_metric(float(row["auc"])),
                f1=format_metric(float(row["f1"])),
            )
        )

    lines.extend(
        [
            "",
            "Data and protocol metadata:",
            "",
            "```json",
            json.dumps(
                payload["data"],
                indent=2,
                sort_keys=True,
            ),
            "```",
        ]
    )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def default_prefix(
    source: str,
) -> str:
    """Return the output prefix associated with the data source."""
    if source == "synthetic":
        return "synthetic_surrogate_table8"

    if source == "synthetic_file":
        return "synthetic_file_table8"

    return "real_table8"


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
            "real",
        ),
        default="synthetic",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
    )

    parser.add_argument(
        "--output-prefix",
        default=None,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--q",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--reps",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--kernel-normalization",
        choices=(
            "trace",
            "none",
        ),
        default="trace",
    )

    # Parameters used only when data are generated in memory.
    parser.add_argument(
        "--n-samples",
        type=int,
        default=300,
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
        default=0.20,
    )

    parser.add_argument(
        "--signal",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=1.0,
    )

    return parser.parse_args()


def main() -> None:
    """Generate the Table 8 CSV, JSON, and Markdown artifacts."""
    args = parse_args()

    if args.q <= 0:
        raise ValueError("--q must be positive.")

    if args.reps < 1:
        raise ValueError("--reps must be at least 1.")

    if args.C <= 0.0:
        raise ValueError("--C must be positive.")

    if args.source in {"synthetic_file", "real"} and args.data_root is None:
        raise ValueError(f"--data-root is required for source={args.source!r}.")

    synthetic = SyntheticSpec(
        n_samples=args.n_samples,
        ambient_dim=args.ambient_dim,
        latent_dim=args.latent_dim,
        minority_frac=args.minority_frac,
        signal=args.signal,
        noise=args.noise,
    )

    summary_rows: list[dict[str, object]] = []

    for model in TABLE8_MODELS:
        summary_rows.extend(
            compute_table8_rows(
                source=args.source,
                model=model,
                q=args.q,
                reps=args.reps,
                c=args.C,
                seed=args.seed,
                data_root=args.data_root,
                synthetic=synthetic,
                kernel_normalization=(args.kernel_normalization),
            )
        )

    prefix = args.output_prefix or default_prefix(args.source)

    args.results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = args.results_dir / f"{prefix}_summary.csv"

    json_path = args.results_dir / f"{prefix}.json"

    markdown_path = args.results_dir / f"{prefix}.md"

    write_csv(
        summary_path,
        summary_rows,
    )

    payload: dict[str, object] = {
        "artifact": prefix,
        "paper_table": "Table 8",
        "paper_pointer": PAPER_TABLE8_POINTER,
        "paths": {
            "summary_csv": str(summary_path),
            "json": str(json_path),
            "markdown": str(markdown_path),
        },
        "data": {
            "source": args.source,
            "synthetic_surrogate": (args.source != "real"),
            "synthetic_spec": (
                asdict(synthetic) if args.source == "synthetic" else None
            ),
            "data_root": (str(args.data_root) if args.data_root else None),
            "seed": args.seed,
            "q": args.q,
            "reps": args.reps,
            "C": args.C,
            "kernel_normalization": (args.kernel_normalization),
            "split": ("80/10/10 stratified via lib.svm_pipeline.split_indices"),
            "models": list(TABLE8_MODELS),
            "circuits": list(TABLE8_CIRCUITS),
            "circuit_definition": {
                "1-DOF": {
                    "encoding": "Ry",
                    "pca_dimension": "q",
                    "entanglement": "CNOT ring",
                },
                "3-DOF": {
                    "encoding": "Rz-Ry-Rz",
                    "pca_dimension": "3*q",
                    "entanglement": "CNOT ring",
                },
            },
        },
        "summary_rows": summary_rows,
    }

    json_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    write_markdown(
        markdown_path,
        payload=payload,
    )

    print(
        json.dumps(
            {
                "rows": len(summary_rows),
                "seed": args.seed,
            },
            indent=2,
        )
    )

    print(f"Wrote {summary_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
