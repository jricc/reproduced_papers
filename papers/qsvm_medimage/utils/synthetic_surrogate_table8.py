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

The 3-DOF circuit receives 3 times q PCA components, providing three
independent input angles per qubit.

When synthetic data are used, this script reproduces the structure of the
Table 8 experiment, not the numerical results obtained from the inaccessible
MIMIC-CXR-derived embeddings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = PROJECT_ROOT.parents[1]

for root in (
    PROJECT_ROOT,
    REPRO_ROOT,
):
    root_string = str(root)

    if root_string not in sys.path:
        sys.path.insert(
            0,
            root_string,
        )


from lib.quantum_kernel import (  # noqa: E402
    _apply_cnot,
    _apply_single_qubit_gate,
    _ry,
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
    load_dataset,
)

PAPER_TABLE8_POINTER = "https://arxiv.org/html/2604.24597v1#S4.T8"

TABLE8_MODELS = (
    "medsiglip-448",
    "rad-dino",
    "vit-patch32-cls",
)

TABLE8_CIRCUITS = (
    "1-DOF",
    "3-DOF",
)

MODEL_DISPLAY = {
    "medsiglip-448": "MedSigLIP",
    "rad-dino": "RAD-DINO",
    "vit-patch32-cls": "ViT-p32",
}


def _rz(
    theta: float,
) -> np.ndarray:
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
    """Return the number of PCA components required by each circuit."""
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
        x[d + 2 times q]

    One repetition applies the three rotations to every qubit, followed by
    the same CNOT ring used by the 1-DOF circuit.
    """
    features = _validate_3dof_features(
        x,
        q,
        reps,
    )

    state_dimension = 1 << q

    state = np.zeros(
        state_dimension,
        dtype=np.complex128,
    )

    state[0] = 1.0

    for _ in range(reps):
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

        for control in range(q - 1):
            state = _apply_cnot(
                state,
                control,
                control + 1,
                q,
            )

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
    """Validate a matrix of 3 times q dimensional inputs."""
    samples = np.asarray(
        data,
        dtype=np.float64,
    )

    if samples.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")

    if samples.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")

    expected_dimension = 3 * q

    if samples.shape[1] != expected_dimension:
        raise ValueError(
            f"{name} has {samples.shape[1]} features, "
            "but the 3-DOF circuit requires "
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
    """Compute the fidelity kernel for the 3-DOF circuit.

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

        kernel = np.square(np.abs(overlaps))

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

    kernel = np.square(np.abs(overlaps))

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
    expected_dimension = pca_dim_for_circuit(
        q,
        circuit,
    )

    if X_train.ndim != 2:
        raise ValueError("X_train must be two-dimensional.")

    if X_test.ndim != 2:
        raise ValueError("X_test must be two-dimensional.")

    if X_train.shape[1] != expected_dimension:
        raise ValueError(
            f"X_train has {X_train.shape[1]} features, "
            f"but circuit {circuit!r} requires "
            f"{expected_dimension}."
        )

    if X_test.shape[1] != expected_dimension:
        raise ValueError(
            f"X_test has {X_test.shape[1]} features, "
            f"but circuit {circuit!r} requires "
            f"{expected_dimension}."
        )

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

    (
        training_kernel,
        test_kernel,
    ) = normalize_train_test_kernels(
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

    metrics = compute_metrics(
        y_test,
        predictions,
        decision_scores,
    )

    result: dict[str, object] = dict(metrics)

    predicted_class_0 = int(np.sum(predictions == 0))

    predicted_class_1 = int(np.sum(predictions == 1))

    result["predicted_class_0"] = predicted_class_0

    result["predicted_class_1"] = predicted_class_1

    result["collapse"] = bool(predicted_class_1 == 0)

    result["zero_f1"] = bool(
        np.isclose(
            float(metrics["f1"]),
            0.0,
        )
    )

    return result


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

    maximum_required_dimension = 3 * q

    if maximum_required_dimension > X.shape[1]:
        raise ValueError(
            "The 3-DOF circuit requires "
            f"{maximum_required_dimension} raw features for q={q}, "
            f"but model {model!r} provides {X.shape[1]}."
        )

    (
        training_indices,
        validation_indices,
        test_indices,
    ) = split_indices(
        y,
        seed=seed,
    )

    maximum_pca_dimension = min(
        len(training_indices),
        X.shape[1],
    )

    if maximum_required_dimension > maximum_pca_dimension:
        raise ValueError(
            "The 3-DOF circuit requires PCA dimension "
            f"{maximum_required_dimension}, but the maximum supported "
            f"dimension is {maximum_pca_dimension} for model {model!r}."
        )

    y_train = y[training_indices]

    y_validation = y[validation_indices]

    y_test = y[test_indices]

    rows: list[dict[str, object]] = []

    for circuit in TABLE8_CIRCUITS:
        pca_dimension = pca_dim_for_circuit(
            q,
            circuit,
        )

        (
            X_train,
            X_validation,
            X_test,
            explained_variance_ratio,
        ) = preprocess(
            X[training_indices],
            X[validation_indices],
            X[test_indices],
            pca_dimension,
        )

        if X_validation.shape[1] != pca_dimension:
            raise RuntimeError("The processed validation dimension is inconsistent.")

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
            kernel_normalization=(kernel_normalization),
        )

        rows.append(
            {
                "source": source,
                "synthetic_surrogate": (source != "real"),
                "model": model,
                "model_display": (MODEL_DISPLAY[model]),
                "circuit": circuit,
                "q": q,
                "pca_dim": (pca_dimension),
                "reps": reps,
                "C": c,
                "seed": seed,
                "kernel_normalization": (kernel_normalization),
                "raw_feature_dimension": int(X.shape[1]),
                "train_samples": int(len(y_train)),
                "val_samples": int(len(y_validation)),
                "test_samples": int(len(y_test)),
                "train_class_0": int(np.sum(y_train == 0)),
                "train_class_1": int(np.sum(y_train == 1)),
                "val_class_0": int(np.sum(y_validation == 0)),
                "val_class_1": int(np.sum(y_validation == 1)),
                "test_class_0": int(np.sum(y_test == 0)),
                "test_class_1": int(np.sum(y_test == 1)),
                "pca_variance_percent": float(100.0 * explained_variance_ratio),
                "accuracy": float(metrics["accuracy"]),
                "auc": float(metrics["auc"]),
                "f1": float(metrics["f1"]),
                "predicted_class_0": int(metrics["predicted_class_0"]),
                "predicted_class_1": int(metrics["predicted_class_1"]),
                "collapse": bool(metrics["collapse"]),
                "zero_f1": bool(metrics["zero_f1"]),
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
