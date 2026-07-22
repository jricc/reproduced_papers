"""SVM preprocessing, classifiers, and metrics for the QSVM reproduction.

Preprocessing follows the paper's pipeline:

    StandardScaler
    -> PCA(q)
    -> MinMaxScaler[-1, 1]

All preprocessing transformations are fitted on the training set only.

Implemented paper comparisons
-----------------------------
Tier 1:
    qsvm
        Precomputed BSP fidelity-kernel SVM with C=1.

    linear_c1
        Linear SVM with C=1.

Tier 2:
    qsvm
        Precomputed BSP fidelity-kernel SVM with C=1.

    rbf_tuned_c
        RBF SVM with gamma="scale" and C selected on the validation set from
        {0.01, 0.1, 1, 10, 100}.

Additional diagnostics
----------------------
    rbf_c1
        RBF SVM with C=1 and gamma="scale".

    rbf_rank_matched
        RBF SVM with C=1 and gamma selected so that its training-kernel
        effective rank approximately matches the quantum-kernel effective rank.

    linear_balanced
        Linear SVM with C=1 and class weights inversely proportional to class
        frequencies.

    linear_tuned
        Class-weighted linear SVM with C selected on the validation set.
        This is an additional diagnostic, not a baseline from the paper.

Metrics
-------
The primary metric is F1 for label 1, the minority class.

ROC-AUC is also reported as a threshold-independent measure of score ranking.
Similar ROC-AUC with different F1 suggests that threshold placement, margin
scaling, or class imbalance may contribute to the difference. It does not prove
that the F1 difference is only a threshold artifact.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from .quantum_kernel import effective_rank, fidelity_kernel

C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def split_indices(y, seed):
    """Create stratified 80/10/10 train, validation, and test indices.

    Stratification preserves approximately the same minority-class proportion
    in the three subsets.
    """
    indices = np.arange(len(y))

    train_indices, temporary_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    validation_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=0.5,
        random_state=seed,
        stratify=y[temporary_indices],
    )

    return train_indices, validation_indices, test_indices


def preprocess(X_train, X_validation, X_test, q):
    """Apply the paper's preprocessing pipeline without data leakage.

    StandardScaler standardizes every original embedding dimension.

    PCA retains q components and converts the input into the q-dimensional
    representation used by both classical and quantum models.

    MinMaxScaler maps each retained PCA component to [-1, 1], the input range
    expected by the quantum feature map.

    All transformations are fitted on the training set. Validation and test
    data are transformed with the parameters learned from training data.

    Returns
    -------
    X_train_processed, X_validation_processed, X_test_processed
        Preprocessed feature matrices.

    explained_variance
        Fraction of the original standardized variance retained by PCA.
    """
    if q <= 0:
        raise ValueError("q must be positive.")

    maximum_components = min(X_train.shape[0], X_train.shape[1])
    if q > maximum_components:
        raise ValueError(
            f"q={q} exceeds the maximum PCA dimension {maximum_components}."
        )

    standard_scaler = StandardScaler().fit(X_train)

    X_train_scaled = standard_scaler.transform(X_train)
    X_validation_scaled = standard_scaler.transform(X_validation)
    X_test_scaled = standard_scaler.transform(X_test)

    pca = PCA(n_components=q).fit(X_train_scaled)

    X_train_pca = pca.transform(X_train_scaled)
    X_validation_pca = pca.transform(X_validation_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    minmax_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train_pca)

    X_train_processed = np.clip(
        minmax_scaler.transform(X_train_pca),
        -1.0,
        1.0,
    )
    X_validation_processed = np.clip(
        minmax_scaler.transform(X_validation_pca),
        -1.0,
        1.0,
    )
    X_test_processed = np.clip(
        minmax_scaler.transform(X_test_pca),
        -1.0,
        1.0,
    )

    explained_variance = float(pca.explained_variance_ratio_.sum())

    return (
        X_train_processed,
        X_validation_processed,
        X_test_processed,
        explained_variance,
    )


def normalize_train_test_kernels(K_train, K_test, method="trace"):
    """Normalize train and test kernels with one scale learned from training.

    Parameters
    ----------
    K_train
        Square training Gram matrix.

    K_test
        Rectangular test-versus-training Gram matrix.

    method
        ``"trace"`` divides both matrices by the training-kernel trace, as
        described in the paper.

        ``"frobenius"`` divides both matrices by the training-kernel
        Frobenius norm.

        ``"none"`` leaves both matrices unchanged.

    The same scale must be used for training and test kernels because the SVM
    expects both matrices to represent the same kernel function.
    """
    if method == "none":
        return K_train, K_test

    if method == "trace":
        scale = float(np.trace(K_train))
    elif method == "frobenius":
        scale = float(np.linalg.norm(K_train, ord="fro"))
    else:
        raise ValueError(
            "kernel normalization must be 'none', 'trace', or 'frobenius'."
        )

    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Kernel normalization scale must be finite and positive.")

    return K_train / scale, K_test / scale


def _scores(y_true, y_pred, decision_scores):
    """Compute class-aware test metrics.

    Label 1 is treated as the positive minority class.

    ``collapse`` follows the paper's majority-only interpretation: a run
    collapses when the model predicts no samples from class 1.

    ``zero_f1`` is reported separately because F1 can also be zero when a model
    predicts class 1 but all positive predictions are incorrect.
    """
    confusion = confusion_matrix(y_true, y_pred, labels=[0, 1])

    predicted_minority_count = int(np.sum(y_pred == 1))
    true_minority_count = int(np.sum(y_true == 1))

    minority_f1 = float(
        f1_score(
            y_true,
            y_pred,
            pos_label=1,
            zero_division=0,
        )
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1": minority_f1,
        "majority_acc": float(max(np.mean(y_true), 1.0 - np.mean(y_true))),
        "confusion_matrix": confusion.tolist(),
        "predicted_minority_count": predicted_minority_count,
        "true_minority_count": true_minority_count,
        "collapse": bool(predicted_minority_count == 0),
        "zero_f1": bool(minority_f1 == 0.0),
    }

    try:
        metrics["auc"] = float(roc_auc_score(y_true, decision_scores))
    except ValueError:
        # ROC-AUC is undefined if the test subset contains only one class.
        metrics["auc"] = float("nan")

    return metrics


def _fit_score_kernel(
    K_train,
    y_train,
    K_test,
    y_test,
    C,
    class_weight=None,
    seed=0,
):
    """Fit and evaluate an SVM using precomputed kernel matrices."""
    classifier = SVC(
        kernel="precomputed",
        C=C,
        random_state=seed,
        class_weight=class_weight,
    )
    classifier.fit(K_train, y_train)

    predictions = classifier.predict(K_test)
    decision_scores = classifier.decision_function(K_test)

    return _scores(
        y_test,
        predictions,
        decision_scores,
    )


def _fit_score_vectors(
    kernel,
    X_train,
    y_train,
    X_test,
    y_test,
    C,
    class_weight=None,
    gamma="scale",
    seed=0,
):
    """Fit and evaluate a classical SVM directly on feature vectors."""
    classifier = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        random_state=seed,
        class_weight=class_weight,
    )
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    decision_scores = classifier.decision_function(X_test)

    return _scores(
        y_test,
        predictions,
        decision_scores,
    )


def _select_c(
    kernel,
    X_train,
    y_train,
    X_validation,
    y_validation,
    class_weight=None,
    gamma="scale",
    seed=0,
):
    """Select C on validation minority-class F1.

    In case of equal validation F1, the first value in ``C_GRID`` is retained.
    Because ``C_GRID`` is ordered from smallest to largest, ties favor the
    stronger-regularized model.
    """
    best_c = C_GRID[0]
    best_validation_f1 = -1.0

    for candidate_c in C_GRID:
        classifier = SVC(
            kernel=kernel,
            C=candidate_c,
            gamma=gamma,
            class_weight=class_weight,
            random_state=seed,
        )
        classifier.fit(X_train, y_train)

        validation_predictions = classifier.predict(X_validation)
        validation_f1 = float(
            f1_score(
                y_validation,
                validation_predictions,
                pos_label=1,
                zero_division=0,
            )
        )

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1
            best_c = candidate_c

    return best_c, best_validation_f1


def _find_gamma_for_rank(
    X_train,
    target_rank,
    tolerance=0.05,
    max_iterations=50,
):
    """Find an RBF gamma with effective rank close to ``target_rank``.

    The search is performed on a logarithmic gamma scale.

    This implements the paper's separate rank-matched RBF diagnostic. It is
    not the primary Tier-2 comparison, which keeps gamma="scale" and tunes C.
    """
    if not np.isfinite(target_rank) or target_rank <= 0.0:
        raise ValueError("target_rank must be finite and positive.")

    lower_gamma = 1e-6
    upper_gamma = 1e3

    for _ in range(max_iterations):
        candidate_gamma = np.sqrt(lower_gamma * upper_gamma)
        candidate_kernel = rbf_kernel(
            X_train,
            gamma=candidate_gamma,
        )
        candidate_rank = effective_rank(candidate_kernel)

        relative_error = abs(candidate_rank - target_rank) / target_rank
        if relative_error < tolerance:
            return candidate_gamma

        # For an RBF kernel, effective rank generally increases with gamma.
        if candidate_rank < target_rank:
            lower_gamma = candidate_gamma
        else:
            upper_gamma = candidate_gamma

    return np.sqrt(lower_gamma * upper_gamma)


def run_one(
    X,
    y,
    q,
    seed,
    circuit="bsp",
    reps=1,
    C=1.0,
    classifiers=None,
    kernel_normalization="trace",
):
    """Run classifiers for one PCA dimension and one dataset seed.

    Parameters
    ----------
    X, y
        Embeddings and binary labels.

    q
        Number of PCA components and quantum input dimensions.

    seed
        Seed used for the train, validation, and test split.

    circuit, reps
        Quantum feature-map configuration.

    C
        QSVM regularization parameter. The paper uses C=1.

    classifiers
        Names of the classifiers to run.

    kernel_normalization
        Scaling applied consistently to train and test fidelity kernels.
        The manuscript protocol uses ``"trace"``.

    Returns
    -------
    list of dict
        One result dictionary per classifier.
    """
    if classifiers is None:
        classifiers = [
            "qsvm",
            "linear_c1",
            "rbf_tuned_c",
            "rbf_c1",
            "rbf_rank_matched",
            "linear_balanced",
            "linear_tuned",
        ]

    train_indices, validation_indices, test_indices = split_indices(y, seed)

    (
        X_train,
        X_validation,
        X_test,
        explained_variance,
    ) = preprocess(
        X[train_indices],
        X[validation_indices],
        X[test_indices],
        q,
    )

    y_train = y[train_indices]
    y_validation = y[validation_indices]
    y_test = y[test_indices]

    results = []
    quantum_rank = None

    quantum_rank_required = "qsvm" in classifiers or "rbf_rank_matched" in classifiers

    if quantum_rank_required:
        quantum_kernel_train_raw = fidelity_kernel(
            X_train,
            circuit=circuit,
            reps=reps,
        )
        quantum_kernel_test_raw = fidelity_kernel(
            X_test,
            X_train,
            circuit=circuit,
            reps=reps,
        )

        # A global positive scale does not change the mathematical effective
        # rank, so it is computed from the raw training kernel.
        quantum_rank = effective_rank(quantum_kernel_train_raw)

        quantum_kernel_train, quantum_kernel_test = normalize_train_test_kernels(
            quantum_kernel_train_raw,
            quantum_kernel_test_raw,
            method=kernel_normalization,
        )

    if "qsvm" in classifiers:
        metrics = _fit_score_kernel(
            quantum_kernel_train,
            y_train,
            quantum_kernel_test,
            y_test,
            C=C,
            seed=seed,
        )
        metrics["eff_rank"] = quantum_rank
        metrics["kernel_normalization"] = kernel_normalization
        results.append(("qsvm", metrics))

    if "qsvm_photonic" in classifiers:
        try:
            from .photonic_kernel import photonic_fidelity_kernels
        except (ImportError, ModuleNotFoundError) as exc:
            metrics = {
                "accuracy": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "majority_acc": float("nan"),
                "auc": float("nan"),
                "eff_rank": float("nan"),
                "confusion_matrix": [[0, 0], [0, 0]],
                "predicted_minority_count": 0,
                "true_minority_count": int(np.sum(y_test == 1)),
                "collapse": False,
                "zero_f1": False,
                "skipped": True,
                "skip_reason": f"Photonic dependencies unavailable: {exc}",
                "kernel_normalization": kernel_normalization,
            }
        else:
            photonic_kernel_train_raw, photonic_kernel_test_raw = (
                photonic_fidelity_kernels(
                    X_train,
                    X_test,
                    q,
                    n_photons=2,
                    seed=seed,
                )
            )

            photonic_rank = effective_rank(photonic_kernel_train_raw)

            photonic_kernel_train, photonic_kernel_test = normalize_train_test_kernels(
                photonic_kernel_train_raw,
                photonic_kernel_test_raw,
                method=kernel_normalization,
            )

            metrics = _fit_score_kernel(
                photonic_kernel_train,
                y_train,
                photonic_kernel_test,
                y_test,
                C=C,
                seed=seed,
            )
            metrics["eff_rank"] = photonic_rank
            metrics["skipped"] = False
            metrics["skip_reason"] = ""
            metrics["kernel_normalization"] = kernel_normalization

        results.append(("qsvm_photonic", metrics))

    linear_kernel_train = None
    linear_rank = None

    if any(
        name in classifiers
        for name in (
            "linear_c1",
            "linear_balanced",
            "linear_tuned",
        )
    ):
        linear_kernel_train = X_train @ X_train.T
        linear_rank = effective_rank(linear_kernel_train)

    if "linear_c1" in classifiers:
        metrics = _fit_score_vectors(
            "linear",
            X_train,
            y_train,
            X_test,
            y_test,
            C=1.0,
            seed=seed,
        )
        metrics["eff_rank"] = linear_rank
        results.append(("linear_c1", metrics))

    gamma_scale = 1.0 / (X_train.shape[1] * X_train.var())

    if "rbf_c1" in classifiers:
        metrics = _fit_score_vectors(
            "rbf",
            X_train,
            y_train,
            X_test,
            y_test,
            C=1.0,
            gamma="scale",
            seed=seed,
        )
        metrics["eff_rank"] = effective_rank(
            rbf_kernel(
                X_train,
                gamma=gamma_scale,
            )
        )
        results.append(("rbf_c1", metrics))

    if "rbf_tuned_c" in classifiers:
        best_c, validation_f1 = _select_c(
            "rbf",
            X_train,
            y_train,
            X_validation,
            y_validation,
            gamma="scale",
            seed=seed,
        )

        metrics = _fit_score_vectors(
            "rbf",
            X_train,
            y_train,
            X_test,
            y_test,
            C=best_c,
            gamma="scale",
            seed=seed,
        )
        metrics["eff_rank"] = effective_rank(
            rbf_kernel(
                X_train,
                gamma=gamma_scale,
            )
        )
        metrics["best_C"] = best_c
        metrics["validation_f1"] = validation_f1
        results.append(("rbf_tuned_c", metrics))

    if "rbf_rank_matched" in classifiers:
        if quantum_rank is None:
            raise RuntimeError(
                "The quantum effective rank is required for rbf_rank_matched."
            )

        matched_gamma = _find_gamma_for_rank(
            X_train,
            quantum_rank,
        )

        metrics = _fit_score_vectors(
            "rbf",
            X_train,
            y_train,
            X_test,
            y_test,
            C=1.0,
            gamma=matched_gamma,
            seed=seed,
        )
        metrics["eff_rank"] = effective_rank(
            rbf_kernel(
                X_train,
                gamma=matched_gamma,
            )
        )
        metrics["best_gamma"] = matched_gamma
        results.append(("rbf_rank_matched", metrics))

    if "linear_balanced" in classifiers:
        metrics = _fit_score_vectors(
            "linear",
            X_train,
            y_train,
            X_test,
            y_test,
            C=1.0,
            class_weight="balanced",
            seed=seed,
        )
        metrics["eff_rank"] = linear_rank
        results.append(("linear_balanced", metrics))

    if "linear_tuned" in classifiers:
        best_c, validation_f1 = _select_c(
            "linear",
            X_train,
            y_train,
            X_validation,
            y_validation,
            class_weight="balanced",
            seed=seed,
        )

        metrics = _fit_score_vectors(
            "linear",
            X_train,
            y_train,
            X_test,
            y_test,
            C=best_c,
            class_weight="balanced",
            seed=seed,
        )
        metrics["eff_rank"] = linear_rank
        metrics["best_C"] = best_c
        metrics["validation_f1"] = validation_f1
        results.append(("linear_tuned", metrics))

    rows = []

    for method_name, metrics in results:
        row = {
            "method": method_name,
            "q": q,
            "seed": seed,
            "pca_explained_variance": explained_variance,
            "n_train": len(y_train),
            "n_validation": len(y_validation),
            "n_test": len(y_test),
            "minority_fraction_train": float(np.mean(y_train)),
            "minority_fraction_validation": float(np.mean(y_validation)),
            "minority_fraction_test": float(np.mean(y_test)),
        }
        row.update(metrics)
        rows.append(row)

    return rows
