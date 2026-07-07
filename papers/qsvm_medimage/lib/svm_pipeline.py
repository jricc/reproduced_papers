"""Preprocessing, classifiers and metrics for the QSVM medical-embedding reproduction.

Preprocessing mirrors ``qve.process.data_prepare_cv`` and the original multiseed scripts:
    StandardScaler(fit on train) -> PCA(q, fit on train) -> MinMaxScaler[-1, 1].

Classifiers
-----------
Paper (Tier 1 / Tier 2, all *untuned* at C=1):
    * ``qsvm``            : SVC(kernel='precomputed', C=1) on the BSP fidelity kernel
    * ``linear_c1``       : SVC(kernel='linear',  C=1)        (Tier-1 classical baseline)
    * ``rbf_c1``          : SVC(kernel='rbf',     C=1)        (Tier-2 default RBF)
    * ``rbf_rank_matched``: SVC(kernel='rbf', C=1, gamma tuned so eff_rank(RBF)=eff_rank(QSVM))

Fair baselines added by this reproduction (the methodology requires testing whether the
"advantage" survives a *fairly* tuned classical model — F6 baseline-fairness check):
    * ``linear_balanced`` : SVC(kernel='linear', C=1, class_weight='balanced')
    * ``linear_tuned``    : SVC(kernel='linear', class_weight='balanced', C chosen on val by minority-F1)

The headline metric is minority-class F1 (positive class = label 1).  We also report
recall, accuracy and **AUC** — AUC is the key tell: a minority-F1 gain without an AUC gain
is a threshold/regularisation artifact, not genuine discrimination.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from .quantum_kernel import effective_rank, fidelity_kernel


def split_indices(y, seed):
    """80/10/10 stratified train/val/test split, matching the original scripts."""
    idx = np.arange(len(y))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    idx_val, idx_te = train_test_split(idx_tmp, test_size=0.5, random_state=seed,
                                       stratify=y[idx_tmp])
    return idx_tr, idx_val, idx_te


def preprocess(X_tr, X_val, X_te, q):
    """StandardScaler -> PCA(q) -> MinMax[-1,1], all fit on train only (no leakage)."""
    ss = StandardScaler().fit(X_tr)
    X_tr, X_val, X_te = ss.transform(X_tr), ss.transform(X_val), ss.transform(X_te)
    pca = PCA(n_components=q).fit(X_tr)
    X_tr, X_val, X_te = pca.transform(X_tr), pca.transform(X_val), pca.transform(X_te)
    mm = MinMaxScaler(feature_range=(-1, 1)).fit(X_tr)
    X_tr = np.clip(mm.transform(X_tr), -1.0, 1.0)
    X_val = np.clip(mm.transform(X_val), -1.0, 1.0)
    X_te = np.clip(mm.transform(X_te), -1.0, 1.0)
    return X_tr, X_val, X_te, float(pca.explained_variance_ratio_.sum())


def _scores(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    predicted_minority_count = int(np.sum(y_pred == 1))
    true_minority_count = int(np.sum(y_true == 1))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": f1,
        "majority_acc": float(max(np.mean(y_true), 1 - np.mean(y_true))),
        "confusion_matrix": cm.tolist(),
        "predicted_minority_count": predicted_minority_count,
        "true_minority_count": true_minority_count,
        "collapse": bool(f1 == 0.0 or predicted_minority_count == 0),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        out["auc"] = float("nan")
    return out


def _fit_score_kernel(K_tr, y_tr, K_te, y_te, C, class_weight=None, seed=0):
    svc = SVC(kernel="precomputed", C=C, random_state=seed, class_weight=class_weight)
    svc.fit(K_tr, y_tr)
    return _scores(y_te, svc.predict(K_te), svc.decision_function(K_te))


def _fit_score_vec(kernel, X_tr, y_tr, X_te, y_te, C, class_weight=None, gamma="scale", seed=0):
    svc = SVC(kernel=kernel, C=C, gamma=gamma, random_state=seed, class_weight=class_weight)
    svc.fit(X_tr, y_tr)
    return _scores(y_te, svc.predict(X_te), svc.decision_function(X_te))


def _find_gamma_for_rank(X_tr, target_rank, tol=0.05, max_iter=50):
    """Bisection on log-gamma so eff_rank(RBF(gamma)) matches target (Tier-2 protocol)."""
    lo, hi = 1e-6, 1e3
    for _ in range(max_iter):
        mid = np.sqrt(lo * hi)
        er = effective_rank(rbf_kernel(X_tr, gamma=mid))
        if abs(er - target_rank) / target_rank < tol:
            return mid
        if er < target_rank:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


def run_one(X, y, q, seed, circuit="bsp", reps=1, C=1.0, classifiers=None):
    """Run all requested classifiers for one (q, seed). Returns list of metric dicts."""
    if classifiers is None:
        classifiers = ["qsvm", "linear_c1", "rbf_c1", "rbf_rank_matched",
                       "linear_balanced", "linear_tuned"]
    idx_tr, idx_val, idx_te = split_indices(y, seed)
    Xtr, Xval, Xte, evr = preprocess(X[idx_tr], X[idx_val], X[idx_te], q)
    ytr, yval, yte = y[idx_tr], y[idx_val], y[idx_te]

    results = []
    quantum_rank = None

    if "qsvm" in classifiers:
        K_tr = fidelity_kernel(Xtr, circuit=circuit, reps=reps)
        K_te = fidelity_kernel(Xte, Xtr, circuit=circuit, reps=reps)
        quantum_rank = effective_rank(K_tr)
        m = _fit_score_kernel(K_tr, ytr, K_te, yte, C=C, seed=seed)
        m["eff_rank"] = quantum_rank
        results.append(("qsvm", m))

    if "qsvm_photonic" in classifiers:
        try:
            from .photonic_kernel import photonic_fidelity_kernels

            Kp_tr, Kp_te = photonic_fidelity_kernels(Xtr, Xte, q, n_photons=2, seed=seed)
            m = _fit_score_kernel(Kp_tr, ytr, Kp_te, yte, C=C, seed=seed)
            m["eff_rank"] = effective_rank(Kp_tr)
            m["skipped"] = False
            m["skip_reason"] = ""
        except Exception as exc:
            m = {
                "accuracy": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "majority_acc": float("nan"),
                "auc": float("nan"),
                "eff_rank": float("nan"),
                "confusion_matrix": [[0, 0], [0, 0]],
                "predicted_minority_count": 0,
                "true_minority_count": int(np.sum(yte == 1)),
                "collapse": True,
                "skipped": True,
                "skip_reason": f"MerLin/Perceval unavailable: {exc}",
            }
        results.append(("qsvm_photonic", m))

    if "linear_c1" in classifiers:
        m = _fit_score_vec("linear", Xtr, ytr, Xte, yte, C=1.0, seed=seed)
        m["eff_rank"] = effective_rank(Xtr @ Xtr.T)
        results.append(("linear_c1", m))

    if "rbf_c1" in classifiers:
        gamma_scale = 1.0 / (Xtr.shape[1] * Xtr.var())
        m = _fit_score_vec("rbf", Xtr, ytr, Xte, yte, C=1.0, gamma="scale", seed=seed)
        m["eff_rank"] = effective_rank(rbf_kernel(Xtr, gamma=gamma_scale))
        results.append(("rbf_c1", m))

    if "rbf_rank_matched" in classifiers and quantum_rank is not None:
        gamma_star = _find_gamma_for_rank(Xtr, quantum_rank)
        m = _fit_score_vec("rbf", Xtr, ytr, Xte, yte, C=1.0, gamma=gamma_star, seed=seed)
        m["eff_rank"] = effective_rank(rbf_kernel(Xtr, gamma=gamma_star))
        results.append(("rbf_rank_matched", m))

    if "linear_balanced" in classifiers:
        m = _fit_score_vec("linear", Xtr, ytr, Xte, yte, C=1.0,
                           class_weight="balanced", seed=seed)
        m["eff_rank"] = effective_rank(Xtr @ Xtr.T)
        results.append(("linear_balanced", m))

    if "linear_tuned" in classifiers:
        best_c, best_f1 = 1.0, -1.0
        for c in (0.01, 0.1, 1.0, 10.0, 100.0):
            svc = SVC(kernel="linear", C=c, class_weight="balanced", random_state=seed)
            svc.fit(Xtr, ytr)
            f1v = f1_score(yval, svc.predict(Xval), zero_division=0)
            if f1v > best_f1:
                best_f1, best_c = f1v, c
        m = _fit_score_vec("linear", Xtr, ytr, Xte, yte, C=best_c,
                           class_weight="balanced", seed=seed)
        m["eff_rank"] = effective_rank(Xtr @ Xtr.T)
        m["best_C"] = best_c
        results.append(("linear_tuned", m))

    rows = []
    for name, m in results:
        row = {"method": name, "q": q, "seed": seed,
               "pca_explained_variance": evr,
               "n_train": len(ytr), "n_test": len(yte),
               "minority_fraction_test": float(np.mean(yte))}
        row.update(m)
        rows.append(row)
    return rows
