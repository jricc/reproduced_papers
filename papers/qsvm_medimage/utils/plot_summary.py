#!/usr/bin/env python3
"""Plot aggregated metrics from one experiment run.

The run directory must contain ``summary.csv``, produced by ``lib.runner``.

The script generates:

- ``f1_vs_q.png``:
  minority-class F1 versus PCA dimension q;

- ``auc_vs_q.png``:
  test ROC-AUC versus PCA dimension q;

- ``effective_rank_vs_q.png``:
  kernel effective rank versus PCA dimension q;

- ``collapse_rate_vs_q.png``:
  majority-only collapse rate versus PCA dimension q;

- ``f1_auc_bar_q<Q>.png``:
  minority-class F1 and ROC-AUC at one selected q.

Figures are saved in the run directory. They can also be copied to a separate
results directory.

Interpretation
--------------
Minority-class F1 evaluates label 1 at the classifier's decision threshold.

ROC-AUC evaluates score ranking across thresholds. Similar ROC-AUC with
different F1 suggests that threshold placement, margin scaling, or class
imbalance may contribute to the difference. It does not prove that the F1
difference is only a threshold artifact.

Effective rank describes the kernel eigenspectrum. A larger effective rank
does not by itself imply better classification.

The class-weighted and tuned linear models are additional diagnostic
baselines. They are not part of the paper's primary Tier-1 comparison.

Usage
-----
    python utils/plot_summary.py \
        --run-dir outdir/run_XXXX \
        --highlight-q 11

Optional result copy:

    python utils/plot_summary.py \
        --run-dir outdir/run_XXXX \
        --highlight-q 11 \
        --results-dir results
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_LABELS = {
    "qsvm": "QSVM, C=1",
    "qsvm_photonic": "Photonic QSVM, C=1",
    "linear_c1": "Linear SVM, C=1",
    "rbf_c1": "RBF SVM, C=1",
    "rbf_tuned_c": "RBF SVM, validation-selected C",
    "rbf_rank_matched": "Rank-matched RBF, C=1",
    "linear_balanced": "Class-weighted linear SVM, C=1",
    "linear_tuned": "Class-weighted linear SVM, selected C",
}

METHOD_ORDER = (
    "linear_c1",
    "qsvm",
    "qsvm_photonic",
    "rbf_c1",
    "rbf_tuned_c",
    "rbf_rank_matched",
    "linear_balanced",
    "linear_tuned",
)

METHOD_COLORS = {
    "linear_c1": "#4C78A8",
    "qsvm": "#F58518",
    "qsvm_photonic": "#B279A2",
    "rbf_c1": "#54A24B",
    "rbf_tuned_c": "#2E8B57",
    "rbf_rank_matched": "#E45756",
    "linear_balanced": "#72B7B2",
    "linear_tuned": "#FF9DA6",
}

REQUIRED_COLUMNS = {
    "method",
    "q",
    "n_seeds",
}

METRIC_COLUMNS = {
    "f1": (
        "f1_mean",
        "f1_std",
    ),
    "auc": (
        "auc_mean",
        "auc_std",
    ),
    "effective_rank": (
        "eff_rank_mean",
        "eff_rank_std",
    ),
    "collapse_rate": (
        "collapse_mean",
        "collapse_std",
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing summary.csv.",
    )

    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help=(
            "Optional summary CSV path. By default, "
            "<run-dir>/summary.csv is used."
        ),
    )

    parser.add_argument(
        "--highlight-q",
        type=int,
        default=11,
        help="PCA dimension used for the paired F1 and AUC bar chart.",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Optional second directory receiving copies of the figures. "
            "If omitted, figures are saved only in the run directory."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=130,
    )

    parser.add_argument(
        "--title-prefix",
        default="Synthetic benchmark",
    )

    return parser.parse_args()


def validate_arguments(
    args: argparse.Namespace,
) -> None:
    """Validate command-line arguments."""
    if not args.run_dir.is_dir():
        raise FileNotFoundError(
            f"Run directory not found: {args.run_dir}"
        )

    if args.highlight_q <= 0:
        raise ValueError(
            "--highlight-q must be positive."
        )

    if args.dpi <= 0:
        raise ValueError(
            "--dpi must be positive."
        )


def summary_csv_path(
    args: argparse.Namespace,
) -> Path:
    """Return the explicit or default summary CSV path."""
    if args.summary_csv is not None:
        return args.summary_csv

    return args.run_dir / "summary.csv"


def load_summary(
    path: Path,
) -> pd.DataFrame:
    """Load and validate the aggregated experiment table."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Summary CSV not found: {path}"
        )

    dataframe = pd.read_csv(
        path
    )

    if dataframe.empty:
        raise ValueError(
            "Summary CSV contains no rows."
        )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Summary CSV is missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    dataframe["method"] = (
        dataframe["method"]
        .astype(str)
    )

    dataframe["q"] = pd.to_numeric(
        dataframe["q"],
        errors="coerce",
    )

    dataframe["n_seeds"] = pd.to_numeric(
        dataframe["n_seeds"],
        errors="coerce",
    )

    if dataframe[
        [
            "q",
            "n_seeds",
        ]
    ].isna().any().any():
        raise ValueError(
            "Summary CSV contains invalid q or n_seeds values."
        )

    if not np.all(
        np.equal(
            np.mod(
                dataframe["q"],
                1,
            ),
            0,
        )
    ):
        raise ValueError(
            "All q values must be integers."
        )

    dataframe["q"] = dataframe[
        "q"
    ].astype(int)

    duplicate_rows = dataframe.duplicated(
        subset=[
            "method",
            "q",
        ],
        keep=False,
    )

    if duplicate_rows.any():
        duplicates = dataframe.loc[
            duplicate_rows,
            [
                "method",
                "q",
            ],
        ]

        raise ValueError(
            "Summary CSV contains duplicate method-q rows:\n"
            + duplicates.to_string(
                index=False
            )
        )

    numeric_metric_columns = {
        column
        for pair in METRIC_COLUMNS.values()
        for column in pair
        if column in dataframe.columns
    }

    for column in numeric_metric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    return dataframe


def available_methods(
    dataframe: pd.DataFrame,
) -> list:
    """Return known methods present in the preferred display order."""
    present_methods = set(
        dataframe["method"]
    )

    ordered_methods = [
        method
        for method in METHOD_ORDER
        if method in present_methods
    ]

    additional_methods = sorted(
        present_methods
        - set(METHOD_ORDER)
    )

    return (
        ordered_methods
        + additional_methods
    )


def method_label(
    method: str,
) -> str:
    """Return a human-readable method label."""
    return METHOD_LABELS.get(
        method,
        method,
    )


def method_color(
    method: str,
) -> str | None:
    """Return the configured method color when available."""
    return METHOD_COLORS.get(
        method
    )


def error_values(
    rows: pd.DataFrame,
    column: str,
) -> np.ndarray | None:
    """Return non-negative finite error values or None."""
    if column not in rows.columns:
        return None

    values = rows[
        column
    ].to_numpy(
        dtype=np.float64
    )

    if np.all(
        ~np.isfinite(values)
    ):
        return None

    return np.where(
        np.isfinite(values),
        np.maximum(
            values,
            0.0,
        ),
        0.0,
    )


def save_figure(
    *,
    figure: plt.Figure,
    filename: str,
    run_dir: Path,
    results_dir: Path | None,
    dpi: int,
) -> Path:
    """Save one figure and optionally copy it to a result directory."""
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_path = (
        run_dir
        / filename
    )

    figure.savefig(
        run_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    if results_dir is not None:
        results_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        results_path = (
            results_dir
            / filename
        )

        if (
            run_path.resolve()
            != results_path.resolve()
        ):
            shutil.copy2(
                run_path,
                results_path,
            )

    return run_path


def plot_metric_lines(
    *,
    dataframe: pd.DataFrame,
    methods: list[str],
    mean_column: str,
    std_column: str,
    ylabel: str,
    title: str,
    filename: str,
    run_dir: Path,
    results_dir: Path | None,
    dpi: int,
    horizontal_reference: float | None = None,
    horizontal_label: str | None = None,
    y_limits: tuple[float, float] | None = None,
) -> Path | None:
    """Plot one aggregated metric against q for every available method."""
    if mean_column not in dataframe.columns:
        print(
            f"Skipped {filename}: "
            f"missing column {mean_column}."
        )
        return None

    figure, axis = plt.subplots(
        figsize=(8.2, 4.8)
    )

    plotted_methods = 0

    for method in methods:
        rows = (
            dataframe[
                dataframe["method"] == method
            ]
            .sort_values("q")
        )

        if rows.empty:
            continue

        finite_rows = rows[
            np.isfinite(
                rows[mean_column]
            )
        ]

        if finite_rows.empty:
            continue

        axis.errorbar(
            finite_rows["q"],
            finite_rows[mean_column],
            yerr=error_values(
                finite_rows,
                std_column,
            ),
            marker="o",
            markersize=5,
            capsize=3,
            linewidth=1.5,
            color=method_color(method),
            label=method_label(method),
        )

        plotted_methods += 1

    if plotted_methods == 0:
        plt.close(
            figure
        )

        print(
            f"Skipped {filename}: no finite values."
        )

        return None

    if horizontal_reference is not None:
        axis.axhline(
            horizontal_reference,
            linestyle="--",
            color="gray",
            linewidth=1.0,
            label=horizontal_label,
        )

    q_values = sorted(
        dataframe["q"].unique()
    )

    axis.set_xticks(
        q_values
    )

    axis.set_xlabel(
        "PCA dimension q"
    )

    axis.set_ylabel(
        ylabel
    )

    axis.set_title(
        title
    )

    if y_limits is not None:
        axis.set_ylim(
            *y_limits
        )

    axis.grid(
        alpha=0.3,
    )

    axis.legend(
        fontsize=8,
        ncol=2,
    )

    figure.tight_layout()

    return save_figure(
        figure=figure,
        filename=filename,
        run_dir=run_dir,
        results_dir=results_dir,
        dpi=dpi,
    )


def plot_f1_auc_bars(
    *,
    dataframe: pd.DataFrame,
    methods: list[str],
    highlight_q: int,
    run_dir: Path,
    results_dir: Path | None,
    dpi: int,
    title_prefix: str,
) -> Path | None:
    """Plot paired minority-F1 and ROC-AUC bars at one q value."""
    required_columns = {
        "f1_mean",
        "auc_mean",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        print(
            "Skipped F1/AUC bar chart: missing columns "
            + ", ".join(
                sorted(missing_columns)
            )
        )
        return None

    q_rows = dataframe[
        dataframe["q"] == highlight_q
    ].copy()

    if q_rows.empty:
        available_q = sorted(
            dataframe["q"].unique()
        )

        print(
            "Skipped F1/AUC bar chart: "
            f"q={highlight_q} is absent. "
            f"Available q values: {available_q}."
        )

        return None

    rows_by_method = {
        row["method"]: row
        for _, row in q_rows.iterrows()
    }

    selected_methods = [
        method
        for method in methods
        if method in rows_by_method
    ]

    if not selected_methods:
        print(
            "Skipped F1/AUC bar chart: no methods available."
        )
        return None

    f1_values = np.asarray(
        [
            rows_by_method[
                method
            ]["f1_mean"]
            for method in selected_methods
        ],
        dtype=np.float64,
    )

    auc_values = np.asarray(
        [
            rows_by_method[
                method
            ]["auc_mean"]
            for method in selected_methods
        ],
        dtype=np.float64,
    )

    f1_errors = np.asarray(
        [
            rows_by_method[
                method
            ].get(
                "f1_std",
                0.0,
            )
            for method in selected_methods
        ],
        dtype=np.float64,
    )

    auc_errors = np.asarray(
        [
            rows_by_method[
                method
            ].get(
                "auc_std",
                0.0,
            )
            for method in selected_methods
        ],
        dtype=np.float64,
    )

    f1_errors = np.where(
        np.isfinite(f1_errors),
        np.maximum(
            f1_errors,
            0.0,
        ),
        0.0,
    )

    auc_errors = np.where(
        np.isfinite(auc_errors),
        np.maximum(
            auc_errors,
            0.0,
        ),
        0.0,
    )

    positions = np.arange(
        len(selected_methods)
    )

    width = 0.38

    figure, axis = plt.subplots(
        figsize=(max(8.5, 1.4 * len(selected_methods)), 5.0)
    )

    axis.bar(
        positions - width / 2.0,
        f1_values,
        width,
        yerr=f1_errors,
        capsize=3,
        label="Minority-class F1",
        color="#4C78A8",
    )

    axis.bar(
        positions + width / 2.0,
        auc_values,
        width,
        yerr=auc_errors,
        capsize=3,
        label="Test ROC-AUC",
        color="#F58518",
    )

    axis.axhline(
        0.5,
        linestyle="--",
        color="gray",
        linewidth=1.0,
        label="Random-ranking AUC reference",
    )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        [
            method_label(method)
            for method in selected_methods
        ],
        rotation=25,
        ha="right",
    )

    axis.set_ylabel(
        "Score"
    )

    axis.set_ylim(
        0.0,
        1.05,
    )

    axis.set_title(
        f"{title_prefix}: F1 and ROC-AUC at q={highlight_q}"
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    axis.legend(
        fontsize=8,
    )

    figure.tight_layout()

    filename = (
        f"f1_auc_bar_q{highlight_q}.png"
    )

    return save_figure(
        figure=figure,
        filename=filename,
        run_dir=run_dir,
        results_dir=results_dir,
        dpi=dpi,
    )


def main() -> None:
    """Load the run summary and generate all available figures."""
    args = parse_args()

    validate_arguments(
        args
    )

    csv_path = summary_csv_path(
        args
    )

    dataframe = load_summary(
        csv_path
    )

    methods = available_methods(
        dataframe
    )

    generated_paths: list[Path] = []

    f1_path = plot_metric_lines(
        dataframe=dataframe,
        methods=methods,
        mean_column="f1_mean",
        std_column="f1_std",
        ylabel="Minority-class F1",
        title=(
            f"{args.title_prefix}: minority-class F1"
        ),
        filename="f1_vs_q.png",
        run_dir=args.run_dir,
        results_dir=args.results_dir,
        dpi=args.dpi,
        horizontal_reference=0.0,
        horizontal_label="Zero minority F1",
        y_limits=(
            -0.02,
            1.02,
        ),
    )

    if f1_path is not None:
        generated_paths.append(
            f1_path
        )

    auc_path = plot_metric_lines(
        dataframe=dataframe,
        methods=methods,
        mean_column="auc_mean",
        std_column="auc_std",
        ylabel="Test ROC-AUC",
        title=(
            f"{args.title_prefix}: test ROC-AUC"
        ),
        filename="auc_vs_q.png",
        run_dir=args.run_dir,
        results_dir=args.results_dir,
        dpi=args.dpi,
        horizontal_reference=0.5,
        horizontal_label="Random-ranking reference",
        y_limits=(
            0.0,
            1.02,
        ),
    )

    if auc_path is not None:
        generated_paths.append(
            auc_path
        )

    effective_rank_path = plot_metric_lines(
        dataframe=dataframe,
        methods=methods,
        mean_column="eff_rank_mean",
        std_column="eff_rank_std",
        ylabel="Kernel effective rank",
        title=(
            f"{args.title_prefix}: kernel effective rank"
        ),
        filename="effective_rank_vs_q.png",
        run_dir=args.run_dir,
        results_dir=args.results_dir,
        dpi=args.dpi,
    )

    if effective_rank_path is not None:
        generated_paths.append(
            effective_rank_path
        )

    collapse_path = plot_metric_lines(
        dataframe=dataframe,
        methods=methods,
        mean_column="collapse_mean",
        std_column="collapse_std",
        ylabel="Majority-only collapse rate",
        title=(
            f"{args.title_prefix}: majority-only collapse rate"
        ),
        filename="collapse_rate_vs_q.png",
        run_dir=args.run_dir,
        results_dir=args.results_dir,
        dpi=args.dpi,
        horizontal_reference=0.0,
        horizontal_label="No collapsed seeds",
        y_limits=(
            -0.02,
            1.02,
        ),
    )

    if collapse_path is not None:
        generated_paths.append(
            collapse_path
        )

    bar_path = plot_f1_auc_bars(
        dataframe=dataframe,
        methods=methods,
        highlight_q=args.highlight_q,
        run_dir=args.run_dir,
        results_dir=args.results_dir,
        dpi=args.dpi,
        title_prefix=args.title_prefix,
    )

    if bar_path is not None:
        generated_paths.append(
            bar_path
        )

    print(
        f"Generated {len(generated_paths)} figures:"
    )

    for path in generated_paths:
        print(
            f"- {path}"
        )


if __name__ == "__main__":
    main()
