#!/usr/bin/env python3
"""Plot classifier metrics across synthetic signal-strength settings.

The input CSV is expected to contain aggregated results from a signal sweep.
The figure contains two panels:

1. minority-class F1 versus synthetic signal strength;
2. test ROC-AUC versus synthetic signal strength.

The default comparison includes:

- the QSVM with C=1;
- the linear SVM with C=1 used in the paper's Tier-1 comparison;
- an additional class-weighted linear SVM.

Important interpretation
------------------------
The signal-strength parameter belongs to the synthetic generator. It controls
how strongly label-related latent factors are represented in the generated
embeddings. It is not a quantity measured from the inaccessible medical data.

Differences between F1 and ROC-AUC can indicate that threshold placement,
margin scaling, or class imbalance contributes to observed performance
differences. Similar ROC-AUC values do not prove that an F1 difference is only
a threshold artifact.

The class-weighted linear SVM is an additional diagnostic baseline. It is not
part of the paper's primary Tier-1 protocol.

Usage
-----
    python utils/plot_signal_sweep.py \
        --csv results/signal_sweep_q11.csv

Optional output path:

    python utils/plot_signal_sweep.py \
        --csv results/signal_sweep_q11.csv \
        --out results/signal_sweep_q11.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_LABELS = {
    "qsvm": "QSVM, C=1",
    "linear_c1": "Linear SVM, C=1",
    "linear_balanced": "Class-weighted linear SVM, C=1",
}

METHOD_ORDER = tuple(
    METHOD_LABELS
)

REQUIRED_COLUMNS = {
    "method",
    "signal",
    "q",
    "f1_mean",
    "auc_mean",
}

OPTIONAL_STD_COLUMNS = {
    "f1_std",
    "auc_std",
}


def parse_methods(
    raw: str,
) -> list:
    """Parse and validate a comma-separated list of method names."""
    methods = [
        part.strip()
        for part in raw.split(",")
        if part.strip()
    ]

    if not methods:
        raise argparse.ArgumentTypeError(
            "At least one method is required."
        )

    unknown_methods = [
        method
        for method in methods
        if method not in METHOD_LABELS
    ]

    if unknown_methods:
        raise argparse.ArgumentTypeError(
            "Unknown methods: "
            + ", ".join(unknown_methods)
        )

    if len(methods) != len(set(methods)):
        raise argparse.ArgumentTypeError(
            "Method names must be unique."
        )

    return methods


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Aggregated signal-sweep CSV.",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output PNG path. By default, '_fig.png' is added "
            "to the CSV stem."
        ),
    )

    parser.add_argument(
        "--methods",
        type=parse_methods,
        default=list(METHOD_ORDER),
        help=(
            "Comma-separated subset of qsvm, linear_c1, "
            "and linear_balanced."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=130,
    )

    parser.add_argument(
        "--title-prefix",
        default="Synthetic signal sweep",
    )

    return parser.parse_args()


def validate_input_table(
    dataframe: pd.DataFrame,
) -> int:
    """Validate the input schema and return the unique PCA dimension q."""
    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Input CSV is missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    if dataframe.empty:
        raise ValueError(
            "Input CSV contains no rows."
        )

    numeric_columns = [
        "signal",
        "q",
        "f1_mean",
        "auc_mean",
    ]

    for column in OPTIONAL_STD_COLUMNS:
        if column in dataframe.columns:
            numeric_columns.append(
                column
            )

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    required_numeric_columns = [
        "signal",
        "q",
        "f1_mean",
        "auc_mean",
    ]

    if dataframe[
        required_numeric_columns
    ].isna().any().any():
        invalid_rows = dataframe[
            dataframe[
                required_numeric_columns
            ].isna().any(axis=1)
        ]

        raise ValueError(
            "Input CSV contains invalid values in required numeric "
            f"columns. Invalid row count: {len(invalid_rows)}."
        )

    q_values = np.sort(
        dataframe["q"].unique()
    )

    if len(q_values) != 1:
        raise ValueError(
            "The plot expects exactly one q value, but found: "
            + ", ".join(
                str(value)
                for value in q_values
            )
        )

    q_value = float(
        q_values[0]
    )

    if not q_value.is_integer():
        raise ValueError(
            f"q must be an integer, but found {q_value}."
        )

    return int(
        q_value
    )


def prepare_method_rows(
    dataframe: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    """Return one method's rows sorted by synthetic signal strength."""
    method_rows = dataframe[
        dataframe["method"] == method
    ].copy()

    if method_rows.empty:
        raise ValueError(
            f"Method {method!r} is absent from the input CSV."
        )

    duplicate_signals = method_rows[
        "signal"
    ].duplicated(
        keep=False
    )

    if duplicate_signals.any():
        duplicated_values = sorted(
            method_rows.loc[
                duplicate_signals,
                "signal",
            ].unique()
        )

        raise ValueError(
            f"Method {method!r} contains duplicate rows for "
            "signal values: "
            + ", ".join(
                str(value)
                for value in duplicated_values
            )
        )

    return method_rows.sort_values(
        "signal"
    )


def standard_deviation_values(
    rows: pd.DataFrame,
    column: str,
) -> np.ndarray | None:
    """Return finite standard deviations or None when unavailable."""
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

    # Matplotlib does not accept negative error-bar lengths.
    finite_values = np.where(
        np.isfinite(values),
        np.maximum(values, 0.0),
        0.0,
    )

    return finite_values


def output_path(
    csv_path: Path,
    requested_output: Path | None,
) -> Path:
    """Return the requested or default output path."""
    if requested_output is not None:
        return requested_output

    return csv_path.with_name(
        csv_path.stem
        + "_fig.png"
    )


def plot_signal_sweep(
    *,
    dataframe: pd.DataFrame,
    methods: list[str],
    q: int,
    output: Path,
    dpi: int,
    title_prefix: str,
) -> None:
    """Create and save the two-panel signal-sweep figure."""
    figure, (
        f1_axis,
        auc_axis,
    ) = plt.subplots(
        1,
        2,
        figsize=(12, 4.5),
        sharex=True,
    )

    for method in methods:
        rows = prepare_method_rows(
            dataframe,
            method,
        )

        signals = rows[
            "signal"
        ].to_numpy(
            dtype=np.float64
        )

        f1_means = rows[
            "f1_mean"
        ].to_numpy(
            dtype=np.float64
        )

        auc_means = rows[
            "auc_mean"
        ].to_numpy(
            dtype=np.float64
        )

        f1_stds = standard_deviation_values(
            rows,
            "f1_std",
        )

        auc_stds = standard_deviation_values(
            rows,
            "auc_std",
        )

        label = METHOD_LABELS[
            method
        ]

        f1_axis.errorbar(
            signals,
            f1_means,
            yerr=f1_stds,
            marker="o",
            capsize=3,
            linewidth=1.5,
            label=label,
        )

        auc_axis.errorbar(
            signals,
            auc_means,
            yerr=auc_stds,
            marker="o",
            capsize=3,
            linewidth=1.5,
            label=label,
        )

    f1_axis.set_title(
        f"Minority-class F1, q={q}"
    )

    f1_axis.set_xlabel(
        "Synthetic signal strength"
    )

    f1_axis.set_ylabel(
        "Minority-class F1"
    )

    f1_axis.set_ylim(
        -0.02,
        1.02,
    )

    f1_axis.grid(
        alpha=0.3,
    )

    f1_axis.legend(
        fontsize=8,
    )

    auc_axis.axhline(
        0.5,
        linestyle="--",
        color="gray",
        linewidth=1.0,
        label="Random-ranking reference",
    )

    auc_axis.set_title(
        f"Test ROC-AUC, q={q}"
    )

    auc_axis.set_xlabel(
        "Synthetic signal strength"
    )

    auc_axis.set_ylabel(
        "Test ROC-AUC"
    )

    auc_axis.set_ylim(
        -0.02,
        1.02,
    )

    auc_axis.grid(
        alpha=0.3,
    )

    auc_axis.legend(
        fontsize=8,
    )

    figure.suptitle(
        (
            f"{title_prefix}\n"
            "Controlled synthetic benchmark, not medical-data results"
        ),
        fontsize=11,
    )

    figure.tight_layout(
        rect=(
            0.0,
            0.0,
            1.0,
            0.91,
        )
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def main() -> None:
    """Load the signal-sweep results and generate the figure."""
    args = parse_args()

    if not args.csv.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {args.csv}"
        )

    if args.dpi <= 0:
        raise ValueError(
            "--dpi must be positive."
        )

    dataframe = pd.read_csv(
        args.csv
    )

    q = validate_input_table(
        dataframe
    )

    missing_methods = [
        method
        for method in args.methods
        if method
        not in set(
            dataframe["method"]
        )
    ]

    if missing_methods:
        raise ValueError(
            "Requested methods are absent from the CSV: "
            + ", ".join(
                missing_methods
            )
        )

    output = output_path(
        args.csv,
        args.out,
    )

    plot_signal_sweep(
        dataframe=dataframe,
        methods=args.methods,
        q=q,
        output=output,
        dpi=args.dpi,
        title_prefix=args.title_prefix,
    )

    print(
        f"Saved {output}"
    )


if __name__ == "__main__":
    main()
