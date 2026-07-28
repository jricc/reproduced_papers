"""Focused unit tests for preprocessing and trace-scaling protocols."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


process = load_module("qve_process_for_protocol_tests", "qve/process.py")
core = load_module("qve_core_for_protocol_tests", "qve/core.py")


TRAIN_FEATURES = np.array([[-1.0], [0.0], [1.0]])
HELD_OUT_WITHIN_TRAIN_RANGE = np.array([[0.5]])
HELD_OUT_EXTREME = np.array([[10.0]])


def test_legacy_minmax_held_out_extrema_change_transformed_training_data():
    train_with_inner_value, _ = process.data_prepare_cv(
        1,
        TRAIN_FEATURES,
        HELD_OUT_WITHIN_TRAIN_RANGE,
        fix_leakage=False,
    )
    train_with_extreme, _ = process.data_prepare_cv(
        1,
        TRAIN_FEATURES,
        HELD_OUT_EXTREME,
        fix_leakage=False,
    )

    assert not np.allclose(train_with_inner_value, train_with_extreme)


def test_train_only_minmax_is_independent_of_held_out_extrema():
    train_with_inner_value, _ = process.data_prepare_cv(
        1,
        TRAIN_FEATURES,
        HELD_OUT_WITHIN_TRAIN_RANGE,
        fix_leakage=True,
    )
    train_with_extreme, _ = process.data_prepare_cv(
        1,
        TRAIN_FEATURES,
        HELD_OUT_EXTREME,
        fix_leakage=True,
    )

    np.testing.assert_allclose(train_with_inner_value, train_with_extreme)


def test_train_only_minmax_does_not_clip_held_out_values():
    _, held_out = process.data_prepare_cv(
        1,
        TRAIN_FEATURES,
        HELD_OUT_EXTREME,
        fix_leakage=True,
    )

    assert np.any((held_out < -1.0) | (held_out > 1.0))


def test_legacy_trace_normalization_skips_rectangular_matrices():
    square = np.array([[2.0, 1.0], [1.0, 2.0]])
    rectangular = np.array([[4.0, 8.0], [12.0, 16.0], [20.0, 24.0]])

    normalized_square = core.normalize_kernel_trace(square)
    normalized_rectangular = core.normalize_kernel_trace(rectangular)

    np.testing.assert_allclose(normalized_square, square / np.trace(square))
    np.testing.assert_array_equal(normalized_rectangular, rectangular)


def test_train_and_cross_normalization_uses_training_trace_for_both():
    train = np.array([[2.0, 1.0], [1.0, 2.0]])
    cross = np.array([[4.0, 8.0], [12.0, 16.0], [20.0, 24.0]])
    train_trace = np.trace(train)

    normalized_train, normalized_cross = (
        core.normalize_train_and_cross_kernel_trace(train, cross)
    )

    np.testing.assert_allclose(normalized_train, train / train_trace)
    np.testing.assert_allclose(normalized_cross, cross / train_trace)


@pytest.mark.parametrize(
    ("train", "cross"),
    [
        (np.ones(3), np.ones((1, 3))),
        (np.ones((2, 3)), np.ones((1, 2))),
        (np.eye(2), np.ones(2)),
        (np.eye(2), np.ones((1, 3))),
    ],
)
def test_train_and_cross_normalization_rejects_invalid_dimensions(train, cross):
    with pytest.raises(ValueError):
        core.normalize_train_and_cross_kernel_trace(train, cross)


@pytest.mark.parametrize(
    "train",
    [
        np.zeros((2, 2)),
        -np.eye(2),
        np.array([[np.nan, 0.0], [0.0, 1.0]]),
        np.array([[np.inf, 0.0], [0.0, 1.0]]),
    ],
)
def test_train_and_cross_normalization_rejects_invalid_trace(train):
    with pytest.raises(ValueError):
        core.normalize_train_and_cross_kernel_trace(train, np.ones((1, 2)))
