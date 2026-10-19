"""Run the branches of an MRMR model independently, as separate processes, nodes or QPUs would.

In MRMR, branch i is sub-reservoir i followed by its own ridge readout, and the
forecast is the mean of the branch forecasts (Fig. 1, Table I). A branch needs only
the input window and its own seeds, and only its forecast leaves it, so
:func:`branch_forecast` is a self-contained unit of work. Its result equals, up to floating-point rounding,
the corresponding part of :func:`lib.runner.train_and_evaluate`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .architectures import CellSpec
from .data import ForecastData
from .noise import devices_for_components
from .readouts import fit_linear_ridge
from .runner import SPLITS, sub_reservoir_features


def branch_forecast(
    cfg: dict[str, Any],
    data: ForecastData,
    spec: CellSpec,
    cell_index: int,
    branch: int,
    reservoir_type: str = "quantum_ideal",
) -> dict[str, np.ndarray]:
    """Forecast of one MRMR branch: its sub-reservoir and its own linear ridge readout.

    Parameters
    ----------
    cfg : dict
        Resolved configuration (seed, reservoir, readout and noise settings).
    data : ForecastData
        Windowed, normalized splits.
    spec : CellSpec
        An MRMR configuration (one readout per sub-reservoir).
    cell_index : int
        Index of the configuration in the run, part of the reservoir seeds.
    branch : int
        Index of the sub-reservoir (and of its readout).
    reservoir_type : str
        Reservoir type of :data:`lib.runner.RESERVOIR_TYPES`. Default value is
        ``"quantum_ideal"``.

    Returns
    -------
    dict
        Predictions of the branch for every split.
    """

    if spec.architecture != "mrmr":
        raise ValueError(
            f"branch_forecast needs an MRMR configuration, got {spec.architecture}"
        )
    devices = devices_for_components(len(spec.reservoir_sizes), cfg["noise"]["mapping"])
    outputs = sub_reservoir_features(
        cfg,
        data,
        spec.reservoir_sizes[branch],
        cell_index,
        branch,
        reservoir_type,
        devices[branch],
    )
    features = dict(zip(SPLITS, outputs))
    rcfg = cfg["readout"]
    fitted = fit_linear_ridge(
        features,
        data.targets,
        rcfg["lambda_grid"],
        intercept=bool(rcfg["linear_intercept"]),
    )
    return fitted.predictions
