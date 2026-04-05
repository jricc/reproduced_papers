"""Runtime helpers for reproducible HQPINN experiment execution."""

from __future__ import annotations

from copy import deepcopy
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import torch

from .dtypes import DtypeSpec, coerce_dtype_spec, dtype_torch


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "defaults.json"


def configure_logging() -> None:
    """Initialize process-wide logging once the CLI/shared runner starts."""
    level_name = os.getenv("HQPINN_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def log_run_banner(config: dict[str, Any]) -> None:
    """Log the resolved experiment identity before dispatching benchmark code."""
    LOGGER.info(
        "Starting run: experiment=%s mode=%s backend=%s",
        config.get("experiment"),
        config.get("mode"),
        config.get("backend"),
    )
    LOGGER.debug(
        "Resolved config:\n%s",
        json.dumps(config, indent=2, sort_keys=True),
    )


def seed_everything(seed: int = 0) -> None:
    """Seed Python, PyTorch, and NumPy so repeated runs stay comparable."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(seed)


def require_file(path: Path, *, label: str) -> Path:
    """Return a required runtime path or raise a clear error."""
    if path.is_file():
        return path
    raise FileNotFoundError(f"Missing {label}: {path}")


def _is_dtype_key(key: str) -> bool:
    """Recognize config keys that should be interpreted as dtype selectors."""
    return key == "dtype" or key.endswith("_dtype")


def normalize_dtype_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Deep-copy config and replace dtype aliases with validated `DtypeSpec` objects.

    The benchmark modules import `HQPINN.config.DTYPE` directly, so runtime
    normalization must happen before experiment dispatch.
    """

    def _normalize(value: Any) -> Any:
        if isinstance(value, dict):
            normalized: dict[str, Any] = {}
            for key, child in value.items():
                if _is_dtype_key(key) and child is not None:
                    normalized[key] = coerce_dtype_spec(child)
                else:
                    normalized[key] = _normalize(child)
            return normalized
        if isinstance(value, list):
            return [_normalize(item) for item in value]
        return value

    return _normalize(deepcopy(config))


def apply_runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize runtime config and propagate the selected dtype to `HQPINN.config`.

    This keeps all branch implementations on the same numerical precision during
    differentiation of the paper's residual losses.
    """
    normalized = normalize_dtype_config(config)
    global_dtype = normalized.get("dtype")
    if global_dtype is not None:
        from . import config as project_config

        project_config.set_dtype(dtype_torch(global_dtype))
    return normalized
