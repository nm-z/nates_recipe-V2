"""Unified interface for Nate's auto_optuna pipeline."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType

CURRENT_FILE = pathlib.Path(__file__).resolve()
V13_PATH = CURRENT_FILE.with_name("../auto_optuna-V1.3.py").resolve()

_spec = importlib.util.spec_from_file_location("auto_optuna_v1_3", V13_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load auto_optuna-V1.3.py at {V13_PATH}")
_auto_optuna_v1_3: ModuleType = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _auto_optuna_v1_3
_spec.loader.exec_module(_auto_optuna_v1_3)

SystematicOptimizerV13 = _auto_optuna_v1_3.SystematicOptimizerV13
Tree = getattr(_auto_optuna_v1_3, "Tree", None)
console = getattr(_auto_optuna_v1_3, "console", None)

from .legacy import (
    OutlierFilterTransformer,
    IsolationForestTransformer,
    LocalOutlierFactorTransformer,
    HSICFeatureSelector,
    BattleTestedOptimizer,
)

__all__ = [
    "SystematicOptimizerV13",
    "Tree",
    "console",
    "OutlierFilterTransformer",
    "IsolationForestTransformer",
    "LocalOutlierFactorTransformer",
    "HSICFeatureSelector",
    "BattleTestedOptimizer",
]

