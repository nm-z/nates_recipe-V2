"""Compatibility shim exposing auto_optuna package with v1 style API."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from auto_optuna import (
    SystematicOptimizer,
    BattleTestedOptimizer,
    SystematicOptimizerV13,
    KMeansOutlierTransformer,
    IsolationForestTransformer,
    LocalOutlierFactorTransformer,
    OutlierFilterTransformer,
    HSICFeatureSelector,
    CONFIG,
    Colors,
    load_dataset,
    setup_logging,
    console,
    HAS_RICH,
    Tree,
)

__all__ = [
    "SystematicOptimizer",
    "BattleTestedOptimizer",
    "SystematicOptimizerV13",
    "KMeansOutlierTransformer",
    "IsolationForestTransformer",
    "LocalOutlierFactorTransformer",
    "OutlierFilterTransformer",
    "HSICFeatureSelector",
    "CONFIG",
    "Colors",
    "load_dataset",
    "setup_logging",
    "console",
    "HAS_RICH",
    "Tree",
]
