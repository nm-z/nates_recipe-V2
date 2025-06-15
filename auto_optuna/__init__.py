"""
Auto Optuna Package
==================
Modularized ML optimization components extracted from the original auto_optuna scripts.

Main Components:
- SystematicOptimizer: Main optimization class
- Transformers: Custom preprocessing transformers
- Config: Configuration settings
- Utils: Utility functions
"""

from .optimizer import SystematicOptimizer, BattleTestedOptimizer, SystematicOptimizerV13
from .transformers import (
    KMeansOutlierTransformer,
    IsolationForestTransformer,
    LocalOutlierFactorTransformer,
    OutlierFilterTransformer,
    HSICFeatureSelector,
)
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, MinMaxScaler
from sklearn.decomposition import KernelPCA, TruncatedSVD
from sklearn.feature_selection import RFECV
from sklearn.compose import TransformedTargetRegressor
from .config import CONFIG, Colors
from .utils import load_dataset, setup_logging, console, HAS_RICH, Tree

__version__ = "1.3.0"
__all__ = [
    "SystematicOptimizer",
    "BattleTestedOptimizer",
    "SystematicOptimizerV13",
    "KMeansOutlierTransformer",
    "IsolationForestTransformer",
    "LocalOutlierFactorTransformer",
    "OutlierFilterTransformer",
    "HSICFeatureSelector",
    "QuantileTransformer",
    "PowerTransformer",
    "MinMaxScaler",
    "RFECV",
    "KernelPCA",
    "TruncatedSVD",
    "TransformedTargetRegressor",
    "CONFIG",
    "Colors",
    "load_dataset",
    "setup_logging"
    ,
    "console",
    "HAS_RICH",
    "Tree",
]
