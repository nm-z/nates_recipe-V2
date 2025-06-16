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

if __package__ in (None, ""):
    # Support running without installing as a package
    from optimizer import SystematicOptimizer, BattleTestedOptimizer
    from transformers import (
        KMeansOutlierTransformer,
        IsolationForestTransformer,
        LocalOutlierFactorTransformer,
        OutlierFilterTransformer,
        HSICFeatureSelector,
    )
    from config import CONFIG, Colors
    from utils import load_dataset, setup_logging, console, HAS_RICH, Tree
else:
    from .optimizer import SystematicOptimizer, BattleTestedOptimizer
    from .transformers import (
        KMeansOutlierTransformer,
        IsolationForestTransformer,
        LocalOutlierFactorTransformer,
        OutlierFilterTransformer,
        HSICFeatureSelector,
    )
    from .config import CONFIG, Colors
    from .utils import load_dataset, setup_logging, console, HAS_RICH, Tree

from sklearn.preprocessing import QuantileTransformer, PowerTransformer, MinMaxScaler
from sklearn.feature_selection import RFECV
from sklearn.decomposition import KernelPCA, TruncatedSVD
from sklearn.compose import TransformedTargetRegressor

__version__ = "1.3.0"
__all__ = [
    "SystematicOptimizer",
    "BattleTestedOptimizer", 
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
    "setup_logging",
    "console",
    "HAS_RICH",
    "Tree",
] 