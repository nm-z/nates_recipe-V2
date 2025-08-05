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

# Always use relative imports for proper module structure
try:
    from .optimizer import SystematicOptimizer, BattleTestedOptimizer
except ImportError:
    from optimizer import SystematicOptimizer, BattleTestedOptimizer

try:
    from .transformers import (
        KMeansOutlierTransformer,
        IsolationForestTransformer,
        LocalOutlierFactorTransformer,
        OutlierFilterTransformer,
        HSICFeatureSelector,
    )
except ImportError:
    from transformers import (
        KMeansOutlierTransformer,
        IsolationForestTransformer,
        LocalOutlierFactorTransformer,
        OutlierFilterTransformer,
        HSICFeatureSelector,
    )

try:
    from .config import CONFIG, Colors
except ImportError:
    from config import CONFIG, Colors

try:
    from .utils import load_dataset, setup_logging, console, HAS_RICH, Tree
except ImportError:
    from utils import load_dataset, setup_logging, console, HAS_RICH, Tree

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