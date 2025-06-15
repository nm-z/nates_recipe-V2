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

from .optimizer import SystematicOptimizer, BattleTestedOptimizer
from .transformers import (
    KMeansOutlierTransformer,
    IsolationForestTransformer, 
    LocalOutlierFactorTransformer
)
from .config import CONFIG, Colors
from .utils import load_dataset, setup_logging

__version__ = "1.3.0"
__all__ = [
    "SystematicOptimizer",
    "BattleTestedOptimizer", 
    "KMeansOutlierTransformer",
    "IsolationForestTransformer",
    "LocalOutlierFactorTransformer",
    "CONFIG",
    "Colors",
    "load_dataset",
    "setup_logging"
]
