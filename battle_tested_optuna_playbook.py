"""Compatibility wrapper exposing the modular auto_optuna package under the
legacy name `battle_tested_optuna_playbook` expected by tests."""

from auto_optuna import (
    BattleTestedOptimizer,
    KMeansOutlierTransformer,
    IsolationForestTransformer,
    LocalOutlierFactorTransformer,
)
from pathlib import Path

# Hardcoded dataset number to preserve zero configuration
DATASET = 1

__all__ = [
    "BattleTestedOptimizer",
    "KMeansOutlierTransformer",
    "IsolationForestTransformer",
    "LocalOutlierFactorTransformer",
    "DATASET",
    "Path",
    "main",
]


def main():
    """Minimal entry point preserved for legacy tests."""
    pass
