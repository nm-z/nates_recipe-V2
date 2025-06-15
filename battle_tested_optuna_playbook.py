"""Compatibility wrapper for the refactored auto_optuna package."""
from pathlib import Path

from auto_optuna import (
    BattleTestedOptimizer,
    KMeansOutlierTransformer,
    IsolationForestTransformer,
    LocalOutlierFactorTransformer,
)

DATASET = 1


def main():
    """Run Hold-1 optimization using the new package."""
    from auto_optuna.main import run_hold1
    return run_hold1()
