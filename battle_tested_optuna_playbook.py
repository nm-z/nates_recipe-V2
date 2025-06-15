#!/usr/bin/env python3
"""Compatibility wrapper for legacy imports.

This module re-exports the key classes from the new ``auto_optuna`` package so
existing tests and scripts that rely on ``battle_tested_optuna_playbook``
continue to work.
"""

from auto_optuna import (
    BattleTestedOptimizer,
    SystematicOptimizer,
    KMeansOutlierTransformer,
    IsolationForestTransformer,
    LocalOutlierFactorTransformer,
    CONFIG,
)
from pathlib import Path
from auto_optuna.main import run_hold1, run_hold2, run_hold3

# Default dataset to use when ``main()`` is executed
DATASET = CONFIG["DATASET"]["DEFAULT"]


def main():
    """Run the optimization for the configured dataset without CLI arguments."""
    if DATASET == 1:
        return run_hold1()
    if DATASET == 2:
        return run_hold2()
    return run_hold3()

__all__ = [
    "BattleTestedOptimizer",
    "SystematicOptimizer",
    "KMeansOutlierTransformer",
    "IsolationForestTransformer",
    "LocalOutlierFactorTransformer",
    "DATASET",
    "main",
    "Path",
]
