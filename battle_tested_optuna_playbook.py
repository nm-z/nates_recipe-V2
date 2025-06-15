#!/usr/bin/env python3
"""Compatibility wrapper exposing the legacy API used in tests."""

from pathlib import Path

from auto_optuna import (
    KMeansOutlierTransformer,
    IsolationForestTransformer,
    LocalOutlierFactorTransformer,
    SystematicOptimizer,
    BattleTestedOptimizer,
    CONFIG,
    Colors,
)

# Hardcoded dataset selection (1 = Hold-1 by default)
DATASET = CONFIG["DATASET"].get("DEFAULT", 1)


def main():
    """Minimal entry point used for smoke tests."""
    return {
        "dataset": DATASET,
        "optimizer_class": BattleTestedOptimizer,
    }
