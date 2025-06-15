from __future__ import annotations

"""Production-ready pipeline entry point.

This script maintains backwards compatibility with the original
``battle_tested_optuna_playbook.py`` while delegating all heavy
lifting to the ``auto_optuna`` package. It exposes the same
classes and ``main()`` function used by tests and tooling.
"""

from auto_optuna import (
    BattleTestedOptimizer as _BattleTestedOptimizer,
    SystematicOptimizer,
    KMeansOutlierTransformer,
    IsolationForestTransformer,
    LocalOutlierFactorTransformer,
    load_dataset,
)
from pathlib import Path

# Hardcoded dataset selection abiding by zero‑configuration principle
DATASET: int = 1  # 1 = Hold‑1, 2 = Hold‑2, 3 = Hold‑1 Full


__all__ = [
    "BattleTestedOptimizer",
    "SystematicOptimizer",
    "KMeansOutlierTransformer",
    "IsolationForestTransformer",
    "LocalOutlierFactorTransformer",
    "DATASET",
    "main",
]


class BattleTestedOptimizer(_BattleTestedOptimizer):
    """Wrapper injecting this module's ``Path`` class."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("path_class", Path)
        super().__init__(*args, **kwargs)


def main() -> dict:
    """Run the optimisation pipeline on the selected dataset."""
    X, y = load_dataset(DATASET)
    optimizer = BattleTestedOptimizer(dataset_num=DATASET)
    return optimizer.run_optimization(X, y)


if __name__ == "__main__":
    main()
