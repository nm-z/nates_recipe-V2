"""Legacy battle-tested Optuna playbook."""

from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location("auto_optuna_V1", "auto_optuna-V1.py")
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

BattleTestedOptimizer = _mod.BattleTestedOptimizer
KMeansOutlierTransformer = _mod.OutlierFilterTransformer
IsolationForestTransformer = _mod.IsolationForestTransformer
LocalOutlierFactorTransformer = _mod.LocalOutlierFactorTransformer
from auto_optuna.utils import load_dataset

DATASET = 1  # 1=Hold-1, 2=Hold-2, 3=Hold-1 Full


def main():
    X, y = load_dataset(DATASET)
    optimizer = BattleTestedOptimizer(dataset_num=DATASET)
    return optimizer.run_optimization(X, y)


__all__ = [
    "BattleTestedOptimizer",
    "KMeansOutlierTransformer",
    "IsolationForestTransformer",
    "LocalOutlierFactorTransformer",
    "DATASET",
    "main",
]

