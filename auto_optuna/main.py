"""Simple entry points for running the optimisation pipeline.

This module intentionally avoids any command-line configuration in
order to honour the project's zero‑configuration principle.  A single
``main()`` function runs the default dataset, while convenience
functions allow explicit execution of the three supported datasets.
"""

from __future__ import annotations

from .config import CONFIG, DATASET_FILES, Colors
from .utils import load_dataset, validate_dataset_files
from .optimizer import SystematicOptimizer, BattleTestedOptimizer

DEFAULT_DATASET = CONFIG["DATASET"]["DEFAULT"]


def _run(dataset_num: int, optimizer_type: str = "systematic", trials: int | None = None,
         target_r2: float | None = None) -> dict:
    """Execute the optimisation pipeline for a given dataset."""

    if not validate_dataset_files(dataset_num):
        raise FileNotFoundError(
            f"Dataset {dataset_num} files missing: {DATASET_FILES[dataset_num]}"
        )

    print(f"{Colors.BOLD}{Colors.CYAN}🚀 Starting Auto Optuna Optimisation{Colors.END}")
    print(f"Dataset: {DATASET_FILES[dataset_num]['name']}")
    print(f"Optimizer: {optimizer_type}")

    X, y = load_dataset(dataset_num)

    if optimizer_type == "systematic":
        optimizer = SystematicOptimizer(dataset_num=dataset_num,
                                        max_hyperopt_trials=trials)
        results = optimizer.run_systematic_optimization(X, y)
    else:
        optimizer = BattleTestedOptimizer(
            dataset_num=dataset_num,
            target_r2=target_r2 or CONFIG["THRESHOLDS"]["TARGET_R2"],
            max_trials=trials or CONFIG["OPTUNA"]["MAX_TRIALS_BATTLE_TESTED"],
        )
        results = optimizer.run_optimization(X, y)

    print(f"\n{Colors.BOLD}{Colors.GREEN}🎉 Optimisation Complete!{Colors.END}")
    return results


def main() -> dict:
    """Run the pipeline using the default dataset."""
    return _run(DEFAULT_DATASET)


def run_hold1() -> dict:
    """Run optimisation on Hold‑1 dataset."""
    return _run(1)


def run_hold2() -> dict:
    """Run optimisation on Hold‑2 dataset."""
    return _run(2)


def run_hold3() -> dict:
    """Run optimisation on Hold‑3 (full) dataset."""
    return _run(3)


if __name__ == "__main__":
    main()
