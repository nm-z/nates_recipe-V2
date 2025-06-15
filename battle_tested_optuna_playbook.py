#!/usr/bin/env python3
"""Battle-tested Auto Optuna playbook.

Provides a zero-configuration entry point to train models on a
predefined dataset. This mirrors the original monolithic script
interface so tests and legacy workflows keep working.
"""

from pathlib import Path
from typing import Tuple, Dict
import joblib

from auto_optuna import (
    SystematicOptimizer,
    KMeansOutlierTransformer,
    IsolationForestTransformer,
    LocalOutlierFactorTransformer,
    load_dataset,
    setup_logging,
    Colors,
)

# Hardcoded dataset choice to comply with the zero configuration policy
DATASET = 1  # 1 = Hold-1, 2 = Hold-2, 3 = Hold-1 Full


class BattleTestedOptimizer:
    """Step-by-step wrapper around :class:`SystematicOptimizer`."""

    def __init__(self, dataset_num: int = DATASET, target_r2: float = 0.93,
                 max_trials: int = 40):
        self.dataset_num = dataset_num
        self.target_r2 = target_r2
        self.max_trials = max_trials

        # Use Path from this module so tests can patch it
        self.model_dir = Path(
            f"best_model_hold{dataset_num}"
        )

        # Underlying optimizer
        self._systematic = SystematicOptimizer(
            dataset_num, max_hyperopt_trials=max_trials
        )
        # Override directories and logging to honour patched Path
        self._systematic.model_dir = self.model_dir
        self._systematic.logger = setup_logging(dataset_num, self.model_dir)

        self.cv = self._systematic.cv
        self.logger = self._systematic.logger
        self.best_pipeline = None
        self.best_params: Dict[str, float] | None = None
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.X_clean = None
        self.X_test_clean = None
        self.preprocessing_pipeline = None
        self.noise_ceiling = None
        self.current_best_r2 = None

    def step_1_pin_down_ceiling(self, X, y) -> Tuple[float, float]:
        """Prepare data and estimate the noise ceiling."""
        self.X = X
        self.y = y
        try:
            self._systematic.phase_1_data_preparation(X, y)
        except Exception as exc:  # pragma: no cover - forward readable message
            raise ValueError("insufficient samples") from exc
        self.X_test = self._systematic.X_test
        self.y_test = self._systematic.y_test
        self.noise_ceiling = self._systematic.noise_ceiling
        self.current_best_r2 = self._systematic.current_best_r2
        return float(self.noise_ceiling), float(self.current_best_r2)

    def step_2_bulletproof_preprocessing(self) -> int:
        """Apply robust preprocessing and return feature count."""
        comps = self._systematic.preprocessing_components
        Xt = comps['var_threshold'].transform(self.X)
        Xt = comps['scaler'].transform(Xt)
        self.X_clean = Xt
        self.X_test_clean = self._systematic.X_test
        self.preprocessing_pipeline = comps
        return int(self.X_clean.shape[1])

    def step_3_optuna_search(self) -> Tuple[float, Dict[str, float]]:
        """Run the Optuna hyperparameter search."""
        best_r2, best_params = self._systematic.phase_2_optimization()
        self.best_params = best_params
        self.best_pipeline = self._systematic.final_pipeline
        return float(best_r2), best_params

    def step_4_lock_in_champion(self) -> Tuple[float, Dict[str, float]]:
        """Finalize training and save the best model."""
        results = self._systematic.phase_3_final_evaluation()
        self.best_pipeline = self._systematic.final_pipeline
        # Ensure model artifacts exist in the configured directory
        model_file = self.model_dir / f"hold{self.dataset_num}_final_model.pkl"
        joblib.dump(self.best_pipeline, model_file)
        # Remove intermediate transformer pickles to keep only the final model
        for extra in self.model_dir.glob("hold*_*.pkl"):
            if extra.name != model_file.name:
                extra.unlink(missing_ok=True)
        return float(results["test_r2"]), self.best_params or {}

    def run_optimization(self, X, y) -> Tuple[float, Dict[str, float]]:
        """Convenience wrapper executing all steps."""
        self.step_1_pin_down_ceiling(X, y)
        self.step_2_bulletproof_preprocessing()
        self.step_3_optuna_search()
        return self.step_4_lock_in_champion()


def main() -> Tuple[float, Dict[str, float]]:
    """Execute the full pipeline using the hardcoded dataset."""
    X, y = load_dataset(DATASET)
    optimizer = BattleTestedOptimizer(dataset_num=DATASET)
    return optimizer.run_optimization(X, y)


# Re-export transformer classes for backwards compatibility
KMeansOutlierTransformer = KMeansOutlierTransformer
IsolationForestTransformer = IsolationForestTransformer
LocalOutlierFactorTransformer = LocalOutlierFactorTransformer


if __name__ == "__main__":
    final_r2, params = main()
    print(f"{Colors.GREEN}Final R²: {final_r2:.4f}{Colors.END}")
    print(f"Best params: {params}")

