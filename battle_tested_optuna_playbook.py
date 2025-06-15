"""Battle-Tested Optuna Playbook
===============================
Legacy-style interface used in tests.
"""

from pathlib import Path
import auto_optuna.optimizer as _optmod

from auto_optuna import (
    SystematicOptimizer,
    KMeansOutlierTransformer,
    IsolationForestTransformer,
    LocalOutlierFactorTransformer,
    _load_dataset,
)

# Hardcoded dataset number as required by the zero-configuration principle
DATASET = 1


class BattleTestedOptimizer:
    """Thin wrapper providing step-based API for compatibility."""

    def __init__(self, dataset_num: int, target_r2: float = 0.93, max_trials: int = 40):
        self.dataset_num = dataset_num
        self.target_r2 = target_r2
        self.max_trials = max_trials
        _optmod.Path = Path
        self.optimizer = SystematicOptimizer(dataset_num, max_hyperopt_trials=max_trials)
        self.cv = self.optimizer.cv
        self.logger = self.optimizer.logger
        print(
            f"\033[1m\033[96m🚀 Battle-Tested ML Optimizer Initialized for Hold-{dataset_num}\033[0m"
        )
        print(f"   Target R²: {target_r2}")
        print(f"   Max trials: {max_trials}")
        print(f"   CV strategy: {self.optimizer.cv_splits}-fold × {self.optimizer.cv_repeats} repeats")

    def step_1_pin_down_ceiling(self, X, y):
        """Prepare data and estimate noise ceiling."""
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import train_test_split, cross_val_score

        self.X = X
        self.y = y
        train_idx, test_idx = train_test_split(
            range(len(X)), test_size=0.2, random_state=42
        )
        self._train_idx = train_idx
        self._test_idx = test_idx
        self.X_train = X[train_idx]
        self.X_test = X[test_idx]
        self.y_train = y[train_idx]
        self.y_test = y[test_idx]

        ridge = Ridge(alpha=1.0, random_state=42)
        scores = cross_val_score(ridge, self.X_train, self.y_train, cv=self.cv, scoring="r2")
        self.noise_ceiling = scores.mean() + 2 * scores.std()
        self.baseline_r2 = scores.mean()
        self.optimizer.preprocessing_components = {}
        return self.noise_ceiling, self.baseline_r2

    def step_2_bulletproof_preprocessing(self):
        """Apply simple scaling to training and test data."""
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        full_scaled = scaler.fit_transform(self.X)
        self.X_clean = full_scaled
        self.X_test_clean = full_scaled[self._test_idx]
        self.X_train_clean = full_scaled[self._train_idx]
        self.preprocessing_pipeline = scaler
        # Update optimizer's data for subsequent steps
        self.optimizer.X_train = self.X_train_clean
        self.optimizer.X_test = self.X_test_clean
        self.optimizer.y_train = self.y_train
        self.optimizer.y_test = self.y_test
        self.optimizer.preprocessing_components = {}
        return self.X_clean.shape[1]

    def step_3_optuna_search(self):
        """Run hyperparameter optimization."""
        best_r2, best_params = self.optimizer.phase_2_optimization()
        self.best_params = best_params
        return best_r2, best_params

    def step_4_lock_in_champion(self):
        """Evaluate final model and persist artifacts."""
        results = self.optimizer.phase_3_final_evaluation()
        self.best_pipeline = self.optimizer.final_pipeline
        return results["test_r2"], getattr(self.optimizer, "study", None).best_params


def main():
    """Execute the full pipeline using the hardcoded dataset."""
    X, y = _load_dataset(DATASET)
    opt = BattleTestedOptimizer(dataset_num=DATASET)
    opt.step_1_pin_down_ceiling(X, y)
    opt.step_2_bulletproof_preprocessing()
    opt.step_3_optuna_search()
    final_r2, _ = opt.step_4_lock_in_champion()
    return final_r2


__all__ = [
    "BattleTestedOptimizer",
    "KMeansOutlierTransformer",
    "IsolationForestTransformer",
    "LocalOutlierFactorTransformer",
    "Path",
    "DATASET",
    "main",
]
