from pathlib import Path
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from auto_optuna import (
    SystematicOptimizer,
    KMeansOutlierTransformer,
    IsolationForestTransformer,
    LocalOutlierFactorTransformer,
    load_dataset,
    setup_logging,
    CONFIG,
    console,
    Colors,
)

DATASET = CONFIG["DATASET"]["DEFAULT"]


class BattleTestedOptimizer:
    """Simplified legacy pipeline with step-based API."""

    def __init__(self, dataset_num, max_trials=40, target_r2=0.93, **_):
        self.dataset_num = dataset_num
        self.max_trials = max_trials
        self.target_r2 = target_r2
        print(
            f"{Colors.BOLD}{Colors.CYAN}🚀 Battle-Tested ML Optimizer Initialized for Hold-{dataset_num}{Colors.END}"
        )
        print(f"   Target R²: {target_r2}")
        print(f"   Max trials: {max_trials}")
        print(f"   CV strategy: {CONFIG['CV_SPLITS']}-fold × {CONFIG['CV_REPEATS']} repeats")
        self.cv = RepeatedKFold(n_splits=CONFIG["CV_SPLITS"], n_repeats=CONFIG["CV_REPEATS"], random_state=42)
        self.logger = setup_logging(dataset_num)
        self._optimizer: SystematicOptimizer | None = None

    def step_1_pin_down_ceiling(self, X, y):
        if X.shape[0] < 3:
            raise ValueError("not enough samples for training")
        from auto_optuna import optimizer as optimizer_module
        optimizer_module.Path = Path
        self._optimizer = SystematicOptimizer(self.dataset_num, max_hyperopt_trials=self.max_trials)
        self._optimizer.phase_1_data_preparation(X, y)
        self.X, self.y = X, y
        self.X_test, self.y_test = self._optimizer.X_test, self._optimizer.y_test
        return self._optimizer.noise_ceiling, self._optimizer.current_best_r2

    def step_2_bulletproof_preprocessing(self):
        assert self._optimizer is not None
        import numpy as np
        self.X_clean = np.vstack([self._optimizer.X_train, self._optimizer.X_test])
        self.X_test_clean = self._optimizer.X_test
        self.preprocessing_pipeline = Pipeline([("identity", FunctionTransformer())])
        return self.X_clean.shape[1]

    def step_3_optuna_search(self):
        assert self._optimizer is not None
        best_r2, self.best_params = self._optimizer.phase_2_optimization()
        return best_r2, self.best_params

    def step_4_lock_in_champion(self):
        assert self._optimizer is not None
        results = self._optimizer.phase_3_final_evaluation()
        self.best_pipeline = self._optimizer.final_pipeline
        for pkl_file in self._optimizer.model_dir.glob("*.pkl"):
            if "final_model" not in pkl_file.name:
                pkl_file.unlink(missing_ok=True)
        return results.get("test_r2", 0.0), self.best_params


def main():
    """Run the battle-tested optimization pipeline."""
    X, y = load_dataset(DATASET)
    optimizer = BattleTestedOptimizer(dataset_num=DATASET)
    return optimizer.step_4_lock_in_champion()


__all__ = [
    "BattleTestedOptimizer",
    "KMeansOutlierTransformer",
    "IsolationForestTransformer",
    "LocalOutlierFactorTransformer",
    "load_dataset",
    "setup_logging",
    "CONFIG",
    "console",
    "DATASET",
    "main",
    "Path",
]
