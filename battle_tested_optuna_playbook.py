"""Compatibility wrapper for legacy scripts."""

from pathlib import Path
from sklearn.pipeline import Pipeline
import joblib
from auto_optuna import (
    SystematicOptimizer,
    KMeansOutlierTransformer,
    IsolationForestTransformer,
    LocalOutlierFactorTransformer,
    load_dataset,
    console,
    Tree,
    HAS_RICH,
    Colors,
    CONFIG,
    setup_logging,
)

DATASET = CONFIG["DATASET"]["DEFAULT"]


class BattleTestedOptimizer(SystematicOptimizer):
    """Compatibility layer mimicking the original playbook API."""

    def __init__(self, dataset_num=DATASET, target_r2=0.93, max_trials=40, **kwargs):
        super().__init__(dataset_num, max_hyperopt_trials=max_trials)
        self.target_r2 = target_r2
        self.max_trials = max_trials
        # Override model directory using this module's Path (allows patching)
        self.model_dir = Path(CONFIG["PATHS"]["MODEL_DIR_TEMPLATE"].format(dataset_num=dataset_num))
        self.logger = setup_logging(dataset_num, self.model_dir)

    def step_1_pin_down_ceiling(self, X, y):
        self.X = X
        self.y = y
        try:
            self.phase_1_data_preparation(X, y)
        except Exception as e:
            raise ValueError(f"samples issue: {e}") from e
        self.X_test_indices_ = None
        return self.noise_ceiling, self.current_best_r2

    def step_2_bulletproof_preprocessing(self):
        self.X_clean = self.X
        self.X_test_clean = self.X_test
        self.preprocessing_pipeline = Pipeline([])
        return self.X_train.shape[1]

    def step_3_optuna_search(self):
        self.phase_2_optimization()

    def step_4_lock_in_champion(self):
        results = self.phase_3_final_evaluation()
        self.best_pipeline = self.final_pipeline
        model_file = self.model_dir / f"hold{self.dataset_num}_final_model.pkl"
        joblib.dump(self.best_pipeline, model_file)
        for pkl in self.model_dir.glob("*.pkl"):
            if pkl.name != model_file.name:
                pkl.unlink(missing_ok=True)
        params = self.study.best_params if hasattr(self, "study") else {}
        return results.get("test_r2", 0.0), params


def main() -> dict:
    X, y = load_dataset(DATASET)
    optimizer = BattleTestedOptimizer(dataset_num=DATASET)
    optimizer.step_1_pin_down_ceiling(X, y)
    optimizer.step_2_bulletproof_preprocessing()
    optimizer.step_3_optuna_search()
    final_r2, _ = optimizer.step_4_lock_in_champion()
    return {"test_r2": final_r2}

__all__ = [
    "BattleTestedOptimizer",
    "KMeansOutlierTransformer",
    "IsolationForestTransformer",
    "LocalOutlierFactorTransformer",
    "load_dataset",
    "console",
    "Tree",
    "HAS_RICH",
    "CONFIG",
    "Colors",
    "DATASET",
    "Path",
    "main",
]
