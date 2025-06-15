#!/usr/bin/env python3
"""Battle-tested pipeline wrapper using modular components."""

from __future__ import annotations

from pathlib import Path
import joblib

from auto_optuna import (
    SystematicOptimizer,
    KMeansOutlierTransformer,
    IsolationForestTransformer,
    LocalOutlierFactorTransformer,
    CONFIG,
    load_dataset,
)
from auto_optuna.utils import estimate_noise_ceiling, save_model_artifacts, create_diagnostic_plots, setup_logging
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge

DATASET = CONFIG["DATASET"]["DEFAULT"]


class BattleTestedOptimizer:
    """Step-by-step optimization pipeline compatible with tests."""

    def __init__(self, dataset_num: int, target_r2: float = 0.93, max_trials: int = 40):
        self.dataset_num = dataset_num
        self.target_r2 = target_r2
        self.max_trials = max_trials
        self.logger = setup_logging(dataset_num, Path(CONFIG["PATHS"]["MODEL_DIR_TEMPLATE"].format(dataset_num=dataset_num)))
        self.cv = SystematicOptimizer(dataset_num, max_hyperopt_trials=max_trials).cv

    # ------------------------------------------------------------------
    def step_1_pin_down_ceiling(self, X, y):
        self.X, self.X_test, self.y, self.y_test = train_test_split(
            X,
            y,
            test_size=CONFIG["DATASET"]["TEST_SIZE"],
            random_state=CONFIG["DATASET"]["RANDOM_STATE"],
        )
        self.noise_ceiling, self.baseline_r2 = estimate_noise_ceiling(self.X, self.y, self.cv)
        return self.noise_ceiling, self.baseline_r2

    # ------------------------------------------------------------------
    def step_2_bulletproof_preprocessing(self):
        vt = VarianceThreshold(CONFIG["PREPROCESSING"]["VARIANCE_THRESHOLD"])
        sc = RobustScaler(quantile_range=CONFIG["PREPROCESSING"]["QUANTILE_RANGE"])
        self.X_clean = sc.fit_transform(vt.fit_transform(self.X))
        self.X_test_clean = sc.transform(vt.transform(self.X_test))
        self.preprocessing_pipeline = {"var_threshold": vt, "scaler": sc}
        return self.X_clean.shape[1]

    # ------------------------------------------------------------------
    def step_3_optuna_search(self):
        opt = SystematicOptimizer(self.dataset_num, max_hyperopt_trials=self.max_trials)
        opt.X_train = self.X_clean
        opt.y_train = self.y
        opt.X_test = self.X_test_clean
        opt.y_test = self.y_test
        opt.noise_ceiling = self.noise_ceiling
        opt.current_best_r2 = self.baseline_r2
        opt.cv = self.cv
        opt.preprocessing_components = self.preprocessing_pipeline
        opt.phase_2_model_family_tournament()
        opt.phase_2_optimization()
        self.best_params = opt.study.best_params
        self.best_pipeline = opt.final_pipeline
        return opt.study.best_value

    # ------------------------------------------------------------------
    def step_4_lock_in_champion(self):
        self.best_pipeline.fit(self.X_clean, self.y)
        preds = self.best_pipeline.predict(self.X_test_clean)
        r2 = Ridge().fit(self.X_clean, self.y).score(self.X_test_clean, self.y_test)
        model_dir = Path(CONFIG["PATHS"]["MODEL_DIR_TEMPLATE"].format(dataset_num=self.dataset_num))
        create_diagnostic_plots(self.y_test, preds, dataset_num=self.dataset_num, model_dir=model_dir)
        model_dir.mkdir(exist_ok=True)
        model_file = model_dir / CONFIG["PATHS"]["MODEL_FILE_TEMPLATE"].format(dataset_num=self.dataset_num)
        joblib.dump(self.best_pipeline, model_file)
        with open(model_dir / CONFIG["PATHS"]["RESULTS_FILE_TEMPLATE"].format(dataset_num=self.dataset_num), "w") as f:
            f.write(f"test_r2: {r2}\n")
            f.write(f"noise_ceiling: {self.noise_ceiling}\n")
            f.write(f"cv_best_r2: {self.baseline_r2}\n")
        return r2, self.best_params


def main():
    X, y = load_dataset(DATASET)
    opt = BattleTestedOptimizer(dataset_num=DATASET, max_trials=CONFIG["OPTUNA"]["MAX_TRIALS_BATTLE_TESTED"])
    opt.step_1_pin_down_ceiling(X, y)
    opt.step_2_bulletproof_preprocessing()
    opt.step_3_optuna_search()
    opt.step_4_lock_in_champion()


if __name__ == "__main__":
    main()

# Re-export transformers for tests
__all__ = [
    "BattleTestedOptimizer",
    "KMeansOutlierTransformer",
    "IsolationForestTransformer",
    "LocalOutlierFactorTransformer",
]
