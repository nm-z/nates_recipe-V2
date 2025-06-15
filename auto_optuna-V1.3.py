#!/usr/bin/env python3
"""
Battle-Tested End-to-End ML Playbook v1.3 – Incremental Upgrades
===============================================================
Key changes vs v1.2
1. Central CONFIG dictionary – one-stop tuning of CV, Optuna + thresholds
2. Modularised Phase-3 Optuna callback with _evaluate_trial_result()
3. Automatic console refresh throttle for huge datasets

Implementation strategy:  
• Re-use v1.2 implementation via dynamic import to avoid code duplication  
• Sub-class original SystematicOptimizer and surgically override only the
  behaviours that changed.  
• Maintain 100 % interface compatibility so existing downstream code keeps
  working unchanged.

This is the current recommended entry point for new experiments. Older
versions remain available for reproducibility. A concise overview of all
scripts can be found in `docs/optuna_scripts_overview.md`.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import time
from types import ModuleType

# Export legacy transformers for full backwards compatibility
from auto_optuna.legacy import (
    OutlierFilterTransformer,
    IsolationForestTransformer,
    LocalOutlierFactorTransformer,
    HSICFeatureSelector,
    BattleTestedOptimizer,
)

# -----------------------------------------------------------------------------
# 1️⃣  Centralised configuration
# -----------------------------------------------------------------------------
import optuna
from optuna.pruners import MedianPruner

CONFIG = {
    # Cross-validation --------------------------------------------------------
    "CV_SPLITS": 5,
    "CV_REPEATS": 3,

    # Optuna parameters -------------------------------------------------------
    "OPTUNA": {
        "PREPROCESSING_TRIALS": 50,
        "HYPEROPT_TRIALS": 500,
        "PRUNER_PREPROCESSING": MedianPruner(n_startup_trials=5),
        "PRUNER_HYPEROPT": MedianPruner(n_startup_trials=20),
        "SAMPLER": optuna.samplers.TPESampler(seed=42, multivariate=True),
    },

    # Thresholds --------------------------------------------------------------
    "THRESHOLDS": {
        "EXCELLENT": 0.01,      # ≤ 1 % gap to ceiling ⇒ "excellent"
        "NEAR_CEILING": 0.02,   # ≤ 2 % gap ⇒ "near ceiling"
    },

    # Console refresh throttling ---------------------------------------------
    "LARGE_DATASET_SIZE": 10_000,
    "REFRESH_INTERVAL": 10.0,  # seconds
}

# -----------------------------------------------------------------------------
# 2️⃣  Seamlessly import v1.2 implementation so we can extend it
# -----------------------------------------------------------------------------
CURRENT_FILE = pathlib.Path(__file__).resolve()
V12_PATH = CURRENT_FILE.with_name("auto_optuna-V1.2.py")

spec = importlib.util.spec_from_file_location("auto_optuna_v1_2", V12_PATH)
if spec is None or spec.loader is None:  # pragma: no cover – should never happen
    raise ImportError(f"Could not locate auto_optuna-V1.2.py at {V12_PATH}")

auto_optuna_v1_2: ModuleType = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
sys.modules[spec.name] = auto_optuna_v1_2  # make importable elsewhere
spec.loader.exec_module(auto_optuna_v1_2)  # type: ignore[arg-type]

# Extract common Rich helpers if available
HAS_RICH = getattr(auto_optuna_v1_2, "HAS_RICH", False)
if HAS_RICH:
    Tree = auto_optuna_v1_2.Tree       # noqa: N806 (keep camel-case to match Rich API)
    console = auto_optuna_v1_2.console

# -----------------------------------------------------------------------------
# 3️⃣  Extended SystematicOptimizer (v1.3)
# -----------------------------------------------------------------------------

class SystematicOptimizerV13(auto_optuna_v1_2.SystematicOptimizer):
    """Drop-in replacement adding CONFIG centralisation + console throttling."""

    # ------------------------------------------------------------------
    # Construction / basic setup
    # ------------------------------------------------------------------
    def __init__(self, dataset_num: int,
                 max_preprocessing_trials: int | None = None,
                 max_hyperopt_trials: int | None = None):
        if max_preprocessing_trials is None:
            max_preprocessing_trials = CONFIG["OPTUNA"]["PREPROCESSING_TRIALS"]
        if max_hyperopt_trials is None:
            max_hyperopt_trials = CONFIG["OPTUNA"]["HYPEROPT_TRIALS"]

        super().__init__(dataset_num,
                         max_preprocessing_trials=max_preprocessing_trials,
                         max_hyperopt_trials=max_hyperopt_trials)

        # Override CV params from CONFIG
        self.cv_splits = CONFIG["CV_SPLITS"]
        self.cv_repeats = CONFIG["CV_REPEATS"]

        # Throttling helpers
        self.last_refresh_time: float = 0.0

    # ------------------------------------------------------------------
    # Rich console throttling
    # ------------------------------------------------------------------
    def _patch_console_for_throttle(self) -> None:  # called once after X_train exists
        if not HAS_RICH or len(self.X_train) <= CONFIG["LARGE_DATASET_SIZE"]:
            return  # no need to throttle

        original_print = console.print  # keep reference for wrapping
        self.last_refresh_time = time.time()

        def throttled_print(*args, **kwargs):  # noqa: D401 – verb form intentional
            now = time.time()
            if now - self.last_refresh_time > CONFIG["REFRESH_INTERVAL"]:
                original_print(*args, **kwargs)
                self.last_refresh_time = now
        console.print = throttled_print  # type: ignore[assignment]

    # Inject throttle once data size known (end of Phase-1 train/test split)
    def phase_1_model_family_tournament(self, X, y):  # type: ignore[override]
        result = super().phase_1_model_family_tournament(X, y)
        # X_train populated by parent – safe to evaluate size now
        self._patch_console_for_throttle()
        return result

    # ------------------------------------------------------------------
    # 4️⃣  Modular callback helpers for Phase-3
    # ------------------------------------------------------------------
    def _evaluate_trial_result(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Centralised logic for Optuna progress-reporting & ceiling checks."""
        from optuna.trial import TrialState  # local import to avoid polluting top-level

        if trial.state != TrialState.COMPLETE:
            return  # We only act on successful trials – prune/fail handled silently

        improvement = study.best_value - self.current_best_r2
        if improvement < 1e-3:  # ignore minuscule gains
            return

        ceiling_gap = abs(study.best_value - self.noise_ceiling) if self.noise_ceiling else float("inf")

        if HAS_RICH:
            title = f"🔥 Trial {trial.number}: R²={trial.value:.4f} (+{improvement:.4f})"
            trial_tree = Tree(title)
            trial_tree.add(f"Best so far: {study.best_value:.4f}")
            trial_tree.add(f"Gap to ceiling: {ceiling_gap:.4f}")

            if ceiling_gap <= CONFIG["THRESHOLDS"]["EXCELLENT"]:
                trial_tree.add("🎉 EXCELLENT — within 1 % of ceiling!")
            elif ceiling_gap <= CONFIG["THRESHOLDS"]["NEAR_CEILING"]:
                trial_tree.add("✅ Near ceiling — within 2 %.")

            console.print(trial_tree)

    # ------------------------------------------------------------------
    # 5️⃣  Overridden Phase-3 with clean callback
    # ------------------------------------------------------------------
    def phase_3_deep_hyperparameter_optimization(self):  # type: ignore[override]
        if HAS_RICH:
            header = Tree("🔥 PHASE 3 (v1.3): Deep Hyperparameter Optimisation")
            header.add(f"Target model: {self.winning_model_class.__name__}")
            header.add(f"Current best R²: {self.current_best_r2:.4f}")
            header.add(f"Noise ceiling: {self.noise_ceiling:.4f}")
            console.print(header)

        # Create Optuna study with centralised parameters
        study = optuna.create_study(
            direction="maximize",
            pruner=CONFIG["OPTUNA"]["PRUNER_HYPEROPT"],
            sampler=CONFIG["OPTUNA"]["SAMPLER"],
        )

        # Run optimisation
        study.optimize(
            self._deep_hyperopt_objective,
            n_trials=self.max_hyperopt_trials,
            callbacks=[self._evaluate_trial_result],
            show_progress_bar=False,
        )

        # Post-processing identical to v1.2 implementation ------------------
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            self.logger.error("No successful trials in Phase 3 – returning baseline.")
            return self.current_best_r2, {}

        best_r2 = study.best_value
        improvement = best_r2 - self.current_best_r2

        if HAS_RICH:
            summary = Tree("📊 Phase 3 Summary")
            summary.add(f"Trials completed: {len(completed_trials)}")
            summary.add(f"Best R²: {best_r2:.4f} (+{improvement:.4f})")
            console.print(summary)

        # Build & persist final pipeline -----------------------------------
        self.final_pipeline = self._build_final_pipeline(study.best_params)
        model_path = self.model_dir / f"hold{self.dataset_num}_final_model.pkl"
        import joblib
        joblib.dump(self.final_pipeline, model_path)
        self.logger.info(f"Phase 3 complete – final model saved to {model_path}")

        # Update tracker state so downstream phases have latest metrics
        self.current_best_r2 = best_r2
        return best_r2, study.best_params

# -----------------------------------------------------------------------------
# 6️⃣  Minimal runnable entry-point mirroring v1.2 main()
# -----------------------------------------------------------------------------

DATASET = 3  # 1 = Hold-1, 2 = Hold-2, 3 = Hold-3 (default)

import pandas as pd
import numpy as np

def _load_dataset(dataset_id: int):
    """Utility replicating dataset loader from v1.2."""
    if dataset_id == 1:
        X = pd.read_csv('Predictors_Hold-1_2025-04-14_18-28.csv', header=None).values.astype(np.float32)
        y = pd.read_csv('9_10_24_Hold_01_targets.csv', header=None).values.astype(np.float32).ravel()
    elif dataset_id == 2:
        X = pd.read_csv('hold2_predictor.csv', header=None).values.astype(np.float32)
        y = pd.read_csv('hold2_target.csv', header=None).values.astype(np.float32).ravel()
    elif dataset_id == 3:
        X = pd.read_csv('predictors_Hold 1 Full_20250527_151252.csv', header=None).values.astype(np.float32)
        y = pd.read_csv('targets_Hold 1 Full_20250527_151252.csv', header=None).values.astype(np.float32).ravel()
    else:
        raise ValueError(f"Invalid DATASET value: {dataset_id} (must be 1, 2 or 3)")
    return X, y


def main():  # noqa: D401 – imperative entry-point
    X, y = _load_dataset(DATASET)

    optimizer = SystematicOptimizerV13(dataset_num=DATASET)
    return optimizer.run_systematic_optimization(X, y)


if __name__ == "__main__":
    _ = main() 