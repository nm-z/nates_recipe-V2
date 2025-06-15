"""
Optimizer Module
===============
Main optimization classes for systematic ML model optimization.
"""

import numpy as np
import pandas as pd
import time
import joblib
from pathlib import Path

# Core ML libraries
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_regression
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Models
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor

# Metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Optuna
import optuna
from optuna.pruners import MedianPruner

from .config import CONFIG, Colors
from .transformers import KMeansOutlierTransformer, IsolationForestTransformer, LocalOutlierFactorTransformer
from .utils import (
    setup_logging,
    save_model_artifacts,
    create_diagnostic_plots,
    print_results_summary,
    console,
    Tree,
    HAS_RICH,
)

import warnings
warnings.filterwarnings('ignore')


class SystematicOptimizer:
    """Systematic ML optimizer with comprehensive preprocessing and hyperparameter optimization."""
    
    def __init__(self, dataset_num: int, max_hyperopt_trials: int = None):
        self.dataset_num = dataset_num
        self.max_hyperopt_trials = max_hyperopt_trials or CONFIG["OPTUNA"]["HYPEROPT_TRIALS"]
        
        # CV configuration
        self.cv_splits = CONFIG["CV_SPLITS"]
        self.cv_repeats = CONFIG["CV_REPEATS"]
        self.cv = RepeatedKFold(n_splits=self.cv_splits, n_repeats=self.cv_repeats, random_state=42)
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.noise_ceiling = None
        self.current_best_r2 = 0.0
        self.final_pipeline = None
        
        # Setup logging and directories
        self.model_dir = Path(CONFIG["PATHS"]["MODEL_DIR_TEMPLATE"].format(dataset_num=dataset_num))
        self.logger = setup_logging(dataset_num, self.model_dir)
        
        if HAS_RICH:
            init_tree = Tree(f"🚀 SystematicOptimizer initialized for Hold-{dataset_num}")
            init_tree.add(f"Max trials: {self.max_hyperopt_trials}")
            init_tree.add(f"CV strategy: {self.cv_splits}-fold × {self.cv_repeats} repeats")
            console.print(init_tree)
        else:
            print(f"{Colors.BOLD}{Colors.CYAN}🚀 SystematicOptimizer initialized for Hold-{dataset_num}{Colors.END}")
            print(f"   Max trials: {self.max_hyperopt_trials}")
            print(f"   CV strategy: {self.cv_splits}-fold × {self.cv_repeats} repeats")
    
    def run_systematic_optimization(self, X, y):
        """Main optimization pipeline."""
        start_time = time.time()
        
        # Phase 1: Data preparation and noise ceiling
        self.phase_1_data_preparation(X, y)
        
        # Phase 2: Model optimization
        best_r2, best_params = self.phase_2_optimization()
        
        # Phase 3: Final evaluation
        results = self.phase_3_final_evaluation()
        
        elapsed = time.time() - start_time
        self.logger.info(f"Total optimization time: {elapsed:.1f} seconds")
        
        print_results_summary(results, self.dataset_num)
        
        return results
    
    def phase_1_data_preparation(self, X, y):
        """Phase 1: Data preparation and noise ceiling estimation."""
        if HAS_RICH:
            phase_tree = Tree("📊 Phase 1: Data Preparation")
        else:
            print(f"\n{Colors.BOLD}📊 Phase 1: Data Preparation{Colors.END}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=CONFIG["DATASET"]["TEST_SIZE"], 
            random_state=CONFIG["DATASET"]["RANDOM_STATE"]
        )
        
        # Basic preprocessing
        var_threshold = VarianceThreshold(threshold=CONFIG["PREPROCESSING"]["VARIANCE_THRESHOLD"])
        self.X_train = var_threshold.fit_transform(self.X_train)
        self.X_test = var_threshold.transform(self.X_test)
        
        scaler = RobustScaler(quantile_range=CONFIG["PREPROCESSING"]["QUANTILE_RANGE"])
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        
        # Store preprocessing components
        self.preprocessing_components = {
            'var_threshold': var_threshold,
            'scaler': scaler
        }
        
        if HAS_RICH:
            phase_tree.add(f"✅ Data prepared: {self.X_train.shape}")
        else:
            print(f"✅ Data prepared: {self.X_train.shape}")
        
        # Estimate noise ceiling
        ridge = Ridge(alpha=1.0, random_state=42)
        scores = cross_val_score(ridge, self.X_train, self.y_train, cv=self.cv, scoring='r2')
        self.noise_ceiling = scores.mean() + 2 * scores.std()
        self.current_best_r2 = scores.mean()
        
        if HAS_RICH:
            phase_tree.add(f"📏 Noise ceiling estimate: {self.noise_ceiling:.4f}")
            phase_tree.add(f"🎯 Baseline R²: {self.current_best_r2:.4f}")
            console.print(phase_tree)
        else:
            print(f"📏 Noise ceiling estimate: {self.noise_ceiling:.4f}")
            print(f"🎯 Baseline R²: {self.current_best_r2:.4f}")
    
    def create_model(self, trial):
        """Create model based on trial parameters."""
        model_type = trial.suggest_categorical('model_type', ['ridge', 'elastic', 'gbr', 'rf'])
        
        if model_type == 'ridge':
            alpha = trial.suggest_float('alpha', 1e-3, 100, log=True)
            return Ridge(alpha=alpha, random_state=42)
        
        elif model_type == 'elastic':
            alpha = trial.suggest_float('alpha', 1e-3, 10, log=True)
            l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9)
            return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=2000)
        
        elif model_type == 'gbr':
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            return GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            )
        
        elif model_type == 'rf':
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 5, 20)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            return RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                n_jobs=-1
            )
    
    def objective(self, trial):
        """Optuna objective function."""
        model = self.create_model(trial)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=self.cv, scoring='r2', n_jobs=-1)
        return scores.mean()
    
    def phase_2_optimization(self):
        """Phase 2: Model optimization."""
        if HAS_RICH:
            phase_tree = Tree("🎯 Phase 2: Model Optimization")
        else:
            print(f"\n{Colors.BOLD}🎯 Phase 2: Model Optimization{Colors.END}")
        
        study = optuna.create_study(
            direction='maximize',
            pruner=CONFIG["OPTUNA"]["PRUNER_HYPEROPT"],
            sampler=CONFIG["OPTUNA"]["SAMPLER"]
        )
        
        study.optimize(self.objective, n_trials=self.max_hyperopt_trials, show_progress_bar=False)
        
        if HAS_RICH:
            phase_tree.add(f"🏆 Best R²: {study.best_value:.4f}")
            phase_tree.add(f"🔧 Best params: {study.best_params}")
            console.print(phase_tree)
        else:
            print(f"🏆 Best R²: {study.best_value:.4f}")
            print(f"🔧 Best params: {study.best_params}")
        
        # Build final model
        self.final_pipeline = self._build_final_model(study.best_params)
        self.final_pipeline.fit(self.X_train, self.y_train)
        
        # Save model
        save_model_artifacts(
            self.final_pipeline, 
            self.preprocessing_components, 
            self.dataset_num,
            model_dir=self.model_dir
        )
        
        self.current_best_r2 = study.best_value
        self.study = study
        return study.best_value, study.best_params
    
    def _build_final_model(self, params):
        """Build final model from best parameters."""
        best_params = params.copy()
        model_type = best_params.pop('model_type')
        
        if model_type == 'ridge':
            return Ridge(random_state=42, **best_params)
        elif model_type == 'elastic':
            return ElasticNet(random_state=42, max_iter=2000, **best_params)
        elif model_type == 'gbr':
            return GradientBoostingRegressor(random_state=42, **best_params)
        elif model_type == 'rf':
            return RandomForestRegressor(random_state=42, n_jobs=-1, **best_params)
    
    def phase_3_final_evaluation(self):
        """Phase 3: Final evaluation on test set."""
        if HAS_RICH:
            phase_tree = Tree("📊 Phase 3: Final Evaluation")
        else:
            print(f"\n{Colors.BOLD}📊 Phase 3: Final Evaluation{Colors.END}")
        
        y_pred = self.final_pipeline.predict(self.X_test)
        
        r2 = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
        results = {
            'test_r2': r2,
            'test_mae': mae,
            'test_rmse': rmse,
            'noise_ceiling': self.noise_ceiling,
            'cv_best_r2': self.current_best_r2,
        }
        
        # Create diagnostic plots
        create_diagnostic_plots(
            self.y_test, y_pred, 
            study=getattr(self, 'study', None),
            dataset_num=self.dataset_num,
            noise_ceiling=self.noise_ceiling,
            baseline_r2=self.current_best_r2,
            model_dir=self.model_dir
        )
        
        # Save results
        save_model_artifacts(
            self.final_pipeline,
            self.preprocessing_components,
            self.dataset_num,
            results=results,
            model_dir=self.model_dir
        )
        
        if HAS_RICH:
            phase_tree.add(f"Test R²: {r2:.4f}")
            phase_tree.add(f"Test MAE: {mae:.4f}")
            phase_tree.add(f"Test RMSE: {rmse:.4f}")
            console.print(phase_tree)
        else:
            print(f"Test R²: {r2:.4f}")
            print(f"Test MAE: {mae:.4f}")
            print(f"Test RMSE: {rmse:.4f}")

        return results


class BattleTestedOptimizer:
    """Legacy battle-tested optimizer for compatibility."""
    
    def __init__(self, dataset_num, target_r2=0.93, max_trials=40, cv_splits=5, cv_repeats=3,
                 use_iforest=False, use_lof=False):
        self.dataset_num = dataset_num
        self.target_r2 = target_r2
        self.max_trials = max_trials
        
        if HAS_RICH:
            tree = Tree(f"🚀 Battle-Tested ML Optimizer Initialized for Hold-{dataset_num}")
            tree.add(f"Target R²: {target_r2}")
            tree.add(f"Max trials: {max_trials}")
            tree.add(f"CV strategy: {cv_splits}-fold × {cv_repeats} repeats")
            console.print(tree)
        else:
            print(f"{Colors.BOLD}{Colors.CYAN}🚀 Battle-Tested ML Optimizer Initialized for Hold-{dataset_num}{Colors.END}")
            print(f"   Target R²: {target_r2}")
            print(f"   Max trials: {max_trials}")
            print(f"   CV strategy: {cv_splits}-fold × {cv_repeats} repeats")
    
    def run_optimization(self, X, y):
        """Run the battle-tested optimization pipeline."""
        optimizer = SystematicOptimizer(self.dataset_num, max_hyperopt_trials=self.max_trials)
        return optimizer.run_systematic_optimization(X, y) 