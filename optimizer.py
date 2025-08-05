"""
Optimizer Module
===============
Main optimization classes for systematic ML model optimization.
"""

import numpy as np
import pandas as pd
import time
import joblib
import sklearn.base
import json
import logging
import os
from pathlib import Path
from datetime import datetime

# Core ML libraries
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_regression
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Neural Network libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from pytorch_tabular.models import (
        AutoEncoderModel, CategoryEmbeddingModel, FTTransformerModel, TabNetModel
    )
    HAS_PYTORCH_TABULAR = True
except ImportError:
    HAS_PYTORCH_TABULAR = False

try:
    from pytorch_tabnet import TabNetRegressor
    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False

try:
    import rtdl
    HAS_RTDL = True
except ImportError:
    HAS_RTDL = False

# Models
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import (
    GradientBoostingRegressor,
    BaggingRegressor,
    AdaBoostRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

try:  # Optional third-party regressors
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:  # pragma: no cover - optional
    XGBRegressor = None
    HAS_XGBOOST = False
    
try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except Exception:  # pragma: no cover - optional
    LGBMRegressor = None
    HAS_LIGHTGBM = False

# Metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Optuna
import optuna
from optuna.pruners import MedianPruner
import optuna.visualization as optuna_viz

try:
    from .config import CONFIG, Colors
except ImportError:
    from config import CONFIG, Colors

try:
    from .transformers import KMeansOutlierTransformer, IsolationForestTransformer, LocalOutlierFactorTransformer
except ImportError:
    from transformers import KMeansOutlierTransformer, IsolationForestTransformer, LocalOutlierFactorTransformer

try:
    from .utils import (
        save_model_artifacts,
        create_diagnostic_plots,
        print_results_summary,
        console,
        Tree,
        HAS_RICH,
    )
except ImportError:
    from utils import (
        save_model_artifacts,
        create_diagnostic_plots,
        print_results_summary,
        console,
        Tree,
        HAS_RICH,
    )

import warnings
warnings.filterwarnings('ignore')


def _create_progress_callback(auto_stop_at_noise_ceiling, noise_ceiling, max_hyperopt_trials):
    """Create progress callback function for Optuna optimization.
    
    This function is defined at module level to avoid serialization issues with n_jobs > 1.
    
    Args:
        auto_stop_at_noise_ceiling: Whether to auto-stop at noise ceiling
        noise_ceiling: Target noise ceiling R² value
        max_hyperopt_trials: Maximum number of trials
        
    Returns:
        Callback function for Optuna study
    """
    def progress_callback(study, trial):
        if auto_stop_at_noise_ceiling:
            # Use lower limit for display when auto-stopping
            print(f"Trial {trial.number + 1}: R² = {trial.value:.4f} "
                  f"(Best: {study.best_value:.4f}, Target: {noise_ceiling:.4f})")
            
            # Check if we've reached the noise ceiling (within 1% tolerance)
            if study.best_value >= (noise_ceiling - 0.01):
                print(f"🎯 Reached noise ceiling! Best R² ({study.best_value:.4f}) is at noise ceiling ({noise_ceiling:.4f})")
                study.stop()
        else:
            print(f"Trial {trial.number + 1}/{max_hyperopt_trials}: R² = {trial.value:.4f} "
                  f"(Best: {study.best_value:.4f})")
    
    return progress_callback


class SystematicOptimizer:
    """Systematic ML optimizer with comprehensive preprocessing and hyperparameter optimization."""
    
    def __init__(self, dataset_num: int, max_hyperopt_trials: int = None, n_jobs: int = 3):
        self.dataset_num = dataset_num
        
        # Track if trials limit was explicitly set
        self.auto_stop_at_noise_ceiling = max_hyperopt_trials is None
        self.max_hyperopt_trials = max_hyperopt_trials or CONFIG["OPTUNA"]["HYPEROPT_TRIALS"]
        
        # If auto-stopping to noise ceiling, use a high limit but with callback
        if self.auto_stop_at_noise_ceiling:
            self.max_hyperopt_trials = 10000  # High limit, will be stopped by callback
            
        self.n_jobs = 3  # Always use 3 threads
        
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
        self.study = None
        
        # Setup logging and directories
        self.model_dir = Path(CONFIG["PATHS"]["MODEL_DIR_TEMPLATE"].format(dataset_num=dataset_num))
        self.logger = logging.getLogger(__name__)
        
        # Create Best-Params directory
        self.best_params_dir = Path("Best-Params")
        self.best_params_dir.mkdir(exist_ok=True)
        
        if HAS_RICH:
            init_tree = Tree(f"🚀 SystematicOptimizer initialized for Hold-{dataset_num}")
            if self.auto_stop_at_noise_ceiling:
                init_tree.add(f"Stop condition: Auto-stop at noise ceiling")
            else:
                init_tree.add(f"Max trials: {self.max_hyperopt_trials}")
            init_tree.add(f"CV strategy: {self.cv_splits}-fold × {self.cv_repeats} repeats")
            console.print(init_tree)
        else:
            print(f"{Colors.BOLD}{Colors.CYAN}🚀 SystematicOptimizer initialized for Hold-{dataset_num}{Colors.END}")
            if self.auto_stop_at_noise_ceiling:
                print(f"   Stop condition: Auto-stop at noise ceiling")
            else:
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
        
        # First split: train vs temp (test+holdout)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Second split: test vs holdout (from temp)
        X_test, X_holdout, y_test, y_holdout = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.X_holdout, self.y_holdout = X_holdout, y_holdout
        print(f"Training set: X{self.X_train.shape}, y({len(self.y_train)},)")
        print(f"Test set: X{self.X_test.shape}, y({len(self.y_test)},)")
        print(f"Holdout set: X{self.X_holdout.shape}, y({len(self.y_holdout)},)")
        # Save holdout set to best_model folder
        holdout_dir = self.model_dir
        holdout_dir.mkdir(exist_ok=True, parents=True)
        pd.DataFrame(self.X_holdout).to_csv(holdout_dir / "holdout_features.csv", index=False)
        pd.DataFrame(self.y_holdout).to_csv(holdout_dir / "holdout_targets.csv", index=False)
        
        # No preprocessing here - let objective function handle it dynamically
        
        if HAS_RICH:
            phase_tree.add(f"✅ Data prepared: {self.X_train.shape}")
        else:
            print(f"✅ Data prepared: {self.X_train.shape}")
        
        # Estimate noise ceiling
        print(f"Estimating noise ceiling with {self.cv_splits}-fold x {self.cv_repeats} repeats CV...")
        ridge = Ridge(alpha=1.0, random_state=42)
        scores = cross_val_score(ridge, self.X_train, self.y_train, cv=self.cv, scoring='r2', n_jobs=3)
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
        """Create model based on trial parameters with efficient parameter suggestion."""
        # Get available models
        available_models = ['ridge', 'elastic', 'lasso', 'dt', 'extra', 'gbr', 'rf', 'svr', 'ridge_cv', 'bagging', 'ada', 'mlp']
        
        # Add GPU-enabled models if available
        if HAS_XGBOOST:
            available_models.append('xgb')
        if HAS_LIGHTGBM:
            available_models.append('lgbm')
        
        model_type = trial.suggest_categorical('model_type', available_models)
        
        # Only suggest parameters relevant to the chosen model
        if model_type == 'ridge':
            alpha = trial.suggest_float('ridge_alpha', 1e-3, 100, log=True)
            return Ridge(alpha=alpha, random_state=42)
        
        elif model_type == 'elastic':
            alpha = trial.suggest_float('elastic_alpha', 1e-3, 10, log=True)
            l1_ratio = trial.suggest_float('elastic_l1_ratio', 0.1, 0.9)
            return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=2000)

        elif model_type == 'lasso':
            alpha = trial.suggest_float('lasso_alpha', 1e-3, 10, log=True)
            return Lasso(alpha=alpha, random_state=42)

        elif model_type == 'dt':
            max_depth = trial.suggest_int('dt_max_depth', 3, 20)
            min_samples_split = trial.suggest_int('dt_min_samples_split', 2, 10)
            return DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
            )

        elif model_type == 'extra':
            n_estimators = trial.suggest_int('extra_n_estimators', 50, 300)
            max_depth = trial.suggest_int('extra_max_depth', 3, 20)
            return ExtraTreesRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=self.n_jobs,
            )
        
        elif model_type == 'gbr':
            n_estimators = trial.suggest_int('gbr_n_estimators', 50, 300)
            learning_rate = trial.suggest_float('gbr_learning_rate', 0.01, 0.3)
            max_depth = trial.suggest_int('gbr_max_depth', 3, 10)
            return GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            )
        
        elif model_type == 'rf':
            n_estimators = trial.suggest_int('rf_n_estimators', 50, 300)
            max_depth = trial.suggest_int('rf_max_depth', 5, 20)
            min_samples_split = trial.suggest_int('rf_min_samples_split', 2, 10)
            return RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                n_jobs=self.n_jobs
            )

        elif model_type == 'xgb':
            n_estimators = trial.suggest_int('xgb_n_estimators', 50, 300)
            learning_rate = trial.suggest_float('xgb_learning_rate', 0.01, 0.3)
            max_depth = trial.suggest_int('xgb_max_depth', 3, 10)
            subsample = trial.suggest_float('xgb_subsample', 0.6, 1.0)
            colsample_bytree = trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0)
            # GPU support for XGBoost
            tree_method = trial.suggest_categorical('xgb_tree_method', ['hist', 'gpu_hist'])
            return XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                tree_method=tree_method,
                random_state=42,
                n_jobs=self.n_jobs,
                verbosity=0,
            )

        elif model_type == 'lgbm':
            n_estimators = trial.suggest_int('lgbm_n_estimators', 50, 300)
            learning_rate = trial.suggest_float('lgbm_learning_rate', 0.01, 0.3)
            max_depth = trial.suggest_int('lgbm_max_depth', 3, 10)
            num_leaves = trial.suggest_int('lgbm_num_leaves', 10, 300)
            subsample = trial.suggest_float('lgbm_subsample', 0.6, 1.0)
            colsample_bytree = trial.suggest_float('lgbm_colsample_bytree', 0.6, 1.0)
            # GPU support for LightGBM
            device_type = trial.suggest_categorical('lgbm_device_type', ['cpu', 'gpu'])
            return LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                num_leaves=num_leaves,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                device_type=device_type,
                random_state=42,
                n_jobs=self.n_jobs,
                verbosity=-1,
            )
        
        elif model_type == 'svr':
            C = trial.suggest_float('svr_C', *CONFIG["MODEL_PARAMS"]["SVR"]["C_range"], 
                                   log=CONFIG["MODEL_PARAMS"]["SVR"]["C_log"])
            gamma = trial.suggest_float('svr_gamma', *CONFIG["MODEL_PARAMS"]["SVR"]["gamma_range"], 
                                       log=CONFIG["MODEL_PARAMS"]["SVR"]["gamma_log"])
            return SVR(C=C, gamma=gamma)
        
        elif model_type == 'ridge_cv':
            return RidgeCV(alphas=np.logspace(-3, 2, 50))
        
        elif model_type == 'bagging':
            n_estimators = trial.suggest_int('bagging_n_estimators', 10, 100)
            return BaggingRegressor(n_estimators=n_estimators, random_state=42, n_jobs=self.n_jobs)
        
        elif model_type == 'ada':
            n_estimators = trial.suggest_int('ada_n_estimators', 50, 200)
            learning_rate = trial.suggest_float('ada_learning_rate', 0.01, 2.0)
            return AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
        
        elif model_type == 'mlp':
            hidden_layer_sizes = trial.suggest_categorical('mlp_hidden_layer_sizes', 
                                                          [(50,), (100,), (50, 50), (100, 50)])
            alpha = trial.suggest_float('mlp_alpha', 1e-5, 1e-1, log=True)
            return MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, max_iter=1000, random_state=42)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def objective(self, trial):
        """Optuna objective function with aggressive optimization for large datasets."""
        try:
            # AGGRESSIVE FEATURE REDUCTION FIRST - this is the key fix
            X_reduced = self.X_train
            y_reduced = self.y_train
            
            # For massive feature sets, apply variance threshold immediately
            if X_reduced.shape[1] > 5000:
                var_threshold = VarianceThreshold(threshold=0.1)
                X_reduced = var_threshold.fit_transform(X_reduced)
                print(f"Variance threshold: {self.X_train.shape[1]} → {X_reduced.shape[1]} features")
            
            # If still too many features, use fast correlation-based selection
            if X_reduced.shape[1] > 2000:
                k_select = min(2000, X_reduced.shape[1] // 2)
                # Use f_regression instead of mutual_info_regression (much faster)
                from sklearn.feature_selection import f_regression
                selector = SelectKBest(f_regression, k=k_select)
                X_reduced = selector.fit_transform(X_reduced, y_reduced)
                print(f"SelectKBest (f_regression): → {X_reduced.shape[1]} features")
            
            # Suggest preprocessing options (simplified)
            use_feature_selection = trial.suggest_categorical('use_feature_selection', [True, False])
            scaler_type = trial.suggest_categorical('scaler_type', ['robust', 'standard'])
            
            # Create model
            model = self.create_model(trial)
            
            # SIMPLIFIED CV - use 3-fold, 1 repeat instead of 15 iterations
            simple_cv = RepeatedKFold(n_splits=3, n_repeats=1, random_state=42)
            fold_scores = []
            
            for train_idx, val_idx in simple_cv.split(X_reduced, y_reduced):
                X_fold_train, X_fold_val = X_reduced[train_idx], X_reduced[val_idx]
                y_fold_train, y_fold_val = y_reduced[train_idx], y_reduced[val_idx]
            
                # SIMPLIFIED preprocessing pipeline
                preprocessing_steps = []
                
                # Imputation (if needed)
                if np.isnan(X_fold_train).any():
                    preprocessing_steps.append(('imputer', SimpleImputer(strategy='median')))
                
                # Scaling
                if scaler_type == 'standard':
                    preprocessing_steps.append(('scaler', StandardScaler()))
                else:  # robust (default)
                    preprocessing_steps.append(('scaler', RobustScaler()))
                
                # Optional final feature selection on reduced set
                if use_feature_selection and X_fold_train.shape[1] > 500:
                    k_features = trial.suggest_int('k_features', 100, min(500, X_fold_train.shape[1]))
                    from sklearn.feature_selection import f_regression
                    preprocessing_steps.append(('feature_selection', 
                                              SelectKBest(score_func=f_regression, k=k_features)))
                
                # Create and apply pipeline
                if preprocessing_steps:
                    pipeline = Pipeline(preprocessing_steps + [('model', model)])
                    pipeline.fit(X_fold_train, y_fold_train)
                    y_pred = pipeline.predict(X_fold_val)
                else:
                    model.fit(X_fold_train, y_fold_train)
                    y_pred = model.predict(X_fold_val)
                
                # Calculate fold score
                fold_score = r2_score(y_fold_val, y_pred)
                fold_scores.append(fold_score)
                
                # Early stopping if fold is terrible
                if fold_score < -5.0:
                    return -999.0
            
            # Return mean CV score
            mean_score = np.mean(fold_scores)
            return mean_score if not np.isnan(mean_score) else -999.0
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return -999.0
    
    def phase_2_optimization(self):
        """Phase 2: Model optimization with enhanced visualizations and parameter saving."""
        if HAS_RICH:
            phase_tree = Tree("🎯 Phase 2: Model Optimization")
        else:
            print(f"\n{Colors.BOLD}🎯 Phase 2: Model Optimization{Colors.END}")
        
        # Ensure Optuna logging is captured
        optuna_logger = logging.getLogger('optuna')
        optuna_logger.setLevel(logging.DEBUG)
        
        study = optuna.create_study(
            direction='maximize',
            pruner=CONFIG["OPTUNA"]["PRUNER_HYPEROPT"],
            sampler=CONFIG["OPTUNA"]["SAMPLER"]
        )
        
        # Create progress callback using module-level factory function
        progress_callback = _create_progress_callback(
            self.auto_stop_at_noise_ceiling, 
            self.noise_ceiling, 
            self.max_hyperopt_trials
        )
        
        if self.auto_stop_at_noise_ceiling:
            print(f"Starting optimization until noise ceiling ({self.noise_ceiling:.4f}) is reached...")
        else:
            print(f"Starting optimization with {self.max_hyperopt_trials} trials...")
            
        study.optimize(self.objective, n_trials=self.max_hyperopt_trials, show_progress_bar=False, 
                      n_jobs=1, callbacks=[progress_callback])
        
        if HAS_RICH:
            phase_tree.add(f"🏆 Best R²: {study.best_value:.4f}")
            phase_tree.add(f"🔧 Best params: {study.best_params}")
            console.print(phase_tree)
        else:
            print(f"🏆 Best R²: {study.best_value:.4f}")
            print(f"🔧 Best params: {study.best_params}")
        
        # Save best parameters to JSON (without overwriting)
        self._save_best_parameters(study.best_params, study.best_value)
        
        # Create Optuna visualizations
        self._create_optuna_visualizations(study)
        
        # Build final model
        print(f"Building final model with best parameters...")
        self.final_pipeline = self._build_final_model(study.best_params)
        
        # Handle outlier detection outside pipeline to maintain X/y sync
        X_train_final = self.X_train.copy()
        y_train_final = self.y_train.copy()
        
        best_params = study.best_params
        if best_params.get('use_outlier_detection', False) and best_params.get('outlier_method'):
            outlier_method = best_params['outlier_method']
            
            if outlier_method == 'isolation':
                outlier_detector = IsolationForestTransformer()
            elif outlier_method == 'lof':
                outlier_detector = LocalOutlierFactorTransformer()
            else:
                outlier_detector = KMeansOutlierTransformer()
            
            # Fit outlier detector and get mask
            outlier_detector.fit(X_train_final)
            outlier_mask = outlier_detector.mask_
            
            # Apply mask to both X and y
            X_train_final = X_train_final[outlier_mask]
            y_train_final = y_train_final[outlier_mask]
            
            print(f"Outlier detection removed {(~outlier_mask).sum()} samples")
        
        print(f"Training final model on {X_train_final.shape[0]} samples...")
        self.final_pipeline.fit(X_train_final, y_train_final)
        
        # Save model
        save_model_artifacts(
            self.final_pipeline, 
            {},  # Empty dict since preprocessing is now part of final_pipeline
            self.dataset_num,
            model_dir=self.model_dir
        )
        
        self.current_best_r2 = study.best_value
        self.study = study
        return study.best_value, study.best_params
    
    def _save_best_parameters(self, best_params, best_score):
        """Save best parameters to JSON file without overwriting."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"best_params_hold{self.dataset_num}_{timestamp}.json"
        filepath = self.best_params_dir / filename
        
        params_data = {
            "dataset_num": self.dataset_num,
            "timestamp": timestamp,
            "best_score": best_score,
            "best_parameters": best_params,
            "cv_config": {
                "splits": self.cv_splits,
                "repeats": self.cv_repeats
            },
            "optimization_config": {
                "max_trials": self.max_hyperopt_trials,
                "n_jobs": self.n_jobs
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(params_data, f, indent=2)
        
        print(f"📄 Best parameters saved to: {filepath}")
    
    def _create_optuna_visualizations(self, study):
        """Create Optuna visualizations for parameter importance and slices."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            
            viz_dir = self.model_dir / "optuna_visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # Parameter importance plot
            try:
                fig = optuna_viz.plot_param_importances(study)
                fig.write_html(str(viz_dir / "param_importances.html"))
                print(f"📊 Parameter importance plot saved to: {viz_dir / 'param_importances.html'}")
            except Exception as e:
                print(f"Warning: Could not create parameter importance plot: {e}")
            
            # Slice plot for each parameter
            try:
                fig = optuna_viz.plot_slice(study)
                fig.write_html(str(viz_dir / "param_slices.html"))
                print(f"📊 Parameter slice plot saved to: {viz_dir / 'param_slices.html'}")
            except Exception as e:
                print(f"Warning: Could not create parameter slice plot: {e}")
            
            # Optimization history
            try:
                fig = optuna_viz.plot_optimization_history(study)
                fig.write_html(str(viz_dir / "optimization_history.html"))
                print(f"📊 Optimization history plot saved to: {viz_dir / 'optimization_history.html'}")
            except Exception as e:
                print(f"Warning: Could not create optimization history plot: {e}")
            
        except ImportError:
            print("Warning: Optuna visualizations require plotly. Install with: pip install plotly")
    
    def _build_final_model(self, params):
        """Build final pipeline from best parameters with improved parameter handling."""
        best_params = params.copy()
        
        # Extract preprocessing parameters
        use_feature_selection = best_params.pop('use_feature_selection', False)
        use_pca = best_params.pop('use_pca', False)
        use_outlier_detection = best_params.pop('use_outlier_detection', False)
        scaler_type = best_params.pop('scaler_type', 'robust')
        outlier_method = best_params.pop('outlier_method', None)
        k_features = best_params.pop('k_features', None)
        use_incremental_pca = best_params.pop('use_incremental_pca', False)
        # Handle both integer and float PCA components
        pca_components_int = best_params.pop('pca_components_int', None)
        pca_components_float = best_params.pop('pca_components_float', None)
        pca_components = pca_components_int if pca_components_int is not None else pca_components_float
        
        # Build preprocessing pipeline
        preprocessing_steps = []
        
        # Imputation (if needed)
        if np.isnan(self.X_train).any():
            preprocessing_steps.append(('imputer', SimpleImputer(strategy='median')))
        
        # Scaling
        if scaler_type == 'standard':
            preprocessing_steps.append(('scaler', StandardScaler()))
        elif scaler_type == 'minmax':
            preprocessing_steps.append(('scaler', MinMaxScaler()))
        else:  # robust (default)
            preprocessing_steps.append(('scaler', RobustScaler()))
        
        # Note: Outlier detection will be handled outside the pipeline to avoid X/y shape mismatch
        
        # Feature selection
        if use_feature_selection and k_features:
            preprocessing_steps.append(('feature_selection', 
                                      SelectKBest(score_func=mutual_info_regression, k=k_features)))
        
        # Dimensionality reduction
        if use_pca and not use_feature_selection and pca_components:
            if use_incremental_pca:
                preprocessing_steps.append(('pca', IncrementalPCA(n_components=pca_components)))
            else:
                preprocessing_steps.append(('pca', PCA(n_components=pca_components)))
        
        # Create model from remaining parameters
        model_type = best_params.pop('model_type')
        
        # Filter parameters by model type prefix to get only relevant params
        model_params = {}
        prefix = f"{model_type}_"
        for key, value in best_params.items():
            if key.startswith(prefix):
                # Remove the prefix to get the actual parameter name
                param_name = key[len(prefix):]
                model_params[param_name] = value
        
        if model_type == 'ridge':
            model = Ridge(random_state=42, **model_params)
        elif model_type == 'elastic':
            model = ElasticNet(random_state=42, max_iter=2000, **model_params)
        elif model_type == 'lasso':
            model = Lasso(random_state=42, **model_params)
        elif model_type == 'dt':
            model = DecisionTreeRegressor(random_state=42, **model_params)
        elif model_type == 'extra':
            model = ExtraTreesRegressor(random_state=42, n_jobs=self.n_jobs, **model_params)
        elif model_type == 'gbr':
            model = GradientBoostingRegressor(random_state=42, **model_params)
        elif model_type == 'rf':
            model = RandomForestRegressor(random_state=42, n_jobs=self.n_jobs, **model_params)
        elif model_type == 'xgb' and HAS_XGBOOST:
            model = XGBRegressor(random_state=42, n_jobs=self.n_jobs, verbosity=0, **model_params)
        elif model_type == 'lgbm' and HAS_LIGHTGBM:
            model = LGBMRegressor(random_state=42, n_jobs=self.n_jobs, verbosity=-1, **model_params)
        elif model_type == 'svr':
            model = SVR(**model_params)
        elif model_type == 'ridge_cv':
            model = RidgeCV(alphas=np.logspace(-3, 2, 50))
        elif model_type == 'bagging':
            model = BaggingRegressor(random_state=42, n_jobs=self.n_jobs, **model_params)
        elif model_type == 'ada':
            model = AdaBoostRegressor(random_state=42, **model_params)
        elif model_type == 'mlp':
            model = MLPRegressor(random_state=42, max_iter=1000, **model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create full pipeline
        if preprocessing_steps:
            return Pipeline(preprocessing_steps + [('model', model)])
        else:
            return model
    
    def phase_3_final_evaluation(self):
        """Phase 3: Final evaluation on test set."""
        if HAS_RICH:
            phase_tree = Tree("📊 Phase 3: Final Evaluation")
        else:
            print(f"\n{Colors.BOLD}📊 Phase 3: Final Evaluation{Colors.END}")
        
        print(f"Evaluating final model on {self.X_test.shape[0]} test samples...")
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
        print(f"Creating diagnostic plots and saving results...")
        create_diagnostic_plots(
            self.y_test, y_pred, 
            study=getattr(self, 'study', None),
            dataset_num=self.dataset_num,
            noise_ceiling=self.noise_ceiling,
            baseline_r2=self.current_best_r2,
            model_dir=self.model_dir
        )
        
        # Save results (preprocessing is now part of the pipeline)
        save_model_artifacts(
            self.final_pipeline,
            {},  # Empty dict since preprocessing is now part of final_pipeline
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
                 use_iforest=False, use_lof=False, n_jobs=3):
        self.dataset_num = dataset_num
        self.target_r2 = target_r2
        # Handle None for auto-stopping behavior
        self.max_trials = max_trials if max_trials is not None else CONFIG["OPTUNA"]["MAX_TRIALS_BATTLE_TESTED"]
        self.auto_stop_at_noise_ceiling = max_trials is None
        self.n_jobs = 3
        
        if HAS_RICH:
            tree = Tree(f"🚀 Battle-Tested ML Optimizer Initialized for Hold-{dataset_num}")
            tree.add(f"Target R²: {target_r2}")
            if self.auto_stop_at_noise_ceiling:
                tree.add(f"Stop condition: Auto-stop at noise ceiling")
            else:
                tree.add(f"Max trials: {self.max_trials}")
            tree.add(f"CV strategy: {cv_splits}-fold × {cv_repeats} repeats")
            console.print(tree)
        else:
            print(f"{Colors.BOLD}{Colors.CYAN}🚀 Battle-Tested ML Optimizer Initialized for Hold-{dataset_num}{Colors.END}")
            print(f"   Target R²: {target_r2}")
            if self.auto_stop_at_noise_ceiling:
                print(f"   Stop condition: Auto-stop at noise ceiling")
            else:
                print(f"   Max trials: {self.max_trials}")
            print(f"   CV strategy: {cv_splits}-fold × {cv_repeats} repeats")
    
    def run_optimization(self, X, y):
        """Run the battle-tested optimization pipeline."""
        # Pass None for auto-stopping, otherwise pass the actual max_trials
        max_trials_param = None if self.auto_stop_at_noise_ceiling else self.max_trials
        optimizer = SystematicOptimizer(self.dataset_num, max_hyperopt_trials=max_trials_param, n_jobs=self.n_jobs)
        return optimizer.run_systematic_optimization(X, y)


class NeuralNetworkOptimizer:
    """Neural Network only optimizer with comprehensive deep learning models."""
    
    def __init__(self, dataset_num: int, max_hyperopt_trials: int = None, n_jobs: int = 3, kbest_ratio: str = None):
        self.dataset_num = dataset_num
        self.max_hyperopt_trials = max_hyperopt_trials
        self.n_jobs = n_jobs
        self.kbest_ratio = kbest_ratio
        
        # Track if trials limit was explicitly set
        self.auto_stop_at_noise_ceiling = max_hyperopt_trials is None
        # Neural networks need more trials by default - use 200 minimum instead of 40
        default_nn_trials = max(200, CONFIG["OPTUNA"]["HYPEROPT_TRIALS"] * 5)
        self.max_hyperopt_trials = max_hyperopt_trials or default_nn_trials
        
        # If auto-stopping to noise ceiling, use a high limit but with callback
        if self.auto_stop_at_noise_ceiling:
            self.max_hyperopt_trials = 10000  # High limit, will be stopped by callback
            
        self.n_jobs = n_jobs
        
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
        self.study = None
        
        # Setup logging and directories
        self.model_dir = Path(CONFIG["PATHS"]["MODEL_DIR_TEMPLATE"].format(dataset_num=dataset_num))
        self.logger = logging.getLogger(__name__)
        
        # Create Best-Params directory
        self.best_params_dir = Path("Best-Params")
        self.best_params_dir.mkdir(exist_ok=True)
        
        # Check available neural network libraries
        self.available_libraries = self._check_nn_libraries()
        
        if HAS_RICH:
            init_tree = Tree(f"🧠 NeuralNetworkOptimizer initialized for Hold-{dataset_num}")
            if self.auto_stop_at_noise_ceiling:
                init_tree.add(f"Stop condition: Auto-stop at noise ceiling")
            else:
                init_tree.add(f"Max trials: {self.max_hyperopt_trials}")
            init_tree.add(f"CV strategy: {self.cv_splits}-fold × {self.cv_repeats} repeats")
            init_tree.add(f"Available NN libraries: {len(self.available_libraries)}")
            console.print(init_tree)
        else:
            print(f"{Colors.BOLD}{Colors.CYAN}🧠 NeuralNetworkOptimizer initialized for Hold-{dataset_num}{Colors.END}")
            if self.auto_stop_at_noise_ceiling:
                print(f"   Stop condition: Auto-stop at noise ceiling")
            else:
                print(f"   Max trials: {self.max_hyperopt_trials}")
            print(f"   CV strategy: {self.cv_splits}-fold × {self.cv_repeats} repeats")
            print(f"   Available NN libraries: {len(self.available_libraries)}")
    
    def _check_nn_libraries(self):
        """Check which neural network libraries are available."""
        available = []
        if HAS_TORCH:
            available.append("torch")
        if HAS_PYTORCH_TABULAR:
            available.append("pytorch_tabular")
        if HAS_TABNET:
            available.append("tabnet")
        if HAS_RTDL:
            available.append("rtdl")
        
        print(f"{Colors.GREEN}✅ Available NN libraries: {', '.join(available) if available else 'None'}{Colors.END}")
        return available
    
    def _get_gpu_device(self):
        """Get AMD GPU device with ROCm support - NO NVIDIA BULLSHIT!"""
        import os
        # Set ROCm environment variables for AMD GPU
        os.environ['HIP_VISIBLE_DEVICES'] = '0'
        os.environ['ROCR_VISIBLE_DEVICES'] = '0'
        
        # Check if PyTorch can detect the AMD GPU via ROCm (appears as CUDA device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device = torch.device("cuda:0")
            try:
                # Test if we can actually use the GPU
                test_tensor = torch.tensor([1.0], device=device)
                gpu_name = torch.cuda.get_device_name(0)
                if "AMD" in gpu_name or "Radeon" in gpu_name:
                    print(f"🔥 AMD ROCm GPU detected: {gpu_name}")
                    return device
                else:
                    raise RuntimeError(f"NVIDIA GPU detected: {gpu_name}. --NN flag is ROCm ONLY, FUCK NVIDIA!")
            except Exception as e:
                raise RuntimeError(f"GPU test failed: {e}")
        
        # No AMD GPU available
        raise RuntimeError("--NN flag requires AMD GPU with ROCm support. No compatible GPU detected.")
    
    def _get_gpu_name(self, device):
        """Get AMD GPU name - ROCm backend."""
        if device.type == 'cuda':
            try:
                gpu_name = torch.cuda.get_device_name(device.index or 0)
                memory_gb = torch.cuda.get_device_properties(device.index or 0).total_memory / (1024**3)
                return f"{gpu_name} ({memory_gb:.1f}GB ROCm)"
            except:
                return "AMD GPU (ROCm backend)"
        else:
            return "CPU (ERROR - should never reach here in NN mode)"
    
    def create_neural_network_model(self, trial):
        """Create neural network model based on trial parameters."""
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for Neural Network models but not available")
        
        # FORCE GPU ONLY - NN mode is GPU-only (NVIDIA CUDA or AMD ROCm)
        device = self._get_gpu_device()
        if device.type == 'cpu':
            raise RuntimeError("--NN flag requires GPU acceleration. No GPU detected. Neural networks need GPU-only training.")
        
        print(f"🚀 Using GPU: {device} - {self._get_gpu_name(device)}")
        
        # Available neural network models - COMPREHENSIVE LIST
        available_models = []
        
        # Basic PyTorch models (always available)
        available_models.extend(['simple_mlp', 'deep_mlp', 'wide_deep', 'resnet_custom'])
        
        # TabNet models
        if HAS_TABNET:
            available_models.append('tabnet')
        
        # RTDL models
        if HAS_RTDL:
            available_models.extend(['ft_transformer', 'resnet_rtdl', 'ft_mlp'])
        
        # PyTorch Tabular models
        if HAS_PYTORCH_TABULAR:
            available_models.extend(['autoencoder', 'category_embedding', 'ft_transformer_pt'])
        
        # Feature Interaction Models
        try:
            import deepctr_torch
            available_models.extend(['deepfm', 'dcn'])
        except ImportError:
            pass
            
        # Wide & Deep variants
        try:
            import pytorch_widedeep
            available_models.extend(['wide_deep_advanced', 'autoint'])
        except ImportError:
            pass
        
        # Tree-inspired models
        try:
            import dndt
            available_models.append('dndt')
        except ImportError:
            pass
            
        # Mixture of Experts
        try:
            import pytorch_moe
            available_models.append('moe')
        except ImportError:
            pass
        
        # Self-supervised models
        try:
            import vime
            available_models.append('vime')
        except ImportError:
            pass
            
        try:
            import scarf
            available_models.append('scarf')
        except ImportError:
            pass
        
        # Bayesian models
        try:
            import blitz
            available_models.append('bayesian_nn')
        except ImportError:
            pass
        
        # Transformer variants
        try:
            import tabular_transformers
            available_models.extend(['saint', 'tab_transformer'])
        except ImportError:
            pass
        
        model_type = trial.suggest_categorical('nn_model_type', available_models)
        
        # Basic models
        if model_type == 'simple_mlp':
            return self._create_simple_mlp(trial, device)
        elif model_type == 'deep_mlp':
            return self._create_deep_mlp(trial, device)
        elif model_type == 'wide_deep':
            return self._create_wide_deep(trial, device)
        elif model_type == 'resnet_custom':
            return self._create_resnet_custom(trial, device)
            
        # TabNet models
        elif model_type == 'tabnet':
            return self._create_tabnet(trial)
            
        # RTDL models
        elif model_type == 'ft_transformer':
            return self._create_ft_transformer(trial, device)
        elif model_type == 'resnet_rtdl':
            return self._create_resnet_rtdl(trial, device)
        elif model_type == 'ft_mlp':
            return self._create_ft_mlp(trial, device)
            
        # PyTorch Tabular models
        elif model_type == 'autoencoder':
            return self._create_autoencoder(trial)
        elif model_type == 'category_embedding':
            return self._create_category_embedding(trial)
        elif model_type == 'ft_transformer_pt':
            return self._create_ft_transformer_pt(trial)
            
        # Feature Interaction models
        elif model_type == 'deepfm':
            return self._create_deepfm(trial, device)
        elif model_type == 'dcn':
            return self._create_dcn(trial, device)
            
        # Advanced Wide & Deep
        elif model_type == 'wide_deep_advanced':
            return self._create_wide_deep_advanced(trial, device)
        elif model_type == 'autoint':
            return self._create_autoint(trial, device)
            
        # Tree-inspired models
        elif model_type == 'dndt':
            return self._create_dndt(trial, device)
            
        # Mixture of Experts
        elif model_type == 'moe':
            return self._create_moe(trial, device)
            
        # Self-supervised models
        elif model_type == 'vime':
            return self._create_vime(trial, device)
        elif model_type == 'scarf':
            return self._create_scarf(trial, device)
            
        # Bayesian models
        elif model_type == 'bayesian_nn':
            return self._create_bayesian_nn(trial, device)
            
        # Transformer variants
        elif model_type == 'saint':
            return self._create_saint(trial, device)
        elif model_type == 'tab_transformer':
            return self._create_tab_transformer(trial, device)
            
        else:
            raise ValueError(f"Unknown or unavailable neural network model type: {model_type}")
    
    def _create_simple_mlp(self, trial, device):
        """Create a simple MLP regressor."""
        class SimpleMLP(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, dropout):
                super().__init__()
                layers = []
                current_dim = input_dim
                
                for i in range(num_layers):
                    layers.append(nn.Linear(current_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    current_dim = hidden_dim
                
                layers.append(nn.Linear(current_dim, 1))
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x).squeeze(-1)
        
        hidden_dim = trial.suggest_int('mlp_hidden_dim', 64, 512, step=64)
        num_layers = trial.suggest_int('mlp_num_layers', 2, 6)
        dropout = trial.suggest_float('mlp_dropout', 0.0, 0.5)
        lr = trial.suggest_float('mlp_lr', 1e-5, 1e-2, log=True)
        
        return {
            'model_class': SimpleMLP,
            'model_kwargs': {
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'dropout': dropout
            },
            'lr': lr,
            'device': device
        }
    
    def _create_deep_mlp(self, trial, device):
        """Create a deep MLP with residual connections."""
        class DeepMLP(nn.Module):
            def __init__(self, input_dim, hidden_dims, dropout, use_residual=True):
                super().__init__()
                self.use_residual = use_residual
                layers = []
                current_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(current_dim, hidden_dim))
                    layers.append(nn.BatchNorm1d(hidden_dim))
                    layers.append(nn.ReLU())
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    current_dim = hidden_dim
                
                self.backbone = nn.Sequential(*layers)
                self.head = nn.Linear(current_dim, 1)
                
                # Residual connection if dimensions match
                self.residual = nn.Linear(input_dim, current_dim) if input_dim != current_dim else nn.Identity()
            
            def forward(self, x):
                backbone_out = self.backbone(x)
                if self.use_residual:
                    residual_out = self.residual(x)
                    backbone_out = backbone_out + residual_out
                return self.head(backbone_out).squeeze(-1)
        
        num_layers = trial.suggest_int('deep_mlp_num_layers', 3, 8)
        base_dim = trial.suggest_int('deep_mlp_base_dim', 128, 512, step=64)
        hidden_dims = [base_dim * (2 ** i) for i in range(num_layers)]
        hidden_dims = [min(dim, 1024) for dim in hidden_dims]  # Cap at 1024
        dropout = trial.suggest_float('deep_mlp_dropout', 0.1, 0.5)
        lr = trial.suggest_float('deep_mlp_lr', 1e-5, 1e-2, log=True)
        
        return {
            'model_class': DeepMLP,
            'model_kwargs': {
                'hidden_dims': hidden_dims,
                'dropout': dropout,
                'use_residual': True
            },
            'lr': lr,
            'device': device
        }
    
    def _create_wide_deep(self, trial, device):
        """Create Wide & Deep model."""
        class WideDeepModel(nn.Module):
            def __init__(self, input_dim, deep_dims, wide_dim_ratio=0.5):
                super().__init__()
                # Split features for wide and deep parts
                self.wide_dim = int(input_dim * wide_dim_ratio)
                self.deep_dim = input_dim - self.wide_dim
                
                # Wide part (linear)
                self.wide = nn.Linear(self.wide_dim, 1)
                
                # Deep part (MLP)
                deep_layers = []
                current_dim = self.deep_dim
                for hidden_dim in deep_dims:
                    deep_layers.extend([
                        nn.Linear(current_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2)
                    ])
                    current_dim = hidden_dim
                deep_layers.append(nn.Linear(current_dim, 1))
                self.deep = nn.Sequential(*deep_layers)
            
            def forward(self, x):
                wide_features = x[:, :self.wide_dim]
                deep_features = x[:, self.wide_dim:]
                
                wide_out = self.wide(wide_features)
                deep_out = self.deep(deep_features)
                
                return (wide_out + deep_out).squeeze(-1)
        
        num_deep_layers = trial.suggest_int('wd_deep_layers', 2, 5)
        deep_base_dim = trial.suggest_int('wd_deep_base_dim', 128, 512, step=64)
        deep_dims = [deep_base_dim // (2 ** i) for i in range(num_deep_layers)]
        deep_dims = [max(dim, 32) for dim in deep_dims]  # Minimum 32
        wide_ratio = trial.suggest_float('wd_wide_ratio', 0.2, 0.8)
        lr = trial.suggest_float('wd_lr', 1e-5, 1e-2, log=True)
        
        return {
            'model_class': WideDeepModel,
            'model_kwargs': {
                'deep_dims': deep_dims,
                'wide_dim_ratio': wide_ratio
            },
            'lr': lr,
            'device': device
        }
    
    def _create_tabnet(self, trial):
        """Create TabNet model."""
        n_d = trial.suggest_int('tabnet_n_d', 8, 64)
        n_a = trial.suggest_int('tabnet_n_a', 8, 64)
        n_steps = trial.suggest_int('tabnet_n_steps', 3, 10)
        gamma = trial.suggest_float('tabnet_gamma', 1.0, 2.0)
        lambda_sparse = trial.suggest_float('tabnet_lambda_sparse', 1e-6, 1e-3, log=True)
        lr = trial.suggest_float('tabnet_lr', 1e-4, 1e-2, log=True)
        
        return {
            'model_type': 'tabnet',
            'model_kwargs': {
                'n_d': n_d,
                'n_a': n_a,
                'n_steps': n_steps,
                'gamma': gamma,
                'lambda_sparse': lambda_sparse,
                'optimizer_fn': torch.optim.Adam,
                'optimizer_params': {'lr': lr},
                'scheduler_params': {'step_size': 50, 'gamma': 0.9},
                'scheduler_fn': torch.optim.lr_scheduler.StepLR,
                'mask_type': 'sparsemax'
            }
        }
    
    def _create_ft_transformer(self, trial, device):
        """Create FT-Transformer model using rtdl."""
        d_token = trial.suggest_int('ft_d_token', 32, 256, step=32)
        n_blocks = trial.suggest_int('ft_n_blocks', 2, 6)
        attention_dropout = trial.suggest_float('ft_attention_dropout', 0.0, 0.3)
        ffn_dropout = trial.suggest_float('ft_ffn_dropout', 0.0, 0.3)
        lr = trial.suggest_float('ft_lr', 1e-5, 1e-2, log=True)
        
        return {
            'model_type': 'ft_transformer',
            'model_kwargs': {
                'd_token': d_token,
                'n_blocks': n_blocks,
                'attention_dropout': attention_dropout,
                'ffn_dropout': ffn_dropout,
                'd_out': 1
            },
            'lr': lr,
            'device': device
        }
    
    def _create_resnet(self, trial, device):
        """Create ResNet model using rtdl."""
        d_main = trial.suggest_int('resnet_d_main', 64, 512, step=64)
        d_hidden = trial.suggest_int('resnet_d_hidden', 64, 512, step=64)
        n_blocks = trial.suggest_int('resnet_n_blocks', 2, 8)
        dropout = trial.suggest_float('resnet_dropout', 0.0, 0.3)
        lr = trial.suggest_float('resnet_lr', 1e-5, 1e-2, log=True)
        
        return {
            'model_type': 'resnet',
            'model_kwargs': {
                'd_main': d_main,
                'd_hidden': d_hidden,
                'n_blocks': n_blocks,
                'dropout': dropout,
                'd_out': 1
            },
            'lr': lr,
            'device': device
        }
    
    def _create_autoencoder(self, trial):
        """Create Autoencoder model using pytorch_tabular."""
        return {
            'model_type': 'autoencoder',
            'model_kwargs': {
                'encoder_config': {
                    'hidden_dim': trial.suggest_int('ae_hidden_dim', 64, 512, step=64),
                    'num_layers': trial.suggest_int('ae_num_layers', 2, 6),
                    'dropout': trial.suggest_float('ae_dropout', 0.0, 0.3)
                },
                'decoder_config': {
                    'hidden_dim': trial.suggest_int('ae_decoder_hidden_dim', 64, 512, step=64),
                    'num_layers': trial.suggest_int('ae_decoder_num_layers', 2, 6)
                },
                'latent_dim': trial.suggest_int('ae_latent_dim', 32, 256, step=32),
                'learning_rate': trial.suggest_float('ae_lr', 1e-5, 1e-2, log=True)
            }
        }
    
    def _create_category_embedding(self, trial):
        """Create Category Embedding model using pytorch_tabular."""
        return {
            'model_type': 'category_embedding',
            'model_kwargs': {
                'layers': f"{trial.suggest_int('ce_layer1', 128, 512)}-{trial.suggest_int('ce_layer2', 64, 256)}",
                'activation': trial.suggest_categorical('ce_activation', ['ReLU', 'LeakyReLU', 'Swish']),
                'dropout': trial.suggest_float('ce_dropout', 0.0, 0.3),
                'use_batch_norm': trial.suggest_categorical('ce_batch_norm', [True, False]),
                'learning_rate': trial.suggest_float('ce_lr', 1e-5, 1e-2, log=True)
            }
        }
    
    def neural_network_objective(self, trial):
        """Optuna objective function for neural network models."""
        try:
            X_reduced = self.X_train
            y_reduced = self.y_train
            # Variance threshold for massive feature sets
            if X_reduced.shape[1] > 10000:
                var_threshold = VarianceThreshold(threshold=0.01)
                X_reduced = var_threshold.fit_transform(X_reduced)
                print(f"Variance threshold: {self.X_train.shape[1]} → {X_reduced.shape[1]} features")
            # Only apply SelectKBest if kbest_ratio is set
            if self.kbest_ratio:
                try:
                    num, den = map(float, self.kbest_ratio.split(':'))
                    ratio = num / den
                    k_select = int(X_reduced.shape[1] * ratio)
                    k_select = max(1, min(X_reduced.shape[1], k_select))
                    from sklearn.feature_selection import f_regression
                    selector = SelectKBest(f_regression, k=k_select)
                    X_reduced = selector.fit_transform(X_reduced, y_reduced)
                    print(f"SelectKBest (f_regression): → {X_reduced.shape[1]} features (ratio {self.kbest_ratio})")
                except Exception as e:
                    print(f"Invalid --kbest ratio '{self.kbest_ratio}': {e}. Skipping SelectKBest.")
            
            # Create neural network model
            model_config = self.create_neural_network_model(trial)
            
            # Simplified CV for neural networks (computationally expensive)
            simple_cv = RepeatedKFold(n_splits=3, n_repeats=1, random_state=42)
            fold_scores = []
            
            for train_idx, val_idx in simple_cv.split(X_reduced, y_reduced):
                X_fold_train, X_fold_val = X_reduced[train_idx], X_reduced[val_idx]
                y_fold_train, y_fold_val = y_reduced[train_idx], y_reduced[val_idx]
                
                # Preprocessing
                scaler = StandardScaler()
                X_fold_train_scaled = scaler.fit_transform(X_fold_train)
                X_fold_val_scaled = scaler.transform(X_fold_val)
                
                try:
                    # Train and evaluate model
                    if model_config.get('model_type') == 'tabnet':
                        score = self._train_tabnet_fold(model_config, X_fold_train_scaled, y_fold_train, 
                                                      X_fold_val_scaled, y_fold_val)
                    else:
                        score = self._train_pytorch_fold(model_config, X_fold_train_scaled, y_fold_train, 
                                                       X_fold_val_scaled, y_fold_val)
                    
                    fold_scores.append(score)
                    
                    # Early stopping if fold is terrible
                    if score < -5.0:
                        return -999.0
                        
                except Exception as e:
                    print(f"Fold training failed: {e}")
                    return -999.0
            
            # Return mean CV score
            mean_score = np.mean(fold_scores)
            return mean_score if not np.isnan(mean_score) else -999.0
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return -999.0
    
    def _train_pytorch_fold(self, model_config, X_train, y_train, X_val, y_val):
        """Train a PyTorch model for one fold."""
        device = model_config['device']
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        
        # Create model
        input_dim = X_train.shape[1]
        model_kwargs = model_config['model_kwargs'].copy()
        model_kwargs['input_dim'] = input_dim
        
        model = model_config['model_class'](**model_kwargs).to(device)
        
        # Setup training
        optimizer = optim.Adam(model.parameters(), lr=model_config['lr'])
        criterion = nn.MSELoss()
        
        # Create data loader
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=min(256, len(X_train) // 4), shuffle=True)
        
        # Training loop (reduced epochs for faster optimization)
        model.train()
        for epoch in range(50):  # Reduced from typical 100+ epochs
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor).cpu().numpy()
        
        # Calculate R² score
        score = r2_score(y_val, val_predictions)
        return score
    
    def _train_pytorch_fold_with_predictions(self, model_config, X_train, y_train, X_val, y_val):
        """Train a PyTorch model for one fold and return both score and predictions."""
        device = model_config['device']
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        
        # Create model
        input_dim = X_train.shape[1]
        model_kwargs = model_config['model_kwargs'].copy()
        model_kwargs['input_dim'] = input_dim
        
        model = model_config['model_class'](**model_kwargs).to(device)
        
        # Setup training
        optimizer = optim.Adam(model.parameters(), lr=model_config['lr'])
        criterion = nn.MSELoss()
        
        # Create data loader
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=min(256, len(X_train) // 4), shuffle=True)
        
        # Training loop (reduced epochs for faster optimization)
        model.train()
        for epoch in range(50):  # Reduced from typical 100+ epochs
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor).cpu().numpy()
        
        # Calculate R² score
        score = r2_score(y_val, val_predictions)
        return score, val_predictions
    
    def _train_tabnet_fold(self, model_config, X_train, y_train, X_val, y_val):
        """Train a TabNet model for one fold."""
        from pytorch_tabnet.tab_model import TabNetRegressor
        
        model = TabNetRegressor(**model_config['model_kwargs'])
        
        # Train model
        model.fit(
            X_train, y_train.reshape(-1, 1),
            eval_set=[(X_val, y_val.reshape(-1, 1))],
            max_epochs=50,  # Reduced for faster optimization
            patience=10,
            batch_size=min(256, len(X_train) // 4),
            virtual_batch_size=min(128, len(X_train) // 8),
            num_workers=0,
            drop_last=False,
            eval_metric=['rmse']
        )
        
        # Predict
        val_predictions = model.predict(X_val).flatten()
        
        # Calculate R² score
        score = r2_score(y_val, val_predictions)
        return score
    
    def _train_tabnet_fold_with_predictions(self, model_config, X_train, y_train, X_val, y_val):
        """Train a TabNet model for one fold and return both score and predictions."""
        from pytorch_tabnet.tab_model import TabNetRegressor
        
        model = TabNetRegressor(**model_config['model_kwargs'])
        
        # Train model
        model.fit(
            X_train, y_train.reshape(-1, 1),
            eval_set=[(X_val, y_val.reshape(-1, 1))],
            max_epochs=50,  # Reduced for faster optimization
            patience=10,
            batch_size=min(256, len(X_train) // 4),
            virtual_batch_size=min(128, len(X_train) // 8),
            num_workers=0,
            drop_last=False,
            eval_metric=['rmse']
        )
        
        # Predict
        val_predictions = model.predict(X_val).flatten()
        
        # Calculate R² score
        score = r2_score(y_val, val_predictions)
        return score, val_predictions
    
    def run_neural_network_optimization(self, X, y):
        """Main neural network optimization pipeline."""
        start_time = time.time()
        
        print(f"{Colors.BOLD}{Colors.CYAN}🧠 Starting Neural Network Optimization{Colors.END}")
        
        # Phase 1: Data preparation
        self._phase_1_data_preparation(X, y)
        
        # Phase 2: Neural network optimization
        best_r2, best_params = self._phase_2_nn_optimization()
        
        # Phase 3: Final evaluation
        results = self._phase_3_final_evaluation()
        
        elapsed = time.time() - start_time
        self.logger.info(f"Total neural network optimization time: {elapsed:.1f} seconds")
        
        print_results_summary(results, self.dataset_num)
        
        return results
    
    def _phase_1_data_preparation(self, X, y):
        """Phase 1: Data preparation and noise ceiling estimation."""
        print(f"{Colors.BOLD}{Colors.YELLOW}📊 Phase 1: Data Preparation{Colors.END}")
        # First split: train vs temp (test+holdout)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Second split: test vs holdout (from temp)
        X_test, X_holdout, y_test, y_holdout = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.X_holdout, self.y_holdout = X_holdout, y_holdout
        print(f"Training set: X{self.X_train.shape}, y({len(self.y_train)},)")
        print(f"Test set: X{self.X_test.shape}, y({len(self.y_test)},)")
        print(f"Holdout set: X{self.X_holdout.shape}, y({len(self.y_holdout)},)")
        # Save holdout set to best_model folder
        holdout_dir = self.model_dir
        holdout_dir.mkdir(exist_ok=True, parents=True)
        pd.DataFrame(self.X_holdout).to_csv(holdout_dir / "holdout_features.csv", index=False)
        pd.DataFrame(self.y_holdout).to_csv(holdout_dir / "holdout_targets.csv", index=False)
        # Estimate noise ceiling
        self._estimate_noise_ceiling()
    
    def _phase_2_nn_optimization(self):
        """Phase 2: Neural network hyperparameter optimization."""
        print(f"{Colors.BOLD}{Colors.YELLOW}🔍 Phase 2: Neural Network Optimization{Colors.END}")
        
        # Create Optuna study
        self.study = optuna.create_study(
            direction='maximize',
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Progress callback
        progress_callback = _create_progress_callback(
            self.auto_stop_at_noise_ceiling, 
            self.noise_ceiling, 
            self.max_hyperopt_trials
        )
        
        # Run optimization
        self.study.optimize(
            self.neural_network_objective,
            n_trials=self.max_hyperopt_trials,
            callbacks=[progress_callback] if self.auto_stop_at_noise_ceiling else None,
            n_jobs=1,  # Neural networks should use single process due to GPU
            show_progress_bar=True
        )
        
        best_params = self.study.best_params
        best_r2 = self.study.best_value
        
        print(f"{Colors.GREEN}✅ Best Neural Network R²: {best_r2:.4f}{Colors.END}")
        
        # Save best parameters
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        params_file = self.best_params_dir / f"best_nn_params_hold{self.dataset_num}_{timestamp}.json"
        with open(params_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        return best_r2, best_params
    
    def _phase_3_final_evaluation(self):
        """Phase 3: Final model evaluation on test set."""
        print(f"{Colors.BOLD}{Colors.YELLOW}🎯 Phase 3: Final Evaluation{Colors.END}")
        
        # Recreate and train the best model from scratch
        best_params = self.study.best_params
        
        # Handle case where trials failed (no valid parameters)
        if not best_params or 'nn_model_type' not in best_params:
            print(f"{Colors.RED}❌ No valid neural network model found (all trials failed){Colors.END}")
            return {
                'test_r2': -999.0,
                'test_mae': 999.0,
                'test_mse': 999.0,
                'best_params': {},
                'model_type': 'neural_network',
                'dataset_num': self.dataset_num,
                'noise_ceiling': self.noise_ceiling
            }
        
        # Feature reduction (same as in objective)
        X_reduced = self.X_train
        var_threshold = None
        selector = None
        
        if X_reduced.shape[1] > 10000:
            var_threshold = VarianceThreshold(threshold=0.01)
            X_reduced = var_threshold.fit_transform(X_reduced)
            X_test_reduced = var_threshold.transform(self.X_test)
        else:
            X_test_reduced = self.X_test
        
        if X_reduced.shape[1] > 5000:
            k_select = min(5000, X_reduced.shape[1] // 2)
            from sklearn.feature_selection import f_regression
            selector = SelectKBest(f_regression, k=k_select)
            X_reduced = selector.fit_transform(X_reduced, self.y_train)
            X_test_reduced = selector.transform(X_test_reduced)
        
        # Preprocessing
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_reduced)
        X_test_scaled = scaler.transform(X_test_reduced)
        
        # Manually recreate the best model configuration from parameters
        model_type = best_params['nn_model_type']
        device = self._get_gpu_device()  # AMD ROCm GPU only
        
        if model_type == 'simple_mlp':
            model_config = {
                'model_class': self._get_simple_mlp_class(),
                'model_kwargs': {
                    'hidden_dim': best_params['mlp_hidden_dim'],
                    'num_layers': best_params['mlp_num_layers'],
                    'dropout': best_params['mlp_dropout']
                },
                'lr': best_params['mlp_lr'],
                'device': device
            }
        elif model_type == 'deep_mlp':
            num_layers = best_params['deep_mlp_num_layers']
            base_dim = best_params['deep_mlp_base_dim']
            hidden_dims = [base_dim * (2 ** i) for i in range(num_layers)]
            hidden_dims = [min(dim, 1024) for dim in hidden_dims]
            model_config = {
                'model_class': self._get_deep_mlp_class(),
                'model_kwargs': {
                    'hidden_dims': hidden_dims,
                    'dropout': best_params['deep_mlp_dropout'],
                    'use_residual': True
                },
                'lr': best_params['deep_mlp_lr'],
                'device': device
            }
        elif model_type == 'wide_deep':
            num_deep_layers = best_params['wd_deep_layers']
            deep_base_dim = best_params['wd_deep_base_dim']
            deep_dims = [deep_base_dim // (2 ** i) for i in range(num_deep_layers)]
            deep_dims = [max(dim, 32) for dim in deep_dims]
            model_config = {
                'model_class': self._get_wide_deep_class(),
                'model_kwargs': {
                    'deep_dims': deep_dims,
                    'wide_dim_ratio': best_params['wd_wide_ratio']
                },
                'lr': best_params['wd_lr'],
                'device': device
            }
        elif model_type == 'resnet_custom':
            # Directly use the CustomResNet class and best_params for kwargs
            class ResNetBlock(nn.Module):
                def __init__(self, dim, dropout=0.1):
                    super().__init__()
                    self.norm1 = nn.LayerNorm(dim)
                    self.linear1 = nn.Linear(dim, dim)
                    self.dropout1 = nn.Dropout(dropout)
                    self.norm2 = nn.LayerNorm(dim)
                    self.linear2 = nn.Linear(dim, dim)
                    self.dropout2 = nn.Dropout(dropout)
                def forward(self, x):
                    residual = x
                    x = self.norm1(x)
                    x = torch.relu(self.linear1(x))
                    x = self.dropout1(x)
                    x = self.norm2(x)
                    x = self.linear2(x)
                    x = self.dropout2(x)
                    return x + residual
            class CustomResNet(nn.Module):
                def __init__(self, input_dim, hidden_dim, num_blocks, dropout):
                    super().__init__()
                    self.input_proj = nn.Linear(input_dim, hidden_dim)
                    self.blocks = nn.ModuleList([ResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)])
                    self.output = nn.Linear(hidden_dim, 1)
                def forward(self, x):
                    x = self.input_proj(x)
                    for block in self.blocks:
                        x = block(x)
                    return self.output(x).squeeze(-1)
            model_config = {
                'model_class': CustomResNet,
                'model_kwargs': {
                    'hidden_dim': best_params['resnet_hidden_dim'],
                    'num_blocks': best_params['resnet_num_blocks'],
                    'dropout': best_params['resnet_dropout']
                },
                'lr': best_params['resnet_lr'],
                'device': device
            }
        elif model_type == 'tabnet':
            model_config = {
                'model_type': 'tabnet',
                'model_kwargs': {
                    'n_d': best_params['tabnet_n_d'],
                    'n_a': best_params['tabnet_n_a'],
                    'n_steps': best_params['tabnet_n_steps'],
                    'gamma': best_params['tabnet_gamma'],
                    'lambda_sparse': best_params['tabnet_lambda_sparse'],
                    'optimizer_fn': torch.optim.Adam,
                    'optimizer_params': {'lr': best_params['tabnet_lr']},
                    'scheduler_params': {'step_size': 50, 'gamma': 0.9},
                    'scheduler_fn': torch.optim.lr_scheduler.StepLR,
                    'mask_type': 'sparsemax'
                }
            }
        elif model_type == 'deepfm':
            # Directly use the DeepFM class and best_params for kwargs
            class DeepFM(nn.Module):
                def __init__(self, input_dim, embedding_dim, hidden_dims, dropout):
                    super().__init__()
                    self.input_dim = input_dim
                    self.embedding_dim = embedding_dim
                    
                    # FM part
                    self.fm_linear = nn.Linear(input_dim, 1)
                    self.fm_embedding = nn.Linear(input_dim, embedding_dim)
                    
                    # Deep part
                    deep_layers = []
                    current_dim = input_dim
                    for hidden_dim in hidden_dims:
                        deep_layers.extend([
                            nn.Linear(current_dim, hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(dropout)
                        ])
                        current_dim = hidden_dim
                    deep_layers.append(nn.Linear(current_dim, 1))
                    self.deep = nn.Sequential(*deep_layers)
                    
                def forward(self, x):
                    # FM part
                    linear_part = self.fm_linear(x)
                    embed_x = self.fm_embedding(x)
                    
                    # Ensure embed_x has correct shape for FM computation
                    if embed_x.dim() == 1:
                        embed_x = embed_x.unsqueeze(0)
                        
                    square_of_sum = torch.pow(torch.sum(embed_x, dim=-1), 2)
                    sum_of_square = torch.sum(torch.pow(embed_x, 2), dim=-1)
                    fm_part = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=-1, keepdim=True)
                    
                    # Deep part
                    deep_part = self.deep(x)
                    
                    return (linear_part + fm_part + deep_part).squeeze(-1)
            
            num_layers = best_params['deepfm_num_layers']
            base_dim = best_params['deepfm_base_dim']
            hidden_dims = [base_dim // (2 ** i) for i in range(num_layers)]
            hidden_dims = [max(dim, 32) for dim in hidden_dims]
            model_config = {
                'model_class': DeepFM,
                'model_kwargs': {
                    'embedding_dim': best_params['deepfm_embedding_dim'],
                    'hidden_dims': hidden_dims,
                    'dropout': best_params['deepfm_dropout']
                },
                'lr': best_params['deepfm_lr'],
                'device': device
            }
        elif model_type == 'dcn':
            # DCN (Deep & Cross Network)
            class DCN(nn.Module):
                def __init__(self, input_dim, cross_layers, deep_dims, dropout):
                    super().__init__()
                    self.input_dim = input_dim
                    self.cross_layers = cross_layers
                    
                    # Cross network
                    self.cross_weights = nn.ParameterList([
                        nn.Parameter(torch.randn(input_dim, 1)) for _ in range(cross_layers)
                    ])
                    self.cross_biases = nn.ParameterList([
                        nn.Parameter(torch.randn(input_dim)) for _ in range(cross_layers)
                    ])
                    
                    # Deep network
                    deep_layers = []
                    current_dim = input_dim
                    for deep_dim in deep_dims:
                        deep_layers.extend([
                            nn.Linear(current_dim, deep_dim),
                            nn.ReLU(),
                            nn.Dropout(dropout)
                        ])
                        current_dim = deep_dim
                    self.deep = nn.Sequential(*deep_layers)
                    
                    # Output layer
                    self.output = nn.Linear(input_dim + current_dim, 1)
                    
                def forward(self, x):
                    # Cross network
                    x_cross = x
                    for i in range(self.cross_layers):
                        xl_w = torch.mm(x_cross, self.cross_weights[i])
                        x_cross = x * xl_w + self.cross_biases[i] + x_cross
                    
                    # Deep network
                    x_deep = self.deep(x)
                    
                    # Concatenate and output
                    x_combined = torch.cat([x_cross, x_deep], dim=1)
                    return self.output(x_combined).squeeze(-1)
            
            num_deep_layers = best_params['dcn_deep_layers']
            base_dim = best_params['dcn_base_dim']
            deep_dims = [base_dim // (2 ** i) for i in range(num_deep_layers)]
            deep_dims = [max(dim, 32) for dim in deep_dims]
            model_config = {
                'model_class': DCN,
                'model_kwargs': {
                    'cross_layers': best_params['dcn_cross_layers'],
                    'deep_dims': deep_dims,
                    'dropout': best_params['dcn_dropout']
                },
                'lr': best_params['dcn_lr'],
                'device': device
            }
        else:
            raise ValueError(f"Unknown model type for final evaluation: {model_type}")
        
        # Train final model and get actual predictions
        if model_config.get('model_type') == 'tabnet':
            final_score, y_pred = self._train_tabnet_fold_with_predictions(model_config, X_train_scaled, self.y_train, 
                                                                         X_test_scaled, self.y_test)
        else:
            final_score, y_pred = self._train_pytorch_fold_with_predictions(model_config, X_train_scaled, self.y_train, 
                                                                           X_test_scaled, self.y_test)
        
        # Calculate real metrics from actual predictions
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        
        results = {
            'test_r2': final_score,
            'test_mae': mae,
            'test_mse': mse,
            'best_params': best_params,
            'model_type': 'neural_network',
            'dataset_num': self.dataset_num,
            'noise_ceiling': self.noise_ceiling
        }
        
        # Save model artifacts including all preprocessors
        preprocessors = {'scaler': scaler}
        if var_threshold is not None:
            preprocessors['var_threshold'] = var_threshold
        if selector is not None:
            preprocessors['kbest_selector'] = selector
        
        # Train final model with best parameters and save weights
        print(f"🏆 Training final model with best parameters for inference...")
        
        # Train the final model on full training set
        try:
            if model_config.get('model_type') == 'tabnet':
                # TabNet has special training procedure
                final_model, _ = self._train_tabnet_fold_with_predictions(
                    model_config, X_train_scaled, self.y_train, X_test_scaled, self.y_test
                )
                # Save TabNet model 
                model_save_path = self.model_dir / f"hold{self.dataset_num}_final_model.pkl"
                torch.save(final_model.state_dict(), model_save_path)
            else:
                # Regular PyTorch models
                final_model = self._train_pytorch_model_final(
                    model_config, X_train_scaled, self.y_train
                )
                # Save PyTorch model state dict
                model_save_path = self.model_dir / f"hold{self.dataset_num}_final_model.pt"
                torch.save(final_model.state_dict(), model_save_path)
                
            print(f"💾 Model weights saved to: {model_save_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save model weights: {e}")
            print("Saving architecture info as fallback...")
        
        # Save model architecture info for inference recreation (without unpicklable objects)
        neural_network_model_info = {
            'model_type': model_type,
            'best_params': best_params,
            'input_features': X_train_scaled.shape[1]  # Number of features after preprocessing
        }
        
        save_model_artifacts(neural_network_model_info, preprocessors, self.dataset_num, results)
        
        return results
    
    def _train_pytorch_model_final(self, model_config, X_train, y_train):
        """Train final PyTorch model with best parameters for saving."""
        device = model_config['device']
        
        # Create model instance
        model_class = model_config['model_class']
        model_kwargs = model_config['model_kwargs'].copy()
        model_kwargs['input_dim'] = X_train.shape[1]
        model = model_class(**model_kwargs).to(device)
        
        # Setup training
        lr = model_config['lr']
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Convert data to tensors
        X_tensor = torch.FloatTensor(X_train).to(device)
        y_tensor = torch.FloatTensor(y_train).to(device)
        
        # Training loop
        model.train()
        epochs = 100  # Quick training for final model
        batch_size = min(512, X_train.shape[0])
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            # Mini-batch training
            for i in range(0, len(X_tensor), batch_size):
                batch_X = X_tensor[i:i+batch_size]
                batch_y = y_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / num_batches
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        model.eval()
        return model
    
    def _get_simple_mlp_class(self):
        """Get the SimpleMLP class definition."""
        class SimpleMLP(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, dropout):
                super().__init__()
                layers = []
                current_dim = input_dim
                
                for i in range(num_layers):
                    layers.append(nn.Linear(current_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    current_dim = hidden_dim
                
                layers.append(nn.Linear(current_dim, 1))
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x).squeeze(-1)
        
        return SimpleMLP
    
    def _get_deep_mlp_class(self):
        """Get the DeepMLP class definition."""
        class DeepMLP(nn.Module):
            def __init__(self, input_dim, hidden_dims, dropout, use_residual=True):
                super().__init__()
                self.use_residual = use_residual
                layers = []
                current_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers.append(nn.Linear(current_dim, hidden_dim))
                    layers.append(nn.BatchNorm1d(hidden_dim))
                    layers.append(nn.ReLU())
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    current_dim = hidden_dim
                
                self.backbone = nn.Sequential(*layers)
                self.head = nn.Linear(current_dim, 1)
                
                # Residual connection if dimensions match
                self.residual = nn.Linear(input_dim, current_dim) if input_dim != current_dim else nn.Identity()
            
            def forward(self, x):
                backbone_out = self.backbone(x)
                if self.use_residual:
                    residual_out = self.residual(x)
                    backbone_out = backbone_out + residual_out
                return self.head(backbone_out).squeeze(-1)
        
        return DeepMLP
    
    def _get_wide_deep_class(self):
        """Get the WideDeepModel class definition."""
        class WideDeepModel(nn.Module):
            def __init__(self, input_dim, deep_dims, wide_dim_ratio=0.5):
                super().__init__()
                # Split features for wide and deep parts
                self.wide_dim = int(input_dim * wide_dim_ratio)
                self.deep_dim = input_dim - self.wide_dim
                
                # Wide part (linear)
                self.wide = nn.Linear(self.wide_dim, 1)
                
                # Deep part (MLP)
                deep_layers = []
                current_dim = self.deep_dim
                for hidden_dim in deep_dims:
                    deep_layers.extend([
                        nn.Linear(current_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.2)
                    ])
                    current_dim = hidden_dim
                deep_layers.append(nn.Linear(current_dim, 1))
                self.deep = nn.Sequential(*deep_layers)
            
            def forward(self, x):
                wide_features = x[:, :self.wide_dim]
                deep_features = x[:, self.wide_dim:]
                
                wide_out = self.wide(wide_features)
                deep_out = self.deep(deep_features)
                
                return (wide_out + deep_out).squeeze(-1)
        
        return WideDeepModel
    
    def _create_resnet_custom(self, trial, device):
        """Create custom ResNet model."""
        class ResNetBlock(nn.Module):
            def __init__(self, dim, dropout=0.1):
                super().__init__()
                self.norm1 = nn.LayerNorm(dim)
                self.linear1 = nn.Linear(dim, dim)
                self.dropout1 = nn.Dropout(dropout)
                self.norm2 = nn.LayerNorm(dim)
                self.linear2 = nn.Linear(dim, dim)
                self.dropout2 = nn.Dropout(dropout)
                
            def forward(self, x):
                residual = x
                x = self.norm1(x)
                x = torch.relu(self.linear1(x))
                x = self.dropout1(x)
                x = self.norm2(x)
                x = self.linear2(x)
                x = self.dropout2(x)
                return x + residual
        
        class CustomResNet(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_blocks, dropout):
                super().__init__()
                self.input_proj = nn.Linear(input_dim, hidden_dim)
                self.blocks = nn.ModuleList([ResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)])
                self.output = nn.Linear(hidden_dim, 1)
                
            def forward(self, x):
                x = self.input_proj(x)
                for block in self.blocks:
                    x = block(x)
                return self.output(x).squeeze(-1)
        
        hidden_dim = trial.suggest_int('resnet_hidden_dim', 64, 512, step=64)
        num_blocks = trial.suggest_int('resnet_num_blocks', 2, 8)
        dropout = trial.suggest_float('resnet_dropout', 0.0, 0.3)
        lr = trial.suggest_float('resnet_lr', 1e-5, 1e-2, log=True)
        
        return {
            'model_class': CustomResNet,
            'model_kwargs': {
                'hidden_dim': hidden_dim,
                'num_blocks': num_blocks,
                'dropout': dropout
            },
            'lr': lr,
            'device': device
        }
    
    def _create_resnet_rtdl(self, trial, device):
        """Create ResNet using rtdl if available."""
        if not HAS_RTDL:
            raise ImportError("rtdl not available")
        import rtdl
        
        d_main = trial.suggest_int('rtdl_resnet_d_main', 64, 512, step=64)
        d_hidden = trial.suggest_int('rtdl_resnet_d_hidden', 64, 512, step=64)
        n_blocks = trial.suggest_int('rtdl_resnet_n_blocks', 2, 8)
        dropout = trial.suggest_float('rtdl_resnet_dropout', 0.0, 0.3)
        lr = trial.suggest_float('rtdl_resnet_lr', 1e-5, 1e-2, log=True)
        
        return {
            'model_type': 'rtdl_resnet',
            'model_class': rtdl.ResNet,
            'model_kwargs': {
                'd_main': d_main,
                'd_hidden': d_hidden,
                'n_blocks': n_blocks,
                'dropout': dropout,
                'd_out': 1
            },
            'lr': lr,
            'device': device
        }
    
    def _create_ft_mlp(self, trial, device):
        """Create FT-MLP using rtdl if available."""
        if not HAS_RTDL:
            raise ImportError("rtdl not available")
        import rtdl
        
        token_dim = trial.suggest_int('ft_mlp_token_dim', 32, 256, step=32)
        depth = trial.suggest_int('ft_mlp_depth', 2, 6)
        lr = trial.suggest_float('ft_mlp_lr', 1e-5, 1e-2, log=True)
        
        return {
            'model_type': 'rtdl_ft_mlp',
            'model_class': rtdl.FTMLP,
            'model_kwargs': {
                'd_token': token_dim,
                'n_blocks': depth,
                'd_out': 1
            },
            'lr': lr,
            'device': device
        }
    
    def _create_ft_transformer_pt(self, trial):
        """Create FT-Transformer using pytorch_tabular if available."""
        if not HAS_PYTORCH_TABULAR:
            raise ImportError("pytorch_tabular not available")
        
        return {
            'model_type': 'ft_transformer_pt',
            'model_kwargs': {
                'd_token': trial.suggest_int('ft_pt_d_token', 32, 256, step=32),
                'n_blocks': trial.suggest_int('ft_pt_n_blocks', 2, 6),
                'attention_dropout': trial.suggest_float('ft_pt_attention_dropout', 0.0, 0.3),
                'ffn_dropout': trial.suggest_float('ft_pt_ffn_dropout', 0.0, 0.3),
                'learning_rate': trial.suggest_float('ft_pt_lr', 1e-5, 1e-2, log=True)
            }
        }
    
    def _create_deepfm(self, trial, device):
        """Create DeepFM model."""
        class DeepFM(nn.Module):
            def __init__(self, input_dim, embedding_dim, hidden_dims, dropout):
                super().__init__()
                self.input_dim = input_dim
                self.embedding_dim = embedding_dim
                
                # FM part
                self.fm_linear = nn.Linear(input_dim, 1)
                self.fm_embedding = nn.Linear(input_dim, embedding_dim)
                
                # Deep part
                deep_layers = []
                current_dim = input_dim
                for hidden_dim in hidden_dims:
                    deep_layers.extend([
                        nn.Linear(current_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ])
                    current_dim = hidden_dim
                deep_layers.append(nn.Linear(current_dim, 1))
                self.deep = nn.Sequential(*deep_layers)
                
            def forward(self, x):
                # FM part
                linear_part = self.fm_linear(x)
                embed_x = self.fm_embedding(x)
                
                # Ensure embed_x has correct shape for FM computation
                if embed_x.dim() == 1:
                    embed_x = embed_x.unsqueeze(0)
                    
                square_of_sum = torch.pow(torch.sum(embed_x, dim=-1), 2)
                sum_of_square = torch.sum(torch.pow(embed_x, 2), dim=-1)
                fm_part = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=-1, keepdim=True)
                
                # Deep part
                deep_part = self.deep(x)
                
                return (linear_part + fm_part + deep_part).squeeze(-1)
        
        embedding_dim = trial.suggest_int('deepfm_embedding_dim', 8, 64)
        num_layers = trial.suggest_int('deepfm_num_layers', 2, 5)
        base_dim = trial.suggest_int('deepfm_base_dim', 128, 512, step=64)
        hidden_dims = [base_dim // (2 ** i) for i in range(num_layers)]
        hidden_dims = [max(dim, 32) for dim in hidden_dims]
        dropout = trial.suggest_float('deepfm_dropout', 0.0, 0.3)
        lr = trial.suggest_float('deepfm_lr', 1e-5, 1e-2, log=True)
        
        return {
            'model_class': DeepFM,
            'model_kwargs': {
                'embedding_dim': embedding_dim,
                'hidden_dims': hidden_dims,
                'dropout': dropout
            },
            'lr': lr,
            'device': device
        }
    
    def _create_dcn(self, trial, device):
        """Create Deep & Cross Network."""
        class DCN(nn.Module):
            def __init__(self, input_dim, cross_layers, deep_dims, dropout):
                super().__init__()
                self.input_dim = input_dim
                self.cross_layers = cross_layers
                
                # Cross network
                self.cross_weights = nn.ParameterList([
                    nn.Parameter(torch.randn(input_dim, 1)) for _ in range(cross_layers)
                ])
                self.cross_biases = nn.ParameterList([
                    nn.Parameter(torch.randn(input_dim)) for _ in range(cross_layers)
                ])
                
                # Deep network
                deep_layers = []
                current_dim = input_dim
                for hidden_dim in deep_dims:
                    deep_layers.extend([
                        nn.Linear(current_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ])
                    current_dim = hidden_dim
                self.deep = nn.Sequential(*deep_layers)
                
                # Final layer
                self.final = nn.Linear(input_dim + current_dim, 1)
                
            def forward(self, x):
                # Cross network
                x_cross = x
                for i in range(self.cross_layers):
                    xl_w = torch.mm(x_cross, self.cross_weights[i])
                    x_cross = x * xl_w + self.cross_biases[i] + x_cross
                
                # Deep network
                x_deep = self.deep(x)
                
                # Concatenate and final layer
                final_input = torch.cat([x_cross, x_deep], dim=1)
                return self.final(final_input).squeeze(-1)
        
        cross_layers = trial.suggest_int('dcn_cross_layers', 2, 6)
        num_deep_layers = trial.suggest_int('dcn_deep_layers', 2, 5)
        base_dim = trial.suggest_int('dcn_base_dim', 128, 512, step=64)
        deep_dims = [base_dim // (2 ** i) for i in range(num_deep_layers)]
        deep_dims = [max(dim, 32) for dim in deep_dims]
        dropout = trial.suggest_float('dcn_dropout', 0.0, 0.3)
        lr = trial.suggest_float('dcn_lr', 1e-5, 1e-2, log=True)
        
        return {
            'model_class': DCN,
            'model_kwargs': {
                'cross_layers': cross_layers,
                'deep_dims': deep_dims,
                'dropout': dropout
            },
            'lr': lr,
            'device': device
        }
    
    def _create_wide_deep_advanced(self, trial, device):
        """Create advanced Wide & Deep with attention."""
        class AdvancedWideDeep(nn.Module):
            def __init__(self, input_dim, deep_dims, wide_dim_ratio, attention_dim):
                super().__init__()
                self.wide_dim = int(input_dim * wide_dim_ratio)
                self.deep_dim = input_dim - self.wide_dim
                
                # Wide part with attention
                self.wide_attention = nn.Sequential(
                    nn.Linear(self.wide_dim, attention_dim),
                    nn.Tanh(),
                    nn.Linear(attention_dim, self.wide_dim),
                    nn.Softmax(dim=1)
                )
                self.wide = nn.Linear(self.wide_dim, 1)
                
                # Deep part
                deep_layers = []
                current_dim = self.deep_dim
                for hidden_dim in deep_dims:
                    deep_layers.extend([
                        nn.Linear(current_dim, hidden_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_dim),
                        nn.Dropout(0.2)
                    ])
                    current_dim = hidden_dim
                deep_layers.append(nn.Linear(current_dim, 1))
                self.deep = nn.Sequential(*deep_layers)
                
            def forward(self, x):
                wide_features = x[:, :self.wide_dim]
                deep_features = x[:, self.wide_dim:]
                
                # Wide with attention
                attention_weights = self.wide_attention(wide_features)
                wide_weighted = wide_features * attention_weights
                wide_out = self.wide(wide_weighted)
                
                # Deep
                deep_out = self.deep(deep_features)
                
                return (wide_out + deep_out).squeeze(-1)
        
        num_deep_layers = trial.suggest_int('wd_adv_deep_layers', 2, 6)
        deep_base_dim = trial.suggest_int('wd_adv_deep_base_dim', 128, 512, step=64)
        deep_dims = [deep_base_dim // (2 ** i) for i in range(num_deep_layers)]
        deep_dims = [max(dim, 32) for dim in deep_dims]
        wide_ratio = trial.suggest_float('wd_adv_wide_ratio', 0.2, 0.8)
        attention_dim = trial.suggest_int('wd_adv_attention_dim', 32, 128, step=16)
        lr = trial.suggest_float('wd_adv_lr', 1e-5, 1e-2, log=True)
        
        return {
            'model_class': AdvancedWideDeep,
            'model_kwargs': {
                'deep_dims': deep_dims,
                'wide_dim_ratio': wide_ratio,
                'attention_dim': attention_dim
            },
            'lr': lr,
            'device': device
        }
    
    def _create_autoint(self, trial, device):
        """Create AutoInt model."""
        class AutoInt(nn.Module):
            def __init__(self, input_dim, embed_dim, num_heads, num_layers, dropout):
                super().__init__()
                self.embed_dim = embed_dim
                self.input_projection = nn.Linear(input_dim, embed_dim)
                
                # Multi-head attention layers
                self.attention_layers = nn.ModuleList([
                    nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
                    for _ in range(num_layers)
                ])
                
                self.layer_norms = nn.ModuleList([
                    nn.LayerNorm(embed_dim) for _ in range(num_layers)
                ])
                
                self.output = nn.Linear(embed_dim, 1)
                
            def forward(self, x):
                # Project to embedding dimension
                x = self.input_projection(x)  # [batch, embed_dim]
                x = x.unsqueeze(1)  # [batch, 1, embed_dim] for attention
                
                # Apply attention layers
                for attention, norm in zip(self.attention_layers, self.layer_norms):
                    residual = x
                    x, _ = attention(x, x, x)
                    x = norm(x + residual)
                
                # Global average pooling and output
                x = x.squeeze(1)  # [batch, embed_dim]
                return self.output(x).squeeze(-1)
        
        embed_dim = trial.suggest_int('autoint_embed_dim', 64, 256, step=32)
        num_heads = trial.suggest_int('autoint_num_heads', 2, 8)
        num_layers = trial.suggest_int('autoint_num_layers', 2, 6)
        dropout = trial.suggest_float('autoint_dropout', 0.0, 0.3)
        lr = trial.suggest_float('autoint_lr', 1e-5, 1e-2, log=True)
        
        return {
            'model_class': AutoInt,
            'model_kwargs': {
                'embed_dim': embed_dim,
                'num_heads': num_heads,
                'num_layers': num_layers,
                'dropout': dropout
            },
            'lr': lr,
            'device': device
        }
    
    def _create_dndt(self, trial, device):
        """Create Deep Neural Decision Tree."""
        class DNDT(nn.Module):
            def __init__(self, input_dim, tree_depth, num_trees, dropout):
                super().__init__()
                self.tree_depth = tree_depth
                self.num_trees = num_trees
                self.num_leaves = 2 ** tree_depth
                
                # Decision trees
                self.trees = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(input_dim, 64),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(64, self.num_leaves)
                    ) for _ in range(num_trees)
                ])
                
                # Leaf values
                self.leaf_values = nn.Parameter(torch.randn(num_trees, self.num_leaves))
                
            def forward(self, x):
                tree_outputs = []
                for i, tree in enumerate(self.trees):
                    tree_logits = tree(x)
                    tree_probs = torch.softmax(tree_logits, dim=1)
                    tree_output = torch.sum(tree_probs * self.leaf_values[i], dim=1)
                    tree_outputs.append(tree_output)
                
                return torch.mean(torch.stack(tree_outputs), dim=0)
        
        tree_depth = trial.suggest_int('dndt_tree_depth', 3, 7)
        num_trees = trial.suggest_int('dndt_num_trees', 5, 20)
        dropout = trial.suggest_float('dndt_dropout', 0.0, 0.3)
        lr = trial.suggest_float('dndt_lr', 1e-5, 1e-2, log=True)
        
        return {
            'model_class': DNDT,
            'model_kwargs': {
                'tree_depth': tree_depth,
                'num_trees': num_trees,
                'dropout': dropout
            },
            'lr': lr,
            'device': device
        }
    
    def _create_moe(self, trial, device):
        """Create Mixture of Experts."""
        class MoE(nn.Module):
            def __init__(self, input_dim, num_experts, expert_dim, top_k):
                super().__init__()
                self.num_experts = num_experts
                self.top_k = top_k
                
                # Gate network
                self.gate = nn.Linear(input_dim, num_experts)
                
                # Expert networks
                self.experts = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(input_dim, expert_dim),
                        nn.ReLU(),
                        nn.Linear(expert_dim, expert_dim),
                        nn.ReLU(),
                        nn.Linear(expert_dim, 1)
                    ) for _ in range(num_experts)
                ])
                
            def forward(self, x):
                # Gate scores
                gate_scores = torch.softmax(self.gate(x), dim=1)
                
                # Get top-k experts
                top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=1)
                top_k_scores = torch.softmax(top_k_scores, dim=1)
                
                # Expert outputs
                expert_outputs = torch.stack([expert(x).squeeze(-1) for expert in self.experts], dim=1)
                
                # Weighted combination
                batch_size = x.size(0)
                final_output = torch.zeros(batch_size, device=x.device)
                
                for i in range(batch_size):
                    for j in range(self.top_k):
                        expert_idx = top_k_indices[i, j]
                        weight = top_k_scores[i, j]
                        final_output[i] += weight * expert_outputs[i, expert_idx]
                
                return final_output
        
        num_experts = trial.suggest_int('moe_num_experts', 4, 16)
        expert_dim = trial.suggest_int('moe_expert_dim', 64, 256, step=32)
        top_k = trial.suggest_int('moe_top_k', 2, min(4, num_experts))
        lr = trial.suggest_float('moe_lr', 1e-5, 1e-2, log=True)
        
        return {
            'model_class': MoE,
            'model_kwargs': {
                'num_experts': num_experts,
                'expert_dim': expert_dim,
                'top_k': top_k
            },
            'lr': lr,
            'device': device
        }
    
    def _create_vime(self, trial, device):
        """Create VIME self-supervised model."""
        class VIME(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, mask_prob, dropout):
                super().__init__()
                self.mask_prob = mask_prob
                
                # Encoder
                encoder_layers = []
                current_dim = input_dim
                for _ in range(num_layers):
                    encoder_layers.extend([
                        nn.Linear(current_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ])
                    current_dim = hidden_dim
                self.encoder = nn.Sequential(*encoder_layers)
                
                # Predictor
                self.predictor = nn.Linear(hidden_dim, 1)
                
            def forward(self, x, training=True):
                if training and self.mask_prob > 0:
                    # Apply random masking during training
                    mask = torch.rand_like(x) > self.mask_prob
                    x_masked = x * mask.float()
                else:
                    x_masked = x
                    
                encoded = self.encoder(x_masked)
                return self.predictor(encoded).squeeze(-1)
        
        hidden_dim = trial.suggest_int('vime_hidden_dim', 64, 256, step=32)
        num_layers = trial.suggest_int('vime_num_layers', 2, 5)
        mask_prob = trial.suggest_float('vime_mask_prob', 0.1, 0.3)
        dropout = trial.suggest_float('vime_dropout', 0.0, 0.3)
        lr = trial.suggest_float('vime_lr', 1e-5, 1e-2, log=True)
        
        return {
            'model_class': VIME,
            'model_kwargs': {
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'mask_prob': mask_prob,
                'dropout': dropout
            },
            'lr': lr,
            'device': device
        }
    
    def _create_scarf(self, trial, device):
        """Create SCARF contrastive model."""
        class SCARF(nn.Module):
            def __init__(self, input_dim, encoder_dim, projection_dim, temperature, dropout):
                super().__init__()
                self.temperature = temperature
                
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, encoder_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(encoder_dim, encoder_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                
                # Projection head
                self.projection = nn.Linear(encoder_dim, projection_dim)
                
                # Predictor
                self.predictor = nn.Linear(encoder_dim, 1)
                
            def forward(self, x):
                encoded = self.encoder(x)
                return self.predictor(encoded).squeeze(-1)
        
        encoder_dim = trial.suggest_int('scarf_encoder_dim', 64, 256, step=32)
        projection_dim = trial.suggest_int('scarf_projection_dim', 32, 128, step=16)
        temperature = trial.suggest_float('scarf_temperature', 0.1, 1.0)
        dropout = trial.suggest_float('scarf_dropout', 0.0, 0.3)
        lr = trial.suggest_float('scarf_lr', 1e-5, 1e-2, log=True)
        
        return {
            'model_class': SCARF,
            'model_kwargs': {
                'encoder_dim': encoder_dim,
                'projection_dim': projection_dim,
                'temperature': temperature,
                'dropout': dropout
            },
            'lr': lr,
            'device': device
        }
    
    def _create_bayesian_nn(self, trial, device):
        """Create Bayesian Neural Network."""
        class BayesianNN(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, prior_std):
                super().__init__()
                self.prior_std = prior_std
                
                # Bayesian layers
                layers = []
                current_dim = input_dim
                for _ in range(num_layers):
                    layers.append(nn.Linear(current_dim, hidden_dim))
                    layers.append(nn.ReLU())
                    current_dim = hidden_dim
                layers.append(nn.Linear(current_dim, 1))
                
                self.network = nn.Sequential(*layers)
                
                # Initialize with wider priors
                for layer in self.network:
                    if isinstance(layer, nn.Linear):
                        nn.init.normal_(layer.weight, 0, prior_std)
                        nn.init.normal_(layer.bias, 0, prior_std)
                
            def forward(self, x):
                return self.network(x).squeeze(-1)
        
        hidden_dim = trial.suggest_int('bayesian_hidden_dim', 64, 256, step=32)
        num_layers = trial.suggest_int('bayesian_num_layers', 2, 5)
        prior_std = trial.suggest_float('bayesian_prior_std', 0.1, 1.0)
        lr = trial.suggest_float('bayesian_lr', 1e-5, 1e-2, log=True)
        
        return {
            'model_class': BayesianNN,
            'model_kwargs': {
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'prior_std': prior_std
            },
            'lr': lr,
            'device': device
        }
    
    def _create_saint(self, trial, device):
        """Create SAINT transformer model."""
        class SAINT(nn.Module):
            def __init__(self, input_dim, embed_dim, num_heads, num_layers, mix_ratio, dropout):
                super().__init__()
                self.embed_dim = embed_dim
                self.mix_ratio = mix_ratio
                
                # Input embedding
                self.input_embedding = nn.Linear(input_dim, embed_dim)
                
                # Transformer layers
                self.transformer_layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=num_heads,
                        dim_feedforward=embed_dim * 4,
                        dropout=dropout,
                        batch_first=True
                    ) for _ in range(num_layers)
                ])
                
                # Output layer
                self.output = nn.Linear(embed_dim, 1)
                
            def forward(self, x):
                # Embed input
                x = self.input_embedding(x)  # [batch, embed_dim]
                x = x.unsqueeze(1)  # [batch, 1, embed_dim] for transformer
                
                # Apply transformer layers
                for layer in self.transformer_layers:
                    x = layer(x)
                
                # Global pooling and output
                x = x.squeeze(1)  # [batch, embed_dim]
                return self.output(x).squeeze(-1)
        
        embed_dim = trial.suggest_int('saint_embed_dim', 64, 256, step=32)
        num_heads = trial.suggest_int('saint_num_heads', 2, 8)
        num_layers = trial.suggest_int('saint_num_layers', 2, 6)
        mix_ratio = trial.suggest_float('saint_mix_ratio', 0.0, 0.5)
        dropout = trial.suggest_float('saint_dropout', 0.0, 0.3)
        lr = trial.suggest_float('saint_lr', 1e-5, 1e-2, log=True)
        
        return {
            'model_class': SAINT,
            'model_kwargs': {
                'embed_dim': embed_dim,
                'num_heads': num_heads,
                'num_layers': num_layers,
                'mix_ratio': mix_ratio,
                'dropout': dropout
            },
            'lr': lr,
            'device': device
        }
    
    def _create_tab_transformer(self, trial, device):
        """Create TabTransformer model."""
        class TabTransformer(nn.Module):
            def __init__(self, input_dim, embed_dim, num_heads, num_layers, dropout):
                super().__init__()
                self.embed_dim = embed_dim
                
                # Column embeddings
                self.column_embeddings = nn.Parameter(torch.randn(input_dim, embed_dim))
                
                # Transformer
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=num_heads,
                        dim_feedforward=embed_dim * 4,
                        dropout=dropout,
                        batch_first=True
                    ),
                    num_layers=num_layers
                )
                
                # Output
                self.norm = nn.LayerNorm(embed_dim)
                self.output = nn.Linear(embed_dim, 1)
                
            def forward(self, x):
                batch_size = x.size(0)
                
                # Create embeddings for each feature
                x_expanded = x.unsqueeze(-1)  # [batch, input_dim, 1]
                embeddings = x_expanded * self.column_embeddings  # [batch, input_dim, embed_dim]
                
                # Apply transformer
                output = self.transformer(embeddings)
                
                # Global average pooling
                output = torch.mean(output, dim=1)  # [batch, embed_dim]
                
                # Final prediction
                output = self.norm(output)
                return self.output(output).squeeze(-1)
        
        embed_dim = trial.suggest_int('tabtrans_embed_dim', 64, 256, step=32)
        num_heads = trial.suggest_int('tabtrans_num_heads', 2, 8)
        num_layers = trial.suggest_int('tabtrans_num_layers', 2, 6)
        dropout = trial.suggest_float('tabtrans_dropout', 0.0, 0.3)
        lr = trial.suggest_float('tabtrans_lr', 1e-5, 1e-2, log=True)
        
        return {
            'model_class': TabTransformer,
            'model_kwargs': {
                'embed_dim': embed_dim,
                'num_heads': num_heads,
                'num_layers': num_layers,
                'dropout': dropout
            },
            'lr': lr,
            'device': device
        }
    
    def _estimate_noise_ceiling(self):
        """Estimate the noise ceiling using a simple baseline."""
        # Use a simple ridge regression as baseline for noise ceiling
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        
        # Create simple pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0))
        ])
        
        # Estimate with cross-validation
        scores = cross_val_score(pipeline, self.X_train, self.y_train, 
                               cv=3, scoring='r2', n_jobs=self.n_jobs)
        
        self.noise_ceiling = max(scores.mean() + 2 * scores.std(), 0.99)
        
        print(f"{Colors.CYAN}📊 Estimated noise ceiling: {self.noise_ceiling:.4f}{Colors.END}")
        print(f"   (Ridge baseline: {scores.mean():.4f} ± {scores.std():.4f})")
        
        return self.noise_ceiling 