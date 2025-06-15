#!/usr/bin/env python3
"""
Battle-Tested End-to-End ML Playbook
====================================
Pushes theoretical accuracy limits on datasets using:
- Noise ceiling analysis
- Robust preprocessing pipeline
- Cross-validation with multiple model types
- Strategic Optuna optimization until target R² achieved
"""

# =============================================================================
# DATASET CONFIGURATION - Change this value to switch datasets
# =============================================================================
DATASET = 3  # Set to 1 for Hold-1 (400 samples), 2 for Hold-2 (108 samples), or 3 for Hold-1 Full (401 samples)
# =============================================================================

import pandas as pd
import numpy as np
import time
import joblib
import logging
from pathlib import Path

# Core ML libraries
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Model zoo
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor
from sklearn.neural_network import MLPRegressor

# Metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Optuna for hyperparameter optimization
import optuna
from optuna.pruners import MedianPruner

# Visualization and reporting
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ANSI colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


class KMeansOutlierTransformer(BaseEstimator, TransformerMixin):
    """Remove outliers via K-means clustering"""
    def __init__(self, n_clusters=3, min_cluster_size_ratio=0.1):
        self.n_clusters = n_clusters
        self.min_cluster_size_ratio = min_cluster_size_ratio
        self.kmeans = None
        self.valid_clusters_ = None
        self.mask_ = None
    def fit(self, X, y=None):
        del y
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(X)
        counts = np.bincount(labels)
        min_size = int(len(X) * self.min_cluster_size_ratio)
        self.valid_clusters_ = np.where(counts >= min_size)[0]
        self.mask_ = np.isin(labels, self.valid_clusters_)
        return self
    def transform(self, X):
        labels = self.kmeans.predict(X)
        mask = np.isin(labels, self.valid_clusters_)
        return X[mask]
    def get_support_mask(self):
        return self.mask_

class IsolationForestTransformer(BaseEstimator, TransformerMixin):
    """Remove outliers via Isolation Forest"""
    def __init__(self, contamination=0.1, n_estimators=100, random_state=42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.iforest = None
        self.mask_ = None
    def fit(self, X, y=None):
        del y
        self.iforest = IsolationForest(contamination=self.contamination,
                                       n_estimators=self.n_estimators,
                                       random_state=self.random_state,
                                       n_jobs=-1)
        labels = self.iforest.fit_predict(X)
        self.mask_ = labels != -1
        return self
    def transform(self, X):
        labels = self.iforest.predict(X)
        mask = labels != -1
        return X[mask]
    def get_support_mask(self):
        return self.mask_

class LocalOutlierFactorTransformer(BaseEstimator, TransformerMixin):
    """Remove outliers via Local Outlier Factor"""
    def __init__(self, n_neighbors=20, contamination=0.1):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.lof = None
        self.mask_ = None
    def fit(self, X, y=None):
        del y
        self.lof = LocalOutlierFactor(n_neighbors=min(self.n_neighbors, len(X)-1),
                                      contamination=self.contamination,
                                      novelty=True,
                                      n_jobs=-1)
        labels = self.lof.fit_predict(X)
        self.mask_ = labels != -1
        return self
    def transform(self, X):
        labels = self.lof.predict(X)
        mask = labels != -1
        return X[mask]
    def get_support_mask(self):
        return self.mask_
class BattleTestedOptimizer:
    def __init__(self, dataset_num, target_r2=0.93, max_trials=40, cv_splits=5, cv_repeats=3, use_iforest=False, use_lof=False):
        self.dataset_num = dataset_num
        self.target_r2 = target_r2
        self.max_trials = max_trials
        # Use 5×3 Repeated-KFold as specified in recipe
        self.cv = RepeatedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=42)
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.X_clean = None
        self.X_test_clean = None
        self.noise_ceiling = None
        self.baseline_r2 = None
        self.study = None
        self.best_pipeline = None
        self.preprocessing_pipeline = None
        self.use_iforest = use_iforest
        self.use_lof = use_lof
        
        # Setup logging
        self.setup_logging()
        
        dataset_name = f"Hold-{dataset_num}"
        print(f"{Colors.BOLD}{Colors.CYAN}🚀 Battle-Tested ML Optimizer Initialized for {dataset_name}{Colors.END}")
        print(f"   Target R²: {target_r2}")
        print(f"   Max trials: {max_trials}")
        print(f"   CV strategy: {cv_splits}-fold × {cv_repeats} repeats")

    def setup_logging(self):
        """Setup logging to capture training progress"""
        # Create dataset-specific directory
        self.model_dir = Path(f"best_model_hold{self.dataset_num}")
        self.model_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.model_dir / f'hold{self.dataset_num}_training_log.txt'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def step_1_pin_down_ceiling(self, X, y):
        """Step 1: Pin down the *true* ceiling first – don't chase ghosts"""
        self.logger.info("=" * 60)
        self.logger.info("STEP 1: Pin down the *true* ceiling first")
        self.logger.info("=" * 60)
        
        # Split for final test evaluation (20%) and save indices for reproducibility
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, np.arange(len(X)), test_size=0.2, random_state=42, shuffle=True
        )
        
        # Save the exact test indices to prevent data leakage
        test_indices_path = self.model_dir / f"hold{self.dataset_num}_test_indices.npy"
        train_indices_path = self.model_dir / f"hold{self.dataset_num}_train_indices.npy"
        np.save(test_indices_path, test_idx)
        np.save(train_indices_path, train_idx)
        self.logger.info(f"Test indices saved to: {test_indices_path}")
        self.logger.info(f"Train indices saved to: {train_indices_path}")
        self.logger.info(f"Test samples: {len(test_idx)} (indices: {test_idx[:5] if len(test_idx) >= 5 else test_idx}...{test_idx[-5:] if len(test_idx) >= 5 else test_idx})")
        
        # Store the training data for optimization
        self.X = X_train.copy()
        self.y = y_train.copy()
        self.X_test = X_test.copy()
        self.y_test = y_test.copy()
        
        self.logger.info(f"Dataset shape: {X.shape}")
        self.logger.info(f"   Training: {self.X.shape}")
        self.logger.info(f"   Test (held out): {self.X_test.shape}")
        
        # 5×3 Repeated-KFold RidgeCV as specified in recipe
        print(f"\n🔍 5×3 Repeated-KFold RidgeCV for noise ceiling...")
        baseline_pipe = Pipeline([
            ('scale', RobustScaler()),
            ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 20)))
        ])
        
        try:
            scores = cross_val_score(baseline_pipe, self.X, self.y, cv=self.cv, scoring='r2', n_jobs=12)
            scores = scores[np.isfinite(scores)]  # Remove any inf/nan values
            
            if len(scores) > 0:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                # Noise ceiling = mean + 2·std as specified in recipe
                self.noise_ceiling = mean_score + 2*std_score
                self.baseline_r2 = mean_score
            else:
                self.noise_ceiling = 0.95  # Default reasonable ceiling
                self.baseline_r2 = 0.0
                
        except Exception as e:
            self.logger.error(f"Noise ceiling estimation failed: {e}")
            self.noise_ceiling = 0.95  # Default fallback
            self.baseline_r2 = 0.0
        
        self.logger.info(f"Baseline Ridge R²: {self.baseline_r2:.4f} ± {std_score:.4f}")
        self.logger.info(f"Noise ceiling (mean + 2·std): {self.noise_ceiling:.4f}")
        
        # Ceiling analysis
        if self.target_r2 > self.noise_ceiling:
            self.logger.warning(f"Target R² ({self.target_r2:.3f}) > Estimated ceiling ({self.noise_ceiling:.3f})")
            self.target_r2 = max(0.93, self.noise_ceiling - 0.02)  # Keep at least 0.93
            self.logger.info(f"Adjusting target to {self.target_r2:.3f}")
        else:
            self.logger.info(f"Target R² ({self.target_r2:.3f}) is achievable (ceiling: {self.noise_ceiling:.3f})")
        
        return self.noise_ceiling, self.baseline_r2

    def step_2_bulletproof_preprocessing(self):
        """Step 2: Bullet-proof preprocessing (5 lines) as per recipe"""
        self.logger.info("\nSTEP 2: Bullet-proof preprocessing")
        self.logger.info("=" * 40)
        
        # Exact preprocessing as specified in recipe
        self.preprocessing_pipeline = Pipeline([
            ('var', VarianceThreshold(1e-8)),       # kills the 253 all-zero cols
            ('scale', RobustScaler(quantile_range=(5,95)))  # less sensitive than StandardScaler
        ])
        
        self.logger.info("Applying preprocessing pipeline:")
        self.logger.info("  - VarianceThreshold(1e-8)")
        self.logger.info("  - RobustScaler(quantile_range=(5,95))")
        
        # Apply to training data
        initial_features = self.X.shape[1]
        self.X_clean = self.preprocessing_pipeline.fit_transform(self.X)
        
        # Apply to test data
        self.X_test_clean = self.preprocessing_pipeline.transform(self.X_test)
        # Outlier removal using K-means and optional methods
        initial_samples = self.X_clean.shape[0]
        km_filter = KMeansOutlierTransformer(n_clusters=3, min_cluster_size_ratio=0.1)
        km_filter.fit(self.X_clean)
        mask = km_filter.get_support_mask()
        self.X_clean = km_filter.transform(self.X_clean)
        self.y = self.y[mask]
        removed = initial_samples - self.X_clean.shape[0]
        if removed:
            self.logger.info(f"KMeans removed {removed} outliers")

        if self.use_iforest:
            initial_samples = self.X_clean.shape[0]
            if_filter = IsolationForestTransformer(contamination=0.1, n_estimators=100)
            if_filter.fit(self.X_clean)
            mask = if_filter.get_support_mask()
            self.X_clean = if_filter.transform(self.X_clean)
            self.y = self.y[mask]
            removed = initial_samples - self.X_clean.shape[0]
            if removed:
                self.logger.info(f"IsolationForest removed {removed} outliers")

        if self.use_lof:
            initial_samples = self.X_clean.shape[0]
            lof_filter = LocalOutlierFactorTransformer(n_neighbors=20, contamination=0.1)
            lof_filter.fit(self.X_clean)
            mask = lof_filter.get_support_mask()
            self.X_clean = lof_filter.transform(self.X_clean)
            self.y = self.y[mask]
            removed = initial_samples - self.X_clean.shape[0]
            if removed:
                self.logger.info(f"LocalOutlierFactor removed {removed} outliers")
        
        removed_features = initial_features - self.X_clean.shape[1]
        self.logger.info(f"Features before: {initial_features}")
        self.logger.info(f"Features after: {self.X_clean.shape[1]}")
        self.logger.info(f"Removed: {removed_features} zero-variance features")
        
        return self.X_clean, self.X_test_clean

    def make_model(self, trial):
        """Create model based on type and Optuna trial suggestions - exact recipe"""
        mdl = trial.suggest_categorical('mdl', ['ridge','elastic','svr','gbrt','mlp'])
        
        if mdl == 'ridge':
            return Ridge(alpha=trial.suggest_float('α', 1e-3, 10, log=True))
        
        elif mdl == 'elastic':
            return ElasticNet(
                alpha=trial.suggest_float('α', 1e-4, 1, log=True),
                l1_ratio=trial.suggest_float('l1', 0.0, 1.0),
                max_iter=2000
            )
        
        elif mdl == 'svr':
            return SVR(
                C=trial.suggest_float('C', 0.1, 100, log=True),
                gamma=trial.suggest_float('γ', 1e-3, 1, log=True),
                kernel='rbf'
            )
        
        elif mdl == 'gbrt':
            return GradientBoostingRegressor(
                n_estimators=trial.suggest_int('n', 30, 200),
                max_depth=trial.suggest_int('d', 2, 4),
                learning_rate=trial.suggest_float('lr', 0.01, 0.2),
                random_state=42
            )
        
        elif mdl == 'mlp':
            return MLPRegressor(
                hidden_layer_sizes=trial.suggest_categorical(
                    'arch', [(256,), (128,64), (128,128,64)]),
                alpha=trial.suggest_float('l2', 1e-5, 1e-2, log=True),
                learning_rate_init=trial.suggest_float('lr', 1e-4, 1e-2, log=True),
                max_iter=800,
                early_stopping=True,
                random_state=42
            )
        
        else:
            raise ValueError(f"Unknown model type: {mdl}")

    def objective(self, trial):
        """Optuna objective function - exact recipe implementation"""
        try:
            # Pipeline for every trial as specified in recipe
            steps = [
                ('scale', RobustScaler()),
                ('reduce', SelectKBest(
                    mutual_info_regression,
                    k=trial.suggest_int('k', 10, 500) if self.X_clean.shape[1] > 10000 else trial.suggest_int('k', 10, 100)
                )),
                ('mdl', self.make_model(trial))
            ]
            
            pipe = Pipeline(steps)
            
            # Cross-validation scoring
            scores = cross_val_score(pipe, self.X_clean, self.y, cv=self.cv, scoring='r2', n_jobs=12)
            scores = scores[np.isfinite(scores)]  # Remove any inf/nan values
            
            if len(scores) == 0:
                raise optuna.exceptions.TrialPruned()
                
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Early failure safety net
            if mean_score < -2.0:  # Model performing much worse than predicting mean
                raise optuna.exceptions.TrialPruned()
            
            # Store additional metrics and pipeline
            trial.set_user_attr('mean_r2', mean_score)
            trial.set_user_attr('std_r2', std_score)
            trial.set_user_attr('model_type', trial.params['mdl'])
            trial.set_user_attr('pipeline', pipe)  # Store the pipeline
            
            # Progress tracking
            if trial.number % 5 == 0:
                msg = f"Trial {trial.number:3d}: {trial.params['mdl']:7s} R² = {mean_score:.4f} ± {std_score:.4f}"
                print(msg)
                self.logger.info(msg)
            
            return mean_score
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            raise optuna.exceptions.TrialPruned()

    def step_3_optuna_search(self):
        """Step 3: Optuna search space as specified in recipe"""
        self.logger.info("\nSTEP 3: Optuna search space optimization")
        self.logger.info("=" * 45)
        self.logger.info(f"Target R²: {self.target_r2}")
        self.logger.info(f"Max trials: {self.max_trials}")
        self.logger.info("Search space: Ridge | ElasticNet | SVR | GBRT | MLP")
        
        # Create study with MedianPruner as specified
        self.study = optuna.create_study(
            direction='maximize',
            pruner=MedianPruner(n_startup_trials=5),
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        start_time = time.time()
        
        def progress_callback(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                if study.best_value >= self.target_r2:
                    elapsed = time.time() - start_time
                    msg = f"🎉 TARGET ACHIEVED! R² = {study.best_value:.4f} >= {self.target_r2} in {trial.number} trials ({elapsed:.1f}s)"
                    print(f"\n{Colors.GREEN}{msg}{Colors.END}")
                    self.logger.info(msg)
                    study.stop()
        
        try:
            self.study.optimize(
                self.objective,
                n_trials=self.max_trials,
                callbacks=[progress_callback]
            )
        except KeyboardInterrupt:
            self.logger.info("Optimization interrupted by user")
        
        # Results summary
        if len(self.study.trials) > 0:
            msg = f"Best R²: {self.study.best_value:.4f}"
            print(f"\n📊 {msg}")
            self.logger.info(msg)
            self.logger.info(f"Best model: {self.study.best_trial.user_attrs.get('model_type', 'N/A')}")
            self.logger.info(f"Best params: {self.study.best_params}")
            
            return self.study.best_value >= self.target_r2
        else:
            return False

    def step_4_lock_in_champion(self):
        """Step 4: Lock in and export the champion"""
        self.logger.info("\nSTEP 4: Lock in and export the champion")
        self.logger.info("=" * 42)
        
        if self.study is None or len(self.study.trials) == 0:
            self.logger.error("No optimization results available")
            return None
        
        # Get best trial and pipeline
        best_trial = self.study.best_trial
        self.best_pipeline = best_trial.user_attrs['pipeline']
        
        self.logger.info("Fitting best pipeline on full training data...")
        # Fit on training samples
        self.best_pipeline.fit(self.X_clean, self.y)
        
        # Save the best model
        model_path = self.model_dir / f"hold{self.dataset_num}_best_model.pkl"
        joblib.dump(self.best_pipeline, model_path)
        self.logger.info(f"Best model saved to: {model_path}")
        
        # Also save preprocessing pipeline separately
        preprocessing_path = self.model_dir / f"hold{self.dataset_num}_preprocessing_pipeline.pkl"
        joblib.dump(self.preprocessing_pipeline, preprocessing_path)
        self.logger.info(f"Preprocessing pipeline saved to: {preprocessing_path}")
        
        return self.best_pipeline

    def step_5_final_evaluation(self):
        """Step 5: Final evaluation on held-out test set"""
        self.logger.info("\nSTEP 5: Final evaluation on held-out test set")
        self.logger.info("=" * 47)
        
        if self.best_pipeline is None:
            self.logger.error("No trained model available")
            return None
        
        # Evaluate on the proper held-out test set
        self.logger.info(f"Evaluating on {len(self.y_test)}-sample hold-out test...")
        y_pred = self.best_pipeline.predict(self.X_test_clean)
        
        r2_test = r2_score(self.y_test, y_pred)
        mae_test = mean_absolute_error(self.y_test, y_pred)
        rmse_test = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
        self.logger.info(f"Test R²: {r2_test:.4f}")
        self.logger.info(f"Test MAE: {mae_test:.6f}")
        self.logger.info(f"Test RMSE: {rmse_test:.6f}")
        self.logger.info(f"Baseline R² (CV): {self.baseline_r2:.4f}")
        
        # Success criteria
        beats_baseline = r2_test > self.baseline_r2
        near_ceiling = abs(r2_test - self.noise_ceiling) < 0.05
        
        self.logger.info("Model Assessment:")
        self.logger.info(f"  Beats baseline: {'✅ YES' if beats_baseline else '❌ NO'}")
        self.logger.info(f"  Near ceiling: {'✅ YES' if near_ceiling else '❌ NO'}")
        
        if beats_baseline and near_ceiling:
            msg = "🏆 MODEL ACCEPTED - At theoretical maximum!"
            print(f"{Colors.GREEN}{Colors.BOLD}{msg}{Colors.END}")
            self.logger.info(msg)
        
        # Save results
        results = {
            'test_r2': r2_test,
            'test_mae': mae_test,
            'test_rmse': rmse_test,
            'baseline_r2': self.baseline_r2,
            'noise_ceiling': self.noise_ceiling,
            'beats_baseline': beats_baseline,
            'near_ceiling': near_ceiling,
            'best_model_type': self.study.best_trial.user_attrs['model_type'],
            'best_params': self.study.best_params
        }
        
        # Save results to file
        results_path = self.model_dir / f"hold{self.dataset_num}_evaluation_results.txt"
        with open(results_path, 'w') as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        
        return results

    def create_diagnostic_plots(self):
        """Create comprehensive diagnostic plots"""
        if self.best_pipeline is None:
            self.logger.warning("No trained model available for plots")
            return
        
        self.logger.info("Creating diagnostic plots...")
        
        # Predictions on test set
        y_pred = self.best_pipeline.predict(self.X_test_clean)
        
        # Create comprehensive plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Hold-{self.dataset_num} Model: Final Evaluation', fontsize=16, fontweight='bold')
        
        # 1. Predicted vs Actual
        ax1.scatter(self.y_test, y_pred, alpha=0.6, color='blue')
        min_val = min(min(self.y_test), min(y_pred))
        max_val = max(max(self.y_test), max(y_pred))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Predicted vs Actual (Test Set)')
        ax1.grid(True, alpha=0.3)
        
        r2_test = r2_score(self.y_test, y_pred)
        ax1.text(0.05, 0.95, f'R² = {r2_test:.4f}', transform=ax1.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 2. Residuals
        residuals = self.y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, color='green')
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)
        
        # 3. Optimization progress
        if self.study and len(self.study.trials) > 0:
            trial_numbers = [t.number for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            trial_values = [t.value for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            
            if trial_numbers:
                ax3.plot(trial_numbers, trial_values, 'b-', alpha=0.6)
                ax3.axhline(y=self.target_r2, color='r', linestyle='--', 
                           label=f'Target R² = {self.target_r2}')
                ax3.axhline(y=self.noise_ceiling, color='orange', linestyle=':', 
                           label=f'Noise Ceiling = {self.noise_ceiling:.3f}')
                ax3.set_xlabel('Trial Number')
                ax3.set_ylabel('CV R² Score')
                ax3.set_title('Optimization Progress')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                # Auto-scale y-axis to fit the data properly
                if trial_values:
                    min_val = min(min(trial_values), self.baseline_r2 - 0.1)
                    max_val = max(max(trial_values), self.noise_ceiling + 0.05)
                    ax3.set_ylim(min_val, max_val)
        
        # 4. Model comparison
        if self.study:
            model_types = []
            model_scores = []
            for trial in self.study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    model_type = trial.user_attrs.get('model_type', 'unknown')
                    model_types.append(model_type)
                    model_scores.append(trial.value)
            
            if model_types:
                df_models = pd.DataFrame({'model': model_types, 'r2': model_scores})
                model_summary = df_models.groupby('model')['r2'].agg(['mean', 'std', 'count'])
                
                models = model_summary.index
                means = model_summary['mean']
                stds = model_summary['std']
                
                ax4.bar(models, means, yerr=stds, capsize=5, alpha=0.7)
                ax4.axhline(y=self.baseline_r2, color='r', linestyle='--', 
                           label=f'Baseline = {self.baseline_r2:.3f}')
                ax4.set_ylabel('CV R² Score')
                ax4.set_title('Model Type Comparison')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.model_dir / f"hold{self.dataset_num}_diagnostic_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Diagnostic plots saved to: {plot_file}")

def main():
    """Main execution function following the exact recipe"""
    print(f"{Colors.BOLD}{Colors.CYAN}")
    print("=" * 70)
    print("🚀 BATTLE-TESTED END-TO-END ML PLAYBOOK")
    dataset_name = f"Hold-{DATASET}" if DATASET != 3 else "Hold-1 Full"
    print(f"   Running on {dataset_name} dataset")
    print("=" * 70)
    print(f"{Colors.END}")
    
    try:
        # Load data based on dataset selection
        if DATASET == 1:
            print(f"\n{Colors.BOLD}📁 Loading Hold-1 Data{Colors.END}")
            X = pd.read_csv('Predictors_Hold-1_2025-04-14_18-28.csv', header=None).values.astype(np.float32)
            y = pd.read_csv('9_10_24_Hold_01_targets.csv', header=None).values.astype(np.float32).ravel()
        elif DATASET == 2:
            print(f"\n{Colors.BOLD}📁 Loading Hold-2 Data{Colors.END}")
            X = pd.read_csv('hold2_predictor.csv', header=None).values.astype(np.float32)
            y = pd.read_csv('hold2_target.csv', header=None).values.astype(np.float32).ravel()
        elif DATASET == 3:
            print(f"\n{Colors.BOLD}📁 Loading Hold-1 Full Data{Colors.END}")
            X = pd.read_csv('predictors_Hold 1 Full_20250527_151252.csv', header=None).values.astype(np.float32)
            y = pd.read_csv('targets_Hold 1 Full_20250527_151252.csv', header=None).values.astype(np.float32).ravel()
        else:
            raise ValueError(f"Invalid DATASET value: {DATASET}. Must be 1, 2, or 3.")
        
        print(f"✅ Data loaded: {X.shape} features, {len(y)} samples")
        
        # Initialize optimizer with same parameters for both datasets
        optimizer = BattleTestedOptimizer(DATASET, target_r2=0.93, max_trials=40, use_iforest=True, use_lof=True)
        
        # Execute the exact recipe
        print(f"\n{Colors.BOLD}{Colors.GREEN}🎯 EXECUTING EXACT RECIPE{Colors.END}")
        
        # Step 1: Pin down the ceiling
        noise_ceiling, baseline_r2 = optimizer.step_1_pin_down_ceiling(X, y)
        
        # Step 2: Bullet-proof preprocessing
        X_clean, X_test_clean = optimizer.step_2_bulletproof_preprocessing()
        
        # Step 3: Optuna search
        success = optimizer.step_3_optuna_search()
        
        # Step 4: Lock in and export champion
        best_pipeline = optimizer.step_4_lock_in_champion()
        
        # Step 5: Final evaluation
        results = optimizer.step_5_final_evaluation()
        
        # Create diagnostic plots
        optimizer.create_diagnostic_plots()
        
        # Final summary
        print(f"\n{Colors.BOLD}{Colors.CYAN}📋 FINAL SUMMARY{Colors.END}")
        print("=" * 50)
        if optimizer.study and len(optimizer.study.trials) > 0:
            print(f"🏆 Best achieved R²: {optimizer.study.best_value:.4f}")
        print(f"📏 Noise ceiling: {noise_ceiling:.4f}")
        print(f"🎯 Target achieved: {'✅ YES' if success else '❌ NO'}")
        if results:
            print(f"🧪 Test set R²: {results['test_r2']:.4f}")
            print(f"🏗️ Best model: {results['best_model_type']}")
        
        print(f"\n📁 All outputs saved to: {Colors.BOLD}best_model_hold{DATASET}/{Colors.END}")
        print(f"   - hold{DATASET}_best_model.pkl (trained model)")
        print(f"   - hold{DATASET}_preprocessing_pipeline.pkl (preprocessing)")
        print(f"   - hold{DATASET}_training_log.txt (complete log)")
        print(f"   - hold{DATASET}_evaluation_results.txt (test results)")
        print(f"   - hold{DATASET}_diagnostic_plots.png (visualizations)")
        print(f"   - hold{DATASET}_test_indices.npy (test data indices)")
        print(f"   - hold{DATASET}_train_indices.npy (train data indices)")
        
        return results
        
    except FileNotFoundError as e:
        print(f"{Colors.RED}❌ Data files not found: {e}{Colors.END}")
        print("Make sure 'hold2_predictor.csv' and 'hold2_target.csv' exist")
        return None
    except Exception as e:
        print(f"{Colors.RED}❌ Error: {e}{Colors.END}")
        return None

if __name__ == "__main__":
    results = main() 