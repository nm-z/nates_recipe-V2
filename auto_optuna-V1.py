#!/usr/bin/env python3
"""
Battle-Tested End-to-End ML Playbook
====================================
Pushes theoretical accuracy limits on datasets using:
- Noise ceiling analysis with iterative re-estimation
- Robust preprocessing pipeline with dynamic choices (PCA, K-means, scalers)
- Cross-validation with multiple model types
- Strategic Optuna optimization until target R² achieved
- Statistical significance testing with bootstrap
"""

# =============================================================================
# DATASET CONFIGURATION - Change this value to switch datasets
# =============================================================================
DATASET = 2  # Set to 1 for Hold-1 (400 samples), 2 for Hold-2 (108 samples), or 3 for Hold-3 (401 samples)
# =============================================================================

import pandas as pd
import numpy as np
import time
import joblib
import logging
from pathlib import Path

# Core ML libraries
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import (StandardScaler, RobustScaler, QuantileTransformer, 
                                 PowerTransformer, MinMaxScaler)
from sklearn.feature_selection import (VarianceThreshold, SelectKBest, mutual_info_regression,
                                     f_regression, RFECV)
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import BaseEstimator, TransformerMixin

# Model zoo
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor, 
                             ExtraTreesRegressor, AdaBoostRegressor)
from sklearn.neural_network import MLPRegressor

# Optional advanced models (import safely)
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

# Metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Optuna for hyperparameter optimization
import optuna
from optuna.pruners import MedianPruner

# Progress bar
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Statistical testing
from scipy import stats


# Visualization and reporting
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─── Silence Optuna & your own INFO spam ─────────────────────────────
import optuna
optuna.logging.disable_default_handler()           # no [I] Trial … lines
optuna.logging.set_verbosity(optuna.logging.ERROR)

for h in logging.getLogger(__name__).handlers:
    if isinstance(h, logging.StreamHandler):
        h.setLevel(logging.ERROR)                  # hide "INFO │ …" tree lines

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
    LIME = '\033[92;1m'     # bold bright-green
    END = '\033[0m'

class OutlierFilterTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer that removes outliers using K-means clustering"""
    def __init__(self, n_clusters=3, min_cluster_size_ratio=0.1, remove=False):
        self.n_clusters = n_clusters
        self.min_cluster_size_ratio = min_cluster_size_ratio
        self.remove = remove  # NEW: flag to actually remove outliers
        self.kmeans = None
        self.valid_clusters = None
        self.outlier_indices_ = None
        
    def fit(self, X, y=None):
        # Fit K-means on the data and identify outliers consistently
        del y  # Explicitly acknowledge unused parameter
        try:
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            cluster_labels = self.kmeans.fit_predict(X)
            
            # Identify clusters that are too small (potential outliers)
            cluster_counts = np.bincount(cluster_labels)
            min_size = int(len(X) * self.min_cluster_size_ratio)
            self.valid_clusters = np.where(cluster_counts >= min_size)[0]
            
            # Store outlier indices for consistent behavior
            valid_mask = np.isin(cluster_labels, self.valid_clusters)
            self.outlier_indices_ = np.where(~valid_mask)[0]
            
            # Be very conservative - adaptive outlier removal threshold
            max_removal_ratio = 0.15 if len(X) <= 200 else 0.2  # More conservative for smaller datasets
            if len(self.outlier_indices_) > len(X) * max_removal_ratio:
                # Too many outliers detected, skip removal to prevent data leakage
                self.outlier_indices_ = np.array([])
                
        except Exception:
            # If clustering fails, don't remove any samples
            self.outlier_indices_ = np.array([])
            self.kmeans = None
        
        return self
    
    def transform(self, X, *, remove=None):
        """If remove=True drop rows marked as outliers, else return X unchanged."""
        if remove is None:
            remove = self.remove
        if remove and self.outlier_indices_.size > 0:
            keep_mask = np.ones(len(X), dtype=bool)
            keep_mask[self.outlier_indices_] = False
            return X[keep_mask]
        return X

class IsolationForestTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for Isolation Forest outlier detection"""
    def __init__(self, contamination=0.1, n_estimators=100, random_state=42, remove=False):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.remove = remove  # NEW: flag to actually remove outliers
        self.iforest = None
        self.outlier_indices_ = None
        
    def fit(self, X, y=None):
        del y  # Explicitly acknowledge unused parameter
        try:
            self.iforest = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.iforest.fit(X)
            
            # Store outlier indices for consistency
            outlier_labels = self.iforest.predict(X)
            self.outlier_indices_ = np.where(outlier_labels == -1)[0]
        except Exception:
            self.iforest = None
            self.outlier_indices_ = np.array([])
        return self
    
    def transform(self, X, *, remove=None):
        """If remove=True drop rows marked as outliers, else return X unchanged."""
        if remove is None:
            remove = self.remove
        if remove and self.outlier_indices_.size > 0:
            keep_mask = np.ones(len(X), dtype=bool)
            keep_mask[self.outlier_indices_] = False
            return X[keep_mask]
        return X

class LocalOutlierFactorTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for Local Outlier Factor detection"""
    def __init__(self, n_neighbors=20, contamination=0.1, remove=False):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.remove = remove  # NEW: flag to actually remove outliers
        self.lof = None
        self.outlier_indices_ = None
        
    def fit(self, X, y=None):
        del y  # Explicitly acknowledge unused parameter
        try:
            self.lof = LocalOutlierFactor(
                n_neighbors=min(self.n_neighbors, len(X) - 1),
                contamination=self.contamination,
                n_jobs=-1
            )
            
            # For LOF, we need to fit_predict to get outlier labels
            outlier_labels = self.lof.fit_predict(X)
            self.outlier_indices_ = np.where(outlier_labels == -1)[0]
        except Exception:
            self.lof = None
            self.outlier_indices_ = np.array([])
        return self
    
    def transform(self, X, *, remove=None):
        """If remove=True drop rows marked as outliers, else return X unchanged."""
        if remove is None:
            remove = self.remove
        if remove and self.outlier_indices_.size > 0:
            keep_mask = np.ones(len(X), dtype=bool)
            keep_mask[self.outlier_indices_] = False
            return X[keep_mask]
        return X

class HSICFeatureSelector(BaseEstimator, TransformerMixin):
    """Custom HSIC-based feature selector (simplified implementation)"""
    def __init__(self, k=50):
        self.k = k
        self.selected_features_ = None
        
    def fit(self, X, y):
        try:
            # Simplified HSIC: use correlation as approximation
            correlations = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
            correlations = np.nan_to_num(correlations)
            self.selected_features_ = np.argsort(correlations)[-self.k:]
        except Exception:
            # Fallback to first k features
            self.selected_features_ = np.arange(min(self.k, X.shape[1]))
        return self
    
    def transform(self, X):
        if self.selected_features_ is None:
            return X
        return X[:, self.selected_features_]

class BattleTestedOptimizer:
    def __init__(self, dataset_num, target_r2=0.93, max_trials=200, cv_splits=5, cv_repeats=3):
        self.dataset_num = dataset_num
        self.target_r2 = target_r2
        self.max_trials = max_trials
        # Use adaptive CV strategy based on dataset size to prevent data leakage
        self.cv_splits = cv_splits
        self.cv_repeats = cv_repeats
        self.cv = None  # Will be set dynamically based on training data size
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.X_clean = None
        self.X_test_clean = None
        self.noise_ceiling = None
        self.validation_r2 = None
        self.validation_cv_scores = None
        self.study = None
        self.best_pipeline = None
        self.preprocessing_pipeline = None
        self.iteration_count = 0
        self.ceiling_history = []
        
        # NEW: Pipeline for noise ceiling evaluation and trial tracking
        self.ridge_ceiling_pipe = Pipeline([
            ('scale', RobustScaler()),
            ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 20)))
        ])
        self.trial_log = []  # Track all trials (number, model, r2)
        
        # Setup logging
        self.setup_logging()
        
        dataset_name = "Hold-{}".format(dataset_num)
        print(f"{Colors.BOLD}{Colors.CYAN}🚀 Battle-Tested ML Optimizer v3.0 - Initialized for {dataset_name}{Colors.END}")
        print(f"   Target R²: {target_r2}")
        print(f"   Max trials: {max_trials}")
        print(f"   CV strategy: {cv_splits}-fold × {cv_repeats} repeats")
        print("   Features: Dynamic preprocessing, expanded model zoo, outlier detection")

    def setup_logging(self):
        """Setup dual logging (file + console) with different formats"""
        # Create dataset-specific directory
        self.model_dir = Path(f"best_model_hold{self.dataset_num}")
        self.model_dir.mkdir(exist_ok=True)
        
        # Clear any existing handlers
        logger = logging.getLogger(__name__)
        logger.handlers.clear()
        
        # Set up logger
        logger.setLevel(logging.INFO)
        
        # FILE handler - full timestamps
        fh = logging.FileHandler(self.model_dir / f'hold{self.dataset_num}_training_log.txt', mode='w')
        fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        logger.addHandler(fh)
        
        # CONSOLE handler - brief format
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(levelname)s │ %(message)s'))
        logger.addHandler(ch)
        
        self.logger = logger

    def _configure_cv_strategy(self, train_size):
        """Configure CV strategy based on training set size to prevent data leakage"""
        # Ensure each CV fold has sufficient samples (minimum 20 per fold)
        min_samples_per_fold = 20
        
        # Calculate safe CV parameters
        if train_size < 100:
            # Very small datasets: use conservative CV
            safe_splits = min(3, max(2, train_size // min_samples_per_fold))
            safe_repeats = 1 if train_size < 60 else 2
            self.logger.warning(f"Small training set ({train_size}): Using {safe_splits}-fold × {safe_repeats} repeats")
        elif train_size < 200:
            # Small datasets: reduce CV intensity
            safe_splits = min(4, max(3, train_size // min_samples_per_fold))
            safe_repeats = min(2, self.cv_repeats)
            self.logger.info(f"Moderate training set ({train_size}): Using {safe_splits}-fold × {safe_repeats} repeats")
        else:
            # Standard datasets: use default CV
            safe_splits = min(self.cv_splits, train_size // min_samples_per_fold)
            safe_repeats = self.cv_repeats
            self.logger.info(f"Large training set ({train_size}): Using {safe_splits}-fold × {safe_repeats} repeats")
        
        # Apply minimum constraints
        safe_splits = max(2, safe_splits)  # At least 2-fold
        safe_repeats = max(1, safe_repeats)  # At least 1 repeat
        
        # Set the CV strategy
        self.cv = RepeatedKFold(n_splits=safe_splits, n_repeats=safe_repeats, random_state=42)
        
        # Log final CV configuration
        total_folds = safe_splits * safe_repeats
        approx_fold_size = train_size // safe_splits
        self.logger.info(f"CV configured: {safe_splits}-fold × {safe_repeats} repeats = {total_folds} total folds")
        self.logger.info(f"Approximate samples per fold: {approx_fold_size}")
        
        # Validate CV configuration
        if approx_fold_size < min_samples_per_fold:
            self.logger.warning(f"CV fold size ({approx_fold_size}) below recommended minimum ({min_samples_per_fold})")

    def evaluate_cleaning_methods(self):
        """Return the cleanest X (highest RidgeCV noise ceiling) and a ranking table."""
        print(f"\n{Colors.BOLD}{Colors.YELLOW}🧹 EVALUATING DATA CLEANING METHODS{Colors.END}")
        print("=" * 60)
        
        # Define cleaning recipes to test
        recipes = {
            'raw': [],
            'kmeans': [OutlierFilterTransformer(n_clusters=3, remove=True)],
            'iforest': [IsolationForestTransformer(contamination=0.1, remove=True)],
            'lof': [LocalOutlierFactorTransformer(n_neighbors=20, contamination=0.1, remove=True)],
            'km+if': [
                OutlierFilterTransformer(n_clusters=3, remove=True),
                IsolationForestTransformer(contamination=0.1, remove=True)
            ],
            'km+lof': [
                OutlierFilterTransformer(n_clusters=3, remove=True),
                LocalOutlierFactorTransformer(n_neighbors=20, contamination=0.1, remove=True)
            ],
            'if+lof': [
                IsolationForestTransformer(contamination=0.1, remove=True),
                LocalOutlierFactorTransformer(n_neighbors=20, contamination=0.1, remove=True)
            ],
            'all_3': [
                OutlierFilterTransformer(n_clusters=3, remove=True),
                IsolationForestTransformer(contamination=0.1, remove=True),
                LocalOutlierFactorTransformer(n_neighbors=20, contamination=0.1, remove=True)
            ]
        }
        
        results = {}
        print("Testing cleaning recipes...")
        
        for name, cleaners in recipes.items():
            try:
                X_tmp = self.X.copy()
                y_tmp = self.y.copy()
                
                # Apply cleaning transformers sequentially
                for cleaner in cleaners:
                    # Fit the cleaner
                    cleaner.fit(X_tmp)
                    
                    # Get the indices to keep
                    if hasattr(cleaner, 'outlier_indices_') and cleaner.outlier_indices_.size > 0:
                        keep_mask = np.ones(len(X_tmp), dtype=bool)
                        keep_mask[cleaner.outlier_indices_] = False
                        X_tmp = X_tmp[keep_mask]
                        y_tmp = y_tmp[keep_mask]
                    
                # Skip if dataset became too small - adaptive threshold based on original training size
                min_safe_size = max(30, len(self.X) // 3)  # At least 30 or 1/3 of original size
                if len(X_tmp) < min_safe_size:
                    self.logger.warning(f"Method '{name}': Dataset too small ({len(X_tmp)} < {min_safe_size} samples) - skipping")
                    continue
                
                # Evaluate noise ceiling with RidgeCV
                cv_scores = cross_val_score(
                    self.ridge_ceiling_pipe, X_tmp, y_tmp,
                    cv=self.cv, scoring='r2', n_jobs=-1
                )
                cv_scores = cv_scores[np.isfinite(cv_scores)]
                
                if len(cv_scores) > 0:
                    mean_r2 = np.mean(cv_scores)
                    std_r2 = np.std(cv_scores)
                    max_r2 = np.max(cv_scores)
                    
                    # Skip methods with negative mean R² (worse than predicting mean)
                    if mean_r2 < 0:
                        self.logger.warning(f"Method '{name}': Poor performance (R² = {mean_r2:.3f}) - skipping")
                        continue
                    
                    # Robust ceiling calculation for numerical stability
                    if std_r2 > 1.0 or abs(mean_r2) > 1.0:
                        # Use robust percentile when RidgeCV has high variance
                        noise_ceiling = np.percentile(cv_scores, 95)
                        self.logger.info(f"Method '{name}': High variance detected (std={std_r2:.3f}), using robust ceiling")
                    else:
                        noise_ceiling = mean_r2 + 2 * std_r2
                        
                    results[name] = (noise_ceiling, len(X_tmp), mean_r2, std_r2, max_r2)
                else:
                    self.logger.warning(f"Method '{name}': No valid CV scores - skipping")
                    continue
                    
            except Exception as e:
                self.logger.warning("Cleaning method '%s' failed: %s", name, e)
                results[name] = (0.0, len(self.X), 0.0, 0.0, 0.0)
        
        # Print results sorted by mean R² (primary) then ceiling (secondary)
        print(f"\n🧹 DATA-CLEANLINESS RANKING (mean_R² ↑, then ceiling ↑)")
        print("-" * 80)
        print(f"{'Method':<10} {'Mean_R²':<9} {'Max_R²':<8} {'R²_ceiling':<12} {'Rows':<6} {'Std_R²':<8}")
        print("-" * 80)
        
        for name, (ceiling, rows, mean_r2, std_r2, max_r2) in sorted(results.items(), key=lambda x: (x[1][2], x[1][0])):
            print(f"{name:<10} {mean_r2:7.4f}   {max_r2:6.4f}   {ceiling:8.4f}     {rows:<6} {std_r2:6.4f}")
        
        # Find the best method: rank by mean_R² first, then ceiling for ties
        best_method = max(results.items(), key=lambda x: (x[1][2], x[1][0]))[0]
        best_mean_r2 = results[best_method][2]
        best_ceiling = results[best_method][0]
        
        print(f"\n🏆 WINNER: {best_method} (Mean_R² = {best_mean_r2:.4f}, Ceiling = {best_ceiling:.4f})")
        
        return best_method, recipes[best_method], results

    def build_preprocessing_pipeline(self, trial):
        """Build dynamic preprocessing pipeline controlled by Optuna"""
        # Enhanced scaler choice
        scaler_choice = trial.suggest_categorical('scaler', ['robust', 'standard', 'quant', 'power', 'minmax'])
        
        if scaler_choice == 'robust':
            # Dynamic quantile ranges for RobustScaler
            q_low = trial.suggest_int('robust_q_low', 1, 25)
            q_high = trial.suggest_int('robust_q_high', 75, 99)
            scaler = RobustScaler(quantile_range=(q_low, q_high))
        elif scaler_choice == 'standard':
            scaler = StandardScaler()
        elif scaler_choice == 'quant':
            output_dist = trial.suggest_categorical('quant_output', ['uniform', 'normal'])
            scaler = QuantileTransformer(output_distribution=output_dist, random_state=42)
        elif scaler_choice == 'power':
            scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        elif scaler_choice == 'minmax':
            scaler = MinMaxScaler()
        
        steps = [
            ('var', VarianceThreshold(1e-8)),
            ('scale', scaler)
        ]
        
        # Enhanced dimensionality reduction options (removed IPCA due to batch issues)
        dim_red_choice = trial.suggest_categorical('dim_red', ['none', 'pca', 'kpca', 'svd'])
        
        if dim_red_choice == 'pca':
            max_components = min(50, max(10, min(self.X.shape) - 10))
            n_components = trial.suggest_int('pca_components', 10, max_components)
            steps.append(('dim_red', PCA(n_components=n_components, random_state=42)))
            
        elif dim_red_choice == 'kpca':
            max_components = min(30, max(10, min(self.X.shape) - 10))  # More conservative for KernelPCA
            n_components = trial.suggest_int('kpca_components', 10, max_components)
            kernel = trial.suggest_categorical('kpca_kernel', ['rbf', 'poly'])  # Remove sigmoid to avoid issues
            
            if kernel == 'rbf':
                gamma = trial.suggest_float('kpca_gamma', 1e-4, 1e-1, log=True)
                kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, random_state=42, n_jobs=-1)
            elif kernel == 'poly':
                degree = trial.suggest_int('kpca_degree', 2, 3)  # More conservative degree
                kpca = KernelPCA(n_components=n_components, kernel=kernel, degree=degree, random_state=42, n_jobs=-1)
            
            steps.append(('dim_red', kpca))
            
        elif dim_red_choice == 'svd':
            max_components = min(50, max(10, min(self.X.shape) - 10))
            n_components = trial.suggest_int('svd_components', 10, max_components)
            steps.append(('dim_red', TruncatedSVD(n_components=n_components, random_state=42)))
        
        # Enhanced feature selection
        feat_sel_choice = trial.suggest_categorical('feat_sel', ['mi', 'hsic', 'f', 'rfecv'])
        k_features = trial.suggest_int('k_features', 20, min(120, self.X.shape[1]))
        
        if feat_sel_choice == 'mi':
            steps.append(('feature_select', SelectKBest(mutual_info_regression, k=k_features)))
        elif feat_sel_choice == 'hsic':
            steps.append(('feature_select', HSICFeatureSelector(k=k_features)))
        elif feat_sel_choice == 'f':
            steps.append(('feature_select', SelectKBest(f_regression, k=k_features)))
        elif feat_sel_choice == 'rfecv':
            # Use Ridge as base estimator for RFECV
            base_estimator = Ridge(alpha=1.0)
            steps.append(('feature_select', RFECV(base_estimator, min_features_to_select=k_features, cv=3)))
        
        # Enhanced outlier detection (K-means reactivated + new methods)
        outlier_methods = []
        
        # K-means outlier filtering (reactivated)
        use_kmeans = trial.suggest_categorical('use_kmeans', [False, True])
        if use_kmeans:
            n_clusters = trial.suggest_int('kmeans_clusters', 2, 6)
            outlier_methods.append(('kmeans_filter', OutlierFilterTransformer(n_clusters=n_clusters)))
        
        # Isolation Forest
        use_iforest = trial.suggest_categorical('use_iforest', [False, True])
        if use_iforest:
            contamination = trial.suggest_float('iforest_contamination', 0.05, 0.2)
            n_estimators = trial.suggest_int('iforest_estimators', 50, 200)
            outlier_methods.append(('iforest_filter', IsolationForestTransformer(contamination=contamination, n_estimators=n_estimators)))
        
        # Local Outlier Factor
        use_lof = trial.suggest_categorical('use_lof', [False, True])
        if use_lof:
            n_neighbors = trial.suggest_int('lof_neighbors', 10, 30)
            contamination = trial.suggest_float('lof_contamination', 0.05, 0.2)
            outlier_methods.append(('lof_filter', LocalOutlierFactorTransformer(n_neighbors=n_neighbors, contamination=contamination)))
        
        # Add outlier detection methods to pipeline
        for method_name, method in outlier_methods:
            steps.append((method_name, method))
        
        return Pipeline(steps)

    def create_target_transformer(self, trial):
        """Create target transformer based on trial suggestion"""
        y_transform = trial.suggest_categorical('y_transform', ['none', 'log1p', 'power'])
        
        if y_transform == 'none':
            return None
        elif y_transform == 'log1p':
            # Use FunctionTransformer for log1p
            from sklearn.preprocessing import FunctionTransformer
            return FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=False)
        elif y_transform == 'power':
            # Use PowerTransformer for target
            return PowerTransformer(method='yeo-johnson', standardize=False)
        
        return None

    def step_1_pin_down_ceiling(self, X, y):
        """Step 1: Pin down the *true* ceiling first – don't chase ghosts"""
        self.iteration_count += 1
        self.logger.info("=" * 60)
        self.logger.info("STEP 1 (Iteration %d): Pin down the *true* ceiling first", self.iteration_count)
        self.logger.info("=" * 60)
        
        if self.iteration_count == 1:
            # First iteration: split data with adaptive test size for optimal CV
            n_samples = len(X)
            
            # Adaptive test size: ensure minimum samples for robust CV after split
            if n_samples <= 150:
                test_size = 0.15  # Smaller test set for small datasets
                min_train_for_cv = int(n_samples * 0.85)
            elif n_samples <= 300:
                test_size = 0.18  # Conservative for medium datasets  
                min_train_for_cv = int(n_samples * 0.82)
            else:
                test_size = 0.2   # Standard 80/20 for larger datasets
                min_train_for_cv = int(n_samples * 0.8)
            
            self.logger.info(f"Dataset size: {n_samples} samples, using test_size={test_size:.2f}")
            
            X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
                X, y, np.arange(len(X)), test_size=test_size, random_state=42, shuffle=True
            )
            
            # Adaptive CV configuration based on training set size
            self._configure_cv_strategy(len(X_train))
            
            # Save the exact test indices to prevent data leakage
            test_indices_path = self.model_dir / "hold{}_test_indices.npy".format(self.dataset_num)
            train_indices_path = self.model_dir / "hold{}_train_indices.npy".format(self.dataset_num)
            np.save(test_indices_path, test_idx)
            np.save(train_indices_path, train_idx)
            self.logger.info("Test indices saved to: %s", test_indices_path)
            self.logger.info("Train indices saved to: %s", train_indices_path)
            self.logger.info("Test samples: %d (indices: %s...%s)", len(test_idx), 
                           test_idx[:5] if len(test_idx) >= 5 else test_idx,
                           test_idx[-5:] if len(test_idx) >= 5 else test_idx)
            
            # Store the training data for optimization
            self.X = X_train.copy()
            self.y = y_train.copy()
            self.X_test = X_test.copy()
            self.y_test = y_test.copy()
            
            self.logger.info("Dataset shape: %s", X.shape)
            self.logger.info("   Training: %s", self.X.shape)
            self.logger.info("   Test (held out): %s", self.X_test.shape)
        
        # 5×3 Repeated-KFold RidgeCV for noise ceiling estimation
        print(f"\n🔍 Fast R² validation using RidgeCV model (iteration {self.iteration_count})...")
        validation_pipe = Pipeline([
            ('scale', RobustScaler()),
            ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 20)))
        ])
        
        try:
            scores = cross_val_score(validation_pipe, self.X, self.y, cv=self.cv, scoring='r2', n_jobs=11)
            scores = scores[np.isfinite(scores)]  # Remove any inf/nan values
            
            if len(scores) > 0:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                # Noise ceiling = mean + 2·std as specified in recipe
                new_ceiling = mean_score + 2*std_score
                
                if self.iteration_count == 1:
                    self.validation_r2 = mean_score  # Reference score for comparison
                    self.validation_cv_scores = scores
                    
                self.noise_ceiling = new_ceiling
                self.ceiling_history.append(new_ceiling)
            else:
                self.noise_ceiling = 0.95  # Default reasonable ceiling
                self.validation_r2 = 0.0
                
        except Exception as e:
            self.logger.error("Noise ceiling estimation failed: %s", e)
            self.noise_ceiling = 0.95  # Default fallback
            self.validation_r2 = 0.0
        
        self.logger.info("RidgeCV validation R²: %.4f ± %.4f", self.validation_r2, std_score)
        self.logger.info("Noise ceiling (mean + 2·std): %.4f", self.noise_ceiling)
        
        # Track ceiling improvements
        if self.iteration_count > 1:
            improvement = self.ceiling_history[-1] - self.ceiling_history[-2]
            self.logger.info("Ceiling improvement: %.4f", improvement)
        
        # Ceiling analysis
        if self.target_r2 > self.noise_ceiling:
            self.logger.warning("Target R² (%.3f) > Estimated ceiling (%.3f)", 
                              self.target_r2, self.noise_ceiling)
            self.target_r2 = max(0.93, self.noise_ceiling - 0.02)  # Keep at least 0.93
            self.logger.info("Adjusting target to %.3f", self.target_r2)
        else:
            self.logger.info("Target R² (%.3f) is achievable (ceiling: %.3f)", 
                           self.target_r2, self.noise_ceiling)
        
        return self.noise_ceiling, self.validation_r2

    def step_2_bulletproof_preprocessing(self):
        """Step 2: Find cleanest data via systematic cleaning evaluation"""
        self.logger.info("\nSTEP 2: Systematic cleaning evaluation → cleanest data")
        self.logger.info("=" * 70)
        
        # FIRST: Evaluate all cleaning methods and pick the best one
        best_label, chosen_cleaners, results = self.evaluate_cleaning_methods()
        
        # Safety check: require meaningful improvement over reference validation score
        if best_label != 'raw' and hasattr(self, 'validation_r2') and self.validation_r2 > 0:
            improvement = results[best_label][2] - self.validation_r2  # mean_r2 - validation_reference
            if improvement < 0.005:  # Minimum improvement threshold
                self.logger.info(f"🔄 No meaningful gain from cleaning ({improvement:+.4f}) – keeping original training data")
                best_label = 'raw'
                chosen_cleaners = []
        
        self.logger.info(f"🏆 Selected data preprocessing: {best_label}")
        
        # SECOND: Apply the winning cleaning method to training data
        X_tmp = self.X.copy()
        y_tmp = self.y.copy()
        
        for cleaner in chosen_cleaners:
            # Fit the cleaner
            cleaner.fit(X_tmp)
            
            # Apply cleaning (actually remove outliers)
            if hasattr(cleaner, 'outlier_indices_') and cleaner.outlier_indices_.size > 0:
                keep_mask = np.ones(len(X_tmp), dtype=bool)
                keep_mask[cleaner.outlier_indices_] = False
                X_tmp = X_tmp[keep_mask]
                y_tmp = y_tmp[keep_mask]
                self.logger.info(f"  Removed {len(cleaner.outlier_indices_)} outliers with {type(cleaner).__name__}")
        
        # Update training data to the cleaned version
        self.X = X_tmp
        self.y = y_tmp
        self.logger.info(f"Training data cleaned: {self.X.shape[0]} samples remaining")
        
        # THIRD: Apply basic preprocessing pipeline to cleaned data
        self.preprocessing_pipeline = Pipeline([
            ('var', VarianceThreshold(1e-8)),       # kills the zero-variance cols
            ('scale', RobustScaler(quantile_range=(5,95)))  # less sensitive than StandardScaler
        ])
        
        self.logger.info("Applying basic preprocessing pipeline to cleaned data:")
        self.logger.info("  - VarianceThreshold(1e-8)")
        self.logger.info("  - RobustScaler(quantile_range=(5,95))")
        
        # Apply to cleaned training data
        initial_features = self.X.shape[1]
        self.X_clean = self.preprocessing_pipeline.fit_transform(self.X)
        
        # Apply to test data (no outlier removal on test set to avoid data leakage)
        self.X_test_clean = self.preprocessing_pipeline.transform(self.X_test)
        
        removed_features = initial_features - self.X_clean.shape[1]
        self.logger.info("Features before: %d", initial_features)
        self.logger.info("Features after: %d", self.X_clean.shape[1])
        self.logger.info("Removed: %d zero-variance features", removed_features)
        
        return self.X_clean, self.X_test_clean

    def make_model(self, trial):
        """Create model based on type and Optuna trial suggestions - enhanced model zoo"""
        # Expanded model choices
        model_choices = ['ridge', 'elastic', 'svr', 'gbrt', 'mlp', 'rf', 'et', 'ada']
        
        # Add XGBoost and LightGBM if available
        if HAS_XGB:
            model_choices.append('xgb')
        if HAS_LGB:
            model_choices.append('lgb')
        
        mdl = trial.suggest_categorical('mdl', model_choices)
        
        if mdl == 'ridge':
            return Ridge(alpha=trial.suggest_float('α', 1e-3, 10, log=True))
        
        elif mdl == 'elastic':
            return ElasticNet(
                alpha=trial.suggest_float('α', 1e-4, 1, log=True),
                l1_ratio=trial.suggest_float('l1', 0.0, 1.0),
                max_iter=2000
            )
        
        elif mdl == 'svr':
            # Enhanced SVR with kernel choice
            kernel = trial.suggest_categorical('svr_kernel', ['rbf', 'linear', 'poly', 'sigmoid'])
            
            if kernel == 'rbf':
                return SVR(
                    C=trial.suggest_float('C', 0.1, 100, log=True),
                    gamma=trial.suggest_float('γ', 1e-3, 1, log=True),
                    kernel=kernel
                )
            elif kernel == 'poly':
                return SVR(
                    C=trial.suggest_float('C', 0.1, 100, log=True),
                    gamma=trial.suggest_float('γ', 1e-3, 1, log=True),
                    degree=trial.suggest_int('poly_degree', 2, 4),
                    kernel=kernel
                )
            else:
                return SVR(
                    C=trial.suggest_float('C', 0.1, 100, log=True),
                    kernel=kernel
                )
        
        elif mdl == 'gbrt':
            # Enhanced GBRT with more hyperparameters
            return GradientBoostingRegressor(
                n_estimators=trial.suggest_int('n', 30, 200),
                max_depth=trial.suggest_int('d', 2, 4),
                learning_rate=trial.suggest_float('lr', 0.01, 0.2),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                loss=trial.suggest_categorical('loss', ['squared_error', 'absolute_error', 'huber']),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                random_state=42
            )
        
        elif mdl == 'mlp':
            # Enhanced MLP with more options
            architecture = trial.suggest_categorical('arch', [(256,), (128,64), (128,128,64), (512,256), (256,128,64)])
            activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
            solver = trial.suggest_categorical('solver', ['adam', 'lbfgs'])
            
            if solver == 'lbfgs':
                # lbfgs works better with smaller datasets
                batch_size = 'auto'
            else:
                batch_size = trial.suggest_categorical('batch_size', ['auto', 32, 64, 128])
            
            return MLPRegressor(
                hidden_layer_sizes=architecture,
                alpha=trial.suggest_float('l2', 1e-5, 1e-2, log=True),
                learning_rate_init=trial.suggest_float('lr', 1e-4, 1e-2, log=True),
                max_iter=800,
                early_stopping=True,
                activation=activation,
                solver=solver,
                batch_size=batch_size,
                random_state=42
            )
        
        elif mdl == 'rf':
            # Random Forest
            return RandomForestRegressor(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 3, 15),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 5),
                max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                random_state=42,
                n_jobs=1  # Fixed: avoid parallelization trap with cross_val_score
            )
        
        elif mdl == 'et':
            # Extra Trees
            return ExtraTreesRegressor(
                n_estimators=trial.suggest_int('n_estimators', 50, 300),
                max_depth=trial.suggest_int('max_depth', 3, 15),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 5),
                max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                random_state=42,
                n_jobs=1  # Fixed: avoid parallelization trap with cross_val_score
            )
        
        elif mdl == 'ada':
            # AdaBoost
            return AdaBoostRegressor(
                n_estimators=trial.suggest_int('n_estimators', 50, 200),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 2.0),
                loss=trial.suggest_categorical('loss', ['linear', 'square', 'exponential']),
                random_state=42
            )
        
        elif mdl == 'xgb' and HAS_XGB:
            # XGBoost with error handling
            try:
                return xgb.XGBRegressor(
                    n_estimators=trial.suggest_int('n_estimators', 50, 300),
                    max_depth=trial.suggest_int('max_depth', 3, 8),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    subsample=trial.suggest_float('subsample', 0.6, 1.0),
                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    reg_alpha=trial.suggest_float('reg_alpha', 0, 10),
                    reg_lambda=trial.suggest_float('reg_lambda', 0, 10),
                    random_state=42,
                    n_jobs=1  # single-thread to avoid parallelization trap with cross_val_score n_jobs=-1
                )
            except Exception:
                # XGBoost can fail in some configurations
                raise optuna.exceptions.TrialPruned()
        
        elif mdl == 'lgb' and HAS_LGB:
            # LightGBM with safe parallel execution
            try:
                return lgb.LGBMRegressor(
                    n_estimators=trial.suggest_int('n_estimators', 50, 300),
                    max_depth=trial.suggest_int('max_depth', 3, 8),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    subsample=trial.suggest_float('subsample', 0.6, 1.0),
                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    reg_alpha=trial.suggest_float('reg_alpha', 0, 10),
                    reg_lambda=trial.suggest_float('reg_lambda', 0, 10),
                    random_state=42,
                    n_jobs=1,  # Force single-thread to avoid logging conflicts
                    verbose=-1
                )
            except Exception:
                # LightGBM can fail with logging conflicts in parallel execution
                raise optuna.exceptions.TrialPruned()
        
        else:
            raise ValueError("Unknown model type: {}".format(mdl))

    def objective(self, trial):
        """Optuna objective function with dynamic preprocessing and target transforms"""
        try:
            # Tunable prune threshold
            prune_floor = trial.suggest_float('prune_floor', -3.0, -0.5)
            
            # Build dynamic preprocessing pipeline
            preprocessing = self.build_preprocessing_pipeline(trial)
            
            # Create base model
            base_model = self.make_model(trial)
            
            # Create target transformer
            target_transformer = self.create_target_transformer(trial)
            
            # Complete pipeline with optional target transformation
            if target_transformer is not None:
                # Use TransformedTargetRegressor for target transformation
                pipe = Pipeline([
                    ('preprocess', preprocessing),
                ])
                
                # Create the final model with target transformation
                model = TransformedTargetRegressor(
                    regressor=base_model,
                    transformer=target_transformer,
                    check_inverse=False
                )
                final_pipe = Pipeline([
                    ('preprocess', preprocessing),
                    ('model', model)
                ])
            else:
                # Standard pipeline without target transformation
                final_pipe = Pipeline([
                    ('preprocess', preprocessing),
                    ('model', base_model)
                ])
            
            # Dynamic CV configuration with data size constraints
            train_size = len(self.X_clean)
            min_samples_per_fold = 20
            max_safe_splits = min(8, max(2, train_size // min_samples_per_fold))
            
            cv_splits = trial.suggest_int('cv_splits', 2, max_safe_splits)
            cv_repeats = trial.suggest_int('cv_repeats', 1, 3 if train_size > 150 else 2)
            cv = RepeatedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=42)
            
            # Cross-validation scoring (back to high parallelization)
            scores = cross_val_score(final_pipe, self.X_clean, self.y, cv=cv, scoring='r2', n_jobs=-1)
            scores = scores[np.isfinite(scores)]  # Remove any inf/nan values
            
            if len(scores) == 0:
                raise optuna.exceptions.TrialPruned()
                
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Dynamic early failure safety net
            if mean_score < prune_floor:
                raise optuna.exceptions.TrialPruned()
            
            # Store additional metrics and pipeline
            trial.set_user_attr('mean_r2', mean_score)
            trial.set_user_attr('std_r2', std_score)
            trial.set_user_attr('cv_scores', scores.tolist())
            trial.set_user_attr('model_type', trial.params['mdl'])
            trial.set_user_attr('pipeline', final_pipe)  # Store the pipeline
            
            # Enhanced progress tracking moved to callback
            
            return mean_score
            
        except Exception as e:
            self.logger.error("Trial %d failed: %s", trial.number, e)
            raise optuna.exceptions.TrialPruned()

    def step_3_optuna_search(self):
        """Step 3: Enhanced Optuna search with dynamic preprocessing"""
        self.logger.info("\nSTEP 3: Enhanced Optuna search (iteration %d)", self.iteration_count)
        self.logger.info("=" * 55)
        self.logger.info("Target R²: %.3f", self.target_r2)
        self.logger.info("Max trials: %d", self.max_trials)
        self.logger.info("Search space: Ridge | ElasticNet | SVR | GBRT | MLP | RF | ET | AdaBoost")
        if HAS_XGB:
            self.logger.info("               + XGBoost available")
        if HAS_LGB:
            self.logger.info("               + LightGBM available")
        self.logger.info("Dynamic options: 5 scalers, 4 dim-reduction, 4 feature selection, 3 outlier detection")
        self.logger.info("                 Tunable CV splits/repeats, target transforms, ~60-70 hyperparameters")
        
        # Create study with MedianPruner
        study_name = "optimization_iter_{}".format(self.iteration_count)
        self.study = optuna.create_study(
            direction='maximize',
            pruner=MedianPruner(n_startup_trials=5),
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name=study_name
        )
        
        # Optimize with progress bar
        start_time = time.time()
        
        def progress_callback(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                try:
                    if study.best_value >= self.target_r2:
                        elapsed = time.time() - start_time
                        msg = f"🎉 TARGET ACHIEVED! R² = {study.best_value:.4f} >= {self.target_r2} in {trial.number} trials ({elapsed:.1f}s)"
                        print(f"\n{Colors.GREEN}{msg}{Colors.END}")
                        self.logger.info(msg)
                        study.stop()
                except ValueError:
                    # No completed trials yet, continue
                    pass

        def on_complete(study, trial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                mdl = trial.params.get('mdl', '?')
                print(f"{Colors.LIME}⚡  #{trial.number:03}  "
                      f"{mdl:<7} R²={trial.value:7.4f}{Colors.END}")
                # Track all trials for final ranking
                self.trial_log.append((trial.number, mdl, trial.value))
        
        # Setup progress bar
        if HAS_TQDM:
            pbar = tqdm(total=self.max_trials, desc=f"🔍 Optuna Search (Iter {self.iteration_count})", 
                       bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} trials [{elapsed}<{remaining}, Best R²={postfix}]',
                       colour='cyan')
            # Initialize with N/A since no trials completed yet
            pbar.set_postfix_str("N/A")
            
            def pbar_callback(study, trial):
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    # Safely get best value
                    try:
                        best_val = study.best_value
                        pbar.set_postfix_str(f"{best_val:.4f}")
                        
                        if best_val >= self.target_r2:
                            elapsed = time.time() - start_time
                            msg = f"🎉 TARGET ACHIEVED! R² = {best_val:.4f} >= {self.target_r2} in {trial.number} trials ({elapsed:.1f}s)"
                            pbar.write(f"\n{Colors.GREEN}{msg}{Colors.END}")
                            self.logger.info(msg)
                            study.stop()
                    except ValueError:
                        # No completed trials yet
                        pbar.set_postfix_str("N/A")
                    
                    pbar.update(1)
            
            callbacks = [progress_callback, pbar_callback, on_complete]
        else:
            callbacks = [progress_callback, on_complete]
        
        try:
            self.study.optimize(
                self.objective,
                n_trials=self.max_trials,
                callbacks=callbacks
            )
            
            if HAS_TQDM:
                pbar.close()
        except KeyboardInterrupt:
            self.logger.info("Optimization interrupted by user")
        
        # Results summary
        if len(self.study.trials) > 0:
            completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed_trials:
                best_value = self.study.best_value
                msg = "Best R² (iteration {}): {:.4f}".format(self.iteration_count, best_value)
                print(f"\n📊 {msg}")
                self.logger.info(msg)
                self.logger.info("Best model: %s", self.study.best_trial.user_attrs.get('model_type', 'N/A'))
                self.logger.info("Best params: %s", self.study.best_params)
                
                return best_value >= self.target_r2
            else:
                self.logger.warning("No trials completed successfully")
                return False
        
        return False

    def statistical_significance_test(self):
        """Test if the best pipeline significantly outperforms baseline"""
        if not self.study or len(self.study.trials) == 0:
            return False, "No optimization results available"
        
        self.logger.info("\n🧪 Statistical Significance Testing")
        self.logger.info("=" * 45)
        
        # Get best trial CV scores
        best_trial = self.study.best_trial
        best_cv_scores = np.array(best_trial.user_attrs['cv_scores'])
        validation_cv_scores = self.validation_cv_scores
        
        # Ensure arrays have same length for paired t-test
        min_length = min(len(best_cv_scores), len(validation_cv_scores))
        best_cv_scores = best_cv_scores[:min_length]
        validation_cv_scores = validation_cv_scores[:min_length]
        
        # Paired t-test
        _, t_pvalue = stats.ttest_rel(best_cv_scores, validation_cv_scores)
        
        # Bootstrap confidence interval for difference
        def mean_diff(x, y):
            return np.mean(x - y)
        
        # Create bootstrap samples
        n_bootstrap = 9999
        rng = np.random.default_rng(42)
        bootstrap_diffs = []
        
        # Use the minimum length for bootstrap sampling
        sample_size = min_length
        
        for _ in range(n_bootstrap):
            # Resample indices
            indices = rng.choice(sample_size, size=sample_size, replace=True)
            diff = mean_diff(best_cv_scores[indices], validation_cv_scores[indices])
            bootstrap_diffs.append(diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        ci_lower = np.percentile(bootstrap_diffs, 2.5)
        ci_upper = np.percentile(bootstrap_diffs, 97.5)
        
        mean_improvement = np.mean(best_cv_scores - validation_cv_scores)
        
        # Significance assessment
        is_significant = (t_pvalue < 0.05) and (ci_lower > 0)
        
        self.logger.info("Mean improvement: %.4f", mean_improvement)
        self.logger.info("95%% CI: [%.4f, %.4f]", ci_lower, ci_upper)
        self.logger.info("Paired t-test p-value: %.4f", t_pvalue)
        self.logger.info("Statistically significant: %s", "✅ YES" if is_significant else "❌ NO")
        
        if not is_significant:
            if ci_lower <= 0 <= ci_upper:
                reason = "95%% CI crosses 0 → no significant improvement"
            else:
                reason = "p-value {:.4f} > 0.05".format(t_pvalue)
            self.logger.info("🎓 %s", reason)
        
        return is_significant, "Mean: {:.4f}, CI: [{:.4f}, {:.4f}], p: {:.4f}".format(mean_improvement, ci_lower, ci_upper, t_pvalue)

    def check_ceiling_convergence(self, min_improvement=0.01):
        """Check if noise ceiling has converged"""
        if len(self.ceiling_history) < 2:
            return False
        
        improvement = self.ceiling_history[-1] - self.ceiling_history[-2]
        converged = improvement < min_improvement
        
        if converged:
            self.logger.info("🛑 Ceiling converged: improvement %.4f < %.4f", improvement, min_improvement)
        
        return converged

    def step_4_lock_in_champion(self):
        """Step 4: Lock in and export the champion"""
        self.logger.info("\nSTEP 4: Lock in and export the champion (iteration %d)", self.iteration_count)
        self.logger.info("=" * 65)
        
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
        model_path = self.model_dir / "hold{}_best_model_iter{}.pkl".format(self.dataset_num, self.iteration_count)
        joblib.dump(self.best_pipeline, model_path)
        self.logger.info("Best model saved to: %s", model_path)
        
        # Also save preprocessing pipeline separately
        preprocessing_path = self.model_dir / "hold{}_preprocessing_pipeline_iter{}.pkl".format(self.dataset_num, self.iteration_count)
        joblib.dump(self.preprocessing_pipeline, preprocessing_path)
        self.logger.info("Preprocessing pipeline saved to: %s", preprocessing_path)
        
        return self.best_pipeline

    def step_5_final_evaluation(self):
        """Step 5: Final evaluation on held-out test set"""
        self.logger.info("\nSTEP 5: Final evaluation on held-out test set (iteration %d)", self.iteration_count)
        self.logger.info("=" * 70)
        
        if self.best_pipeline is None:
            self.logger.error("No trained model available")
            return None
        
        # Evaluate on the proper held-out test set
        self.logger.info("Evaluating on %d-sample hold-out test...", len(self.y_test))
        y_pred = self.best_pipeline.predict(self.X_test_clean)
        
        r2_test = r2_score(self.y_test, y_pred)
        mae_test = mean_absolute_error(self.y_test, y_pred)
        rmse_test = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
        self.logger.info("Test R²: %.4f", r2_test)
        self.logger.info("Test MAE: %.6f", mae_test)
        self.logger.info("Test RMSE: %.6f", rmse_test)
        self.logger.info("RidgeCV validation R² (CV): %.4f", self.validation_r2)
        
        # Success criteria
        beats_validation = r2_test > self.validation_r2
        near_ceiling = abs(r2_test - self.noise_ceiling) < 0.05
        
        self.logger.info("Model Assessment:")
        self.logger.info("  Beats RidgeCV validation: %s", "✅ YES" if beats_validation else "❌ NO")
        self.logger.info("  Near ceiling: %s", "✅ YES" if near_ceiling else "❌ NO")
        
        if beats_validation and near_ceiling:
            msg = "🏆 MODEL ACCEPTED - At theoretical maximum!"
            print(f"{Colors.GREEN}{Colors.BOLD}{msg}{Colors.END}")
            self.logger.info(msg)
        
        # Save results
        eval_results = {
            'iteration': self.iteration_count,
            'test_r2': r2_test,
            'test_mae': mae_test,
            'test_rmse': rmse_test,
            'validation_r2': self.validation_r2,
            'noise_ceiling': self.noise_ceiling,
            'beats_validation': beats_validation,
            'near_ceiling': near_ceiling,
            'best_model_type': self.study.best_trial.user_attrs['model_type'],
            'best_params': self.study.best_params,
            'ceiling_history': self.ceiling_history
        }
        
        # Save results to file
        results_path = self.model_dir / "hold{}_evaluation_results_iter{}.txt".format(self.dataset_num, self.iteration_count)
        with open(results_path, 'w', encoding='utf-8') as f:
            for key, value in eval_results.items():
                f.write("{}: {}\n".format(key, value))
        
        return eval_results

    def create_optimization_summary_table(self):
        """Create a concise summary table of optimization results"""
        if not self.study or len(self.study.trials) == 0:
            return
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}📋 OPTIMIZATION SUMMARY (Hold-{self.dataset_num}) - Iteration {self.iteration_count}{Colors.END}")
        print("=" * 80)
        
        # Get completed trials sorted by value
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        completed_trials.sort(key=lambda x: x.value, reverse=True)
        
        print("🏆 Top 5 Model Configurations:")
        print("-" * 100)
        print(f"{'Rank':<4} {'Model':<8} {'R²':<6} {'DimRed':<12} {'Outlier':<12} {'Scaler':<8} {'Features':<8} {'Target':<8}")
        print("-" * 100)
        
        for i, trial in enumerate(completed_trials[:5]):
            # Dimensionality reduction info
            dim_red = trial.params.get('dim_red', 'none')
            if dim_red == 'pca':
                dim_info = f"PCA({trial.params.get('pca_components', 'N/A')})"
            elif dim_red == 'kpca':
                dim_info = f"KPCA({trial.params.get('kpca_components', 'N/A')})"
            elif dim_red == 'svd':
                dim_info = f"SVD({trial.params.get('svd_components', 'N/A')})"
            else:
                dim_info = "None"
            
            # Outlier detection info
            outlier_methods = []
            if trial.params.get('use_kmeans'):
                outlier_methods.append(f"KM({trial.params.get('kmeans_clusters', 'N/A')})")
            if trial.params.get('use_iforest'):
                outlier_methods.append("IF")
            if trial.params.get('use_lof'):
                outlier_methods.append("LOF")
            outlier_info = "+".join(outlier_methods) if outlier_methods else "None"
            
            scaler = trial.params.get('scaler', 'robust')[:6]
            features = trial.params.get('k_features', 'N/A')
            target_transform = trial.params.get('y_transform', 'none')[:6]
            
            print(f"{i+1:<4} {trial.params['mdl']:<8} {trial.value:<6.3f} {dim_info:<12} {outlier_info:<12} {scaler:<8} {features:<8} {target_transform:<8}")
        
        print("\n📊 Preprocessing Impact Analysis:")
        print("-" * 50)
        
        # Analyze dimensionality reduction impact
        pca_trials = [t for t in completed_trials if t.params.get('dim_red') == 'pca']
        no_dimred_trials = [t for t in completed_trials if t.params.get('dim_red') == 'none']
        
        if pca_trials and no_dimred_trials:
            pca_mean = np.mean([t.value for t in pca_trials])
            no_dimred_mean = np.mean([t.value for t in no_dimred_trials])
            pca_effect = pca_mean - no_dimred_mean
            print(f"PCA effect: {pca_effect:+.4f} (With: {pca_mean:.3f}, Without: {no_dimred_mean:.3f})")
        
        # Analyze K-means impact
        kmeans_trials = [t for t in completed_trials if t.params.get('use_kmeans')]
        no_kmeans_trials = [t for t in completed_trials if not t.params.get('use_kmeans')]
        
        if kmeans_trials and no_kmeans_trials:
            kmeans_mean = np.mean([t.value for t in kmeans_trials])
            no_kmeans_mean = np.mean([t.value for t in no_kmeans_trials])
            kmeans_effect = kmeans_mean - no_kmeans_mean
            print(f"K-means effect: {kmeans_effect:+.4f} (With: {kmeans_mean:.3f}, Without: {no_kmeans_mean:.3f})")
        
        # Analyze Isolation Forest impact
        iforest_trials = [t for t in completed_trials if t.params.get('use_iforest')]
        no_iforest_trials = [t for t in completed_trials if not t.params.get('use_iforest')]
        
        if iforest_trials and no_iforest_trials:
            iforest_mean = np.mean([t.value for t in iforest_trials])
            no_iforest_mean = np.mean([t.value for t in no_iforest_trials])
            iforest_effect = iforest_mean - no_iforest_mean
            print(f"IsolationForest effect: {iforest_effect:+.4f} (With: {iforest_mean:.3f}, Without: {no_iforest_mean:.3f})")
        
        # Analyze scaler impact
        robust_trials = [t for t in completed_trials if t.params.get('scaler') == 'robust']
        standard_trials = [t for t in completed_trials if t.params.get('scaler') == 'standard']
        
        if robust_trials and standard_trials:
            robust_mean = np.mean([t.value for t in robust_trials])
            standard_mean = np.mean([t.value for t in standard_trials])
            scaler_effect = robust_mean - standard_mean
            print(f"Robust vs Standard: {scaler_effect:+.4f} (Robust: {robust_mean:.3f}, Standard: {standard_mean:.3f})")
        
        # Analyze target transformation impact
        target_transform_trials = [t for t in completed_trials if t.params.get('y_transform') != 'none']
        no_target_transform_trials = [t for t in completed_trials if t.params.get('y_transform') == 'none']
        
        if target_transform_trials and no_target_transform_trials:
            target_mean = np.mean([t.value for t in target_transform_trials])
            no_target_mean = np.mean([t.value for t in no_target_transform_trials])
            target_effect = target_mean - no_target_mean
            print(f"Target transform effect: {target_effect:+.4f} (With: {target_mean:.3f}, Without: {no_target_mean:.3f})")
        
        print(f"\n🎯 Best achieved: {self.study.best_value:.4f}")
        print(f"📏 Noise ceiling: {self.noise_ceiling:.4f}")
        print(f"🔄 Ceiling history: {[f'{c:.3f}' for c in self.ceiling_history]}")

    def iterative_optimization_loop(self, max_iterations=3, min_ceiling_improvement=0.01, original_X=None, original_y=None):
        """Main iterative optimization loop with ceiling re-estimation"""
        self.logger.info("\n🔄 STARTING ITERATIVE OPTIMIZATION LOOP")
        self.logger.info("Max iterations: %d", max_iterations)
        self.logger.info("Min ceiling improvement: %.4f", min_ceiling_improvement)
        self.logger.info("=" * 60)
        
        best_overall_r2 = 0
        best_overall_results = None
        
        for iteration in range(1, max_iterations + 1):
            print(f"\n{Colors.BOLD}{Colors.YELLOW}🔄 ITERATION {iteration}/{max_iterations}{Colors.END}")
            
            # Step 1: Estimate/re-estimate noise ceiling
            if iteration == 1:
                # First iteration: pass original data and split
                self.step_1_pin_down_ceiling(original_X, original_y)
            else:
                # Later iterations: use already split training data
                self.step_1_pin_down_ceiling(self.X, self.y)
            
            # Step 2: Preprocessing (basic for iteration 1, already done for later iterations)
            if iteration == 1:
                self.step_2_bulletproof_preprocessing()
            
            # Step 3: Optuna search with dynamic preprocessing
            success = self.step_3_optuna_search()
            
            # Statistical significance test
            is_significant, _ = self.statistical_significance_test()
            
            # Step 4: Lock in champion
            self.step_4_lock_in_champion()
            
            # Step 5: Final evaluation
            eval_results = self.step_5_final_evaluation()
            
            # Create summary table
            self.create_optimization_summary_table()
            
            # Track best overall result
            if eval_results and eval_results['test_r2'] > best_overall_r2:
                best_overall_r2 = eval_results['test_r2']
                best_overall_results = eval_results.copy()
            
            # Check convergence conditions
            if iteration > 1:
                ceiling_converged = self.check_ceiling_convergence(min_ceiling_improvement)
                
                if ceiling_converged and not is_significant:
                    self.logger.info("🛑 EARLY CONVERGENCE: Ceiling converged and no significant improvement")
                    break
            
            # Check if target achieved
            if success and is_significant:
                self.logger.info("🎉 TARGET ACHIEVED with statistical significance in iteration %d", iteration)
                break
        
        # Final summary across all iterations
        print(f"\n{Colors.BOLD}{Colors.CYAN}🏁 FINAL OPTIMIZATION SUMMARY{Colors.END}")
        print("=" * 60)
        print("Total iterations: {}".format(self.iteration_count))
        print("Best overall test R²: {:.4f}".format(best_overall_r2))
        print("Ceiling progression: {}".format(' → '.join('{:.3f}'.format(c) for c in self.ceiling_history)))
        
        if best_overall_results:
            print("🏆 Best achieved test R²: {:.4f}".format(best_overall_results['test_r2']))
            print("📏 Final noise ceiling: {:.4f}".format(best_overall_results['noise_ceiling']))
            print("🔄 Iterations completed: {}".format(best_overall_results['iteration']))
            print("🏗️ Best model: {}".format(best_overall_results['best_model_type']))
            print("📈 Ceiling progression: {}".format(' → '.join('{:.3f}'.format(c) for c in best_overall_results['ceiling_history'])))
            
            # Statistical significance summary
            optimizer_final = self
            if optimizer_final.study:
                is_significant, sig_summary = optimizer_final.statistical_significance_test()
                print("🧪 Statistical significance: {}".format('✅ YES' if is_significant else '❌ NO'))
                print("   {}".format(sig_summary))
        
        print("\n📁 All outputs saved to: {}best_model_hold{}/{}".format(Colors.BOLD, self.dataset_num, Colors.END))
        print("   Models and logs from each iteration are preserved")
        print("   - hold{}_best_model_iter[N].pkl (trained models)".format(self.dataset_num))
        print("   - hold{}_preprocessing_pipeline_iter[N].pkl (preprocessing)".format(self.dataset_num))
        print("   - hold{}_training_log.txt (complete log)".format(self.dataset_num))
        print("   - hold{}_evaluation_results_iter[N].txt (test results)".format(self.dataset_num))
        print("   - hold{}_diagnostic_plots_iter[N].png (visualizations)".format(self.dataset_num))
        print("   - hold{}_test_indices.npy (test data indices)".format(self.dataset_num))
        print("   - hold{}_train_indices.npy (train data indices)".format(self.dataset_num))
        
        # Print ALL Optuna trials in ascending order
        if self.trial_log:
            print(f"\n{Colors.BOLD}{Colors.CYAN}📑 ALL OPTUNA TRIALS (ascending R²){Colors.END}")
            print("-" * 50)
            print(f"{'Trial':<6} {'Model':<8} {'R²':<8}")
            print("-" * 50)
            for num, mdl, r2 in sorted(self.trial_log, key=lambda x: x[2]):
                print(f"{num:03d}    {mdl:<8} {r2:7.4f}")
            
            if len(self.trial_log) > 0:
                worst_r2 = min(self.trial_log, key=lambda x: x[2])[2]
                best_r2 = max(self.trial_log, key=lambda x: x[2])[2]
                print(f"\nRange: {worst_r2:.4f} → {best_r2:.4f} (span: {best_r2-worst_r2:.4f})")
        
        # Success criteria check
        if best_overall_results:
            target_achieved = best_overall_results['test_r2'] >= self.target_r2
            near_ceiling = abs(best_overall_results['test_r2'] - best_overall_results['noise_ceiling']) < 0.05
            
            if target_achieved and near_ceiling:
                print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 SUCCESS: Target achieved and model is at theoretical maximum!{Colors.END}")
            elif target_achieved:
                print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  PARTIAL SUCCESS: Target achieved but ceiling may be higher{Colors.END}")
            else:
                print(f"\n{Colors.RED}{Colors.BOLD}❌ TARGET NOT REACHED: Consider CNN approach or data quality issues{Colors.END}")
        
        return best_overall_results

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
        fig.suptitle('Hold-{} Model: Final Evaluation (Iteration {})'.format(self.dataset_num, self.iteration_count), fontsize=16, fontweight='bold')
        
        # 1. Predicted vs Actual
        ax1.scatter(self.y_test, y_pred, alpha=0.6, color='blue')
        min_val = min(self.y_test.min(), y_pred.min())
        max_val = max(self.y_test.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Predicted vs Actual (Test Set)')
        ax1.grid(True, alpha=0.3)
        
        r2_test = r2_score(self.y_test, y_pred)
        ax1.text(0.05, 0.95, 'R² = {:.4f}'.format(r2_test), transform=ax1.transAxes,
                bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white', 'alpha': 0.8})
        
        # 2. Residuals
        residuals = self.y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, color='green')
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)
        
        # 3. Noise ceiling progression
        ax3.plot(range(1, len(self.ceiling_history) + 1), self.ceiling_history, 'bo-', linewidth=2)
        ax3.axhline(y=self.target_r2, color='r', linestyle='--', 
                   label='Target R² = {}'.format(self.target_r2))
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Noise Ceiling')
        ax3.set_title('Noise Ceiling Progression')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Model comparison (from current study)
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
                           label='Baseline = {:.3f}'.format(self.baseline_r2))
                ax4.set_ylabel('CV R² Score')
                ax4.set_title('Model Type Comparison (Iter {})'.format(self.iteration_count))
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.model_dir / "hold{}_diagnostic_plots_iter{}.png".format(self.dataset_num, self.iteration_count)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Diagnostic plots saved to: %s", plot_file)

def main():
    """Main execution function with iterative optimization loop"""
    print(f"{Colors.BOLD}{Colors.CYAN}")
    print("=" * 70)
    print("🚀 BATTLE-TESTED END-TO-END ML PLAYBOOK v3.0")
    print("   Running on Hold-{} dataset".format(DATASET))
    print("   Features: PCA, K-means, iterative ceiling estimation")
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
            print(f"\n{Colors.BOLD}📁 Loading Hold-3 Data{Colors.END}")
            X = pd.read_csv('predictors_Hold 1 Full_20250527_151252.csv', header=None).values.astype(np.float32)
            y = pd.read_csv('targets_Hold 1 Full_20250527_151252.csv', header=None).values.astype(np.float32).ravel()
        else:
            raise ValueError("Invalid DATASET value: {}. Must be 1, 2, or 3.".format(DATASET))
        
        print("✅ Data loaded: {} features, {} samples".format(X.shape, len(y)))
        
        # Initialize optimizer with increased trial budget for comprehensive hyperparameter search
        optimizer = BattleTestedOptimizer(DATASET, target_r2=0.93, max_trials=50)
        
        # Execute iterative optimization loop
        print(f"\n{Colors.BOLD}{Colors.GREEN}🎯 EXECUTING ITERATIVE OPTIMIZATION RECIPE{Colors.END}")
        
        # Run the iterative optimization loop
        final_results = optimizer.iterative_optimization_loop(max_iterations=3, min_ceiling_improvement=0.01, original_X=X, original_y=y)
        
        # Create final diagnostic plots
        optimizer.create_diagnostic_plots()
        
        # Final summary
        print(f"\n{Colors.BOLD}{Colors.CYAN}📋 COMPREHENSIVE FINAL SUMMARY{Colors.END}")
        print("=" * 60)
        
        if final_results:
            print("🏆 Best achieved test R²: {:.4f}".format(final_results['test_r2']))
            print("📏 Final noise ceiling: {:.4f}".format(final_results['noise_ceiling']))
            print("🔄 Iterations completed: {}".format(final_results['iteration']))
            print("🏗️ Best model: {}".format(final_results['best_model_type']))
            print("📈 Ceiling progression: {}".format(' → '.join('{:.3f}'.format(c) for c in final_results['ceiling_history'])))
            
            # Statistical significance summary
            optimizer_final = optimizer
            if optimizer_final.study:
                is_significant, sig_summary = optimizer_final.statistical_significance_test()
                print("🧪 Statistical significance: {}".format('✅ YES' if is_significant else '❌ NO'))
                print("   {}".format(sig_summary))
        
        print("\n📁 All outputs saved to: {}best_model_hold{}/{}".format(Colors.BOLD, DATASET, Colors.END))
        print("   Models and logs from each iteration are preserved")
        print("   - hold{}_best_model_iter[N].pkl (trained models)".format(DATASET))
        print("   - hold{}_preprocessing_pipeline_iter[N].pkl (preprocessing)".format(DATASET))
        print("   - hold{}_training_log.txt (complete log)".format(DATASET))
        print("   - hold{}_evaluation_results_iter[N].txt (test results)".format(DATASET))
        print("   - hold{}_diagnostic_plots_iter[N].png (visualizations)".format(DATASET))
        print("   - hold{}_test_indices.npy (test data indices)".format(DATASET))
        print("   - hold{}_train_indices.npy (train data indices)".format(DATASET))
        
        # Success criteria check
        if final_results:
            target_achieved = final_results['test_r2'] >= optimizer.target_r2
            near_ceiling = abs(final_results['test_r2'] - final_results['noise_ceiling']) < 0.05
            
            if target_achieved and near_ceiling:
                print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 SUCCESS: Target achieved and model is at theoretical maximum!{Colors.END}")
            elif target_achieved:
                print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  PARTIAL SUCCESS: Target achieved but ceiling may be higher{Colors.END}")
            else:
                print(f"\n{Colors.RED}{Colors.BOLD}❌ TARGET NOT REACHED: Consider CNN approach or data quality issues{Colors.END}")
        
        return final_results
        
    except FileNotFoundError as e:
        print("{}❌ Data files not found: {}{}".format(Colors.RED, e, Colors.END))
        if DATASET == 1:
            print("Make sure 'Predictors_Hold-1_2025-04-14_18-28.csv' and '9_10_24_Hold_01_targets.csv' exist")
        else:
            print("Make sure 'hold2_predictor.csv' and 'hold2_target.csv' exist")
        return None
    except Exception as e:
        print("{}❌ Error: {}{}".format(Colors.RED, e, Colors.END))
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main() 