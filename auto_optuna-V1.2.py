#!/usr/bin/env python3
"""
Battle-Tested End-to-End ML Playbook v1.2 - SYSTEMATIC 3-PHASE EDITION
======================================================================
Phase 1: MODEL FAMILY TOURNAMENT (Raw Data) - Find best model family
Phase 2: ITERATIVE PREPROCESSING STACK OPTIMIZATION - Add 1 method at a time  
Phase 3: DEEP MODEL HYPERPARAMETER OPTIMIZATION - Reach noise ceiling
"""

# =============================================================================
# DATASET CONFIGURATION - Change this value to switch datasets
# =============================================================================
DATASET = 3  # Set to 1 for Hold-1 (400 samples), 2 for Hold-2 (108 samples), or 3 for Hold-3 (401 samples)
# =============================================================================

import pandas as pd
import numpy as np
import time
import joblib
import logging
from pathlib import Path
import signal
from contextlib import contextmanager
import sys
import ctypes
from collections import OrderedDict

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
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor, 
                             ExtraTreesRegressor, AdaBoostRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

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

# Rich for live tables and trees
try:
    from rich.live import Live
    from rich.table import Table
    from rich.tree import Tree
    from rich.console import Console
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

# Statistical testing
from scipy import stats

# Visualization and reporting
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from ml_utils import Colors, set_console_title, OutlierFilterTransformer, IsolationForestTransformer, LocalOutlierFactorTransformer, HSICFeatureSelector, load_dataset

# Silence Optuna spam
optuna.logging.disable_default_handler()
optuna.logging.set_verbosity(optuna.logging.WARNING)

@contextmanager
def timeout_context(seconds):
    """Context manager for timeout"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler) 

class SystematicOptimizer:
    """
    Systematic 3-Phase ML Optimization Pipeline
    
    Phase 1: Model Family Tournament (Raw Data)
    Phase 2: Iterative Preprocessing Stack Optimization
    Phase 3: Deep Model Hyperparameter Optimization
    """
    
    def __init__(self, dataset_num, max_preprocessing_trials=50, max_hyperopt_trials=500):
        self.dataset_num = dataset_num
        self.max_preprocessing_trials = max_preprocessing_trials
        self.max_hyperopt_trials = max_hyperopt_trials
        
        # Data storage
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        # Phase 1 results
        self.model_family_results = {}
        self.winning_model_class = None
        self.winning_model_params = None
        
        # Phase 2 results
        self.preprocessing_stack = []
        self.preprocessing_history = []
        self.current_best_r2 = 0.0
        
        # Phase 3 results
        self.final_pipeline = None
        self.noise_ceiling = None
        
        # PERSISTENT GLOBAL TREE - Created once and updated throughout
        self.main_tree = None
        self.dataset_node = None
        self.phase1_node = None
        self.phase2_node = None
        self.phase3_node = None
        
        # Setup logging and directories
        self.setup_logging()
        self.setup_cv_strategy()

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
        fh = logging.FileHandler(self.model_dir / f'hold{self.dataset_num}_systematic_log.txt', mode='w')
        fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        logger.addHandler(fh)
        
        # CONSOLE handler - brief format
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(levelname)s │ %(message)s'))
        logger.addHandler(ch)
        
        self.logger = logger

    def setup_cv_strategy(self):
        """Setup cross-validation strategy"""
        # Will be configured adaptively based on training data size
        self.cv = None
        self.cv_splits = 5
        self.cv_repeats = 3

    def configure_cv_for_data_size(self, train_size):
        """Configure CV strategy based on training set size"""
        min_samples_per_fold = 20
        
        if train_size < 100:
            safe_splits = min(3, max(2, train_size // min_samples_per_fold))
            safe_repeats = 1 if train_size < 60 else 2
        elif train_size < 200:
            safe_splits = min(4, max(3, train_size // min_samples_per_fold))
            safe_repeats = min(2, self.cv_repeats)
        else:
            safe_splits = min(self.cv_splits, train_size // min_samples_per_fold)
            safe_repeats = self.cv_repeats
        
        # Apply minimum constraints
        safe_splits = max(2, safe_splits)
        safe_repeats = max(1, safe_repeats)
        
        # Store CV configuration for access later
        self.cv_n_splits = safe_splits
        self.cv_n_repeats = safe_repeats
        
        self.cv = RepeatedKFold(n_splits=safe_splits, n_repeats=safe_repeats, random_state=42)
        
        total_folds = safe_splits * safe_repeats
        approx_fold_size = train_size // safe_splits
        self.logger.info(f"CV configured: {safe_splits}-fold × {safe_repeats} repeats = {total_folds} total folds")
        self.logger.info(f"Approximate samples per fold: {approx_fold_size}")

    def get_model_families(self):
        """Get all model families for Phase 1 tournament (11 base + optional XGB/LGB)"""
        families = OrderedDict([
            ('Ridge', (Ridge, {'alpha': 1.0})),
            ('RPOP', (Ridge, {'alpha': 0.001})),  # Ridge with very small alpha (almost linear)
            ('Lasso', (Lasso, {'alpha': 0.1, 'max_iter': 2000})),
            ('ElasticNet', (ElasticNet, {'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 2000})),
            ('SVR', (SVR, {'C': 10.0, 'gamma': 'scale', 'kernel': 'rbf'})),
            ('DecisionTree', (DecisionTreeRegressor, {'max_depth': 10, 'random_state': 42})),
            ('RandomForest', (RandomForestRegressor, {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'n_jobs': 1})),
            ('ExtraTrees', (ExtraTreesRegressor, {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'n_jobs': 1})),
            ('GradientBoosting', (GradientBoostingRegressor, {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'random_state': 42})),
            ('AdaBoost', (AdaBoostRegressor, {'n_estimators': 100, 'learning_rate': 1.0, 'random_state': 42})),
            ('MLP', (MLPRegressor, {'hidden_layer_sizes': (128, 64), 'alpha': 0.001, 'max_iter': 500, 'random_state': 42}))
        ])
        
        # Add XGBoost and LightGBM if available
        if HAS_XGB:
            families['XGBoost'] = (xgb.XGBRegressor, {
                'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 
                'random_state': 42, 'n_jobs': 1
            })
        if HAS_LGB:
            families['LightGBM'] = (lgb.LGBMRegressor, {
                'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1,
                'random_state': 42, 'n_jobs': 1, 'verbose': -1
            })
        
        return families

    def get_preprocessing_methods(self):
        """Get the 8 preprocessing methods for Phase 2 iterative optimization"""
        methods = OrderedDict([
            ('PCA', {
                'transformer': PCA,
                'param_space': {
                    'n_components': (10, min(50, self.X_train.shape[1] - 5))
                }
            }),
            ('RobustScaler', {
                'transformer': RobustScaler,
                'param_space': {
                    'quantile_range': [(5, 95), (10, 90), (25, 75)]
                }
            }),
            ('StandardScaler', {
                'transformer': StandardScaler,
                'param_space': {}
            }),
            ('KMeansOutlier', {
                'transformer': OutlierFilterTransformer,
                'param_space': {
                    'n_clusters': (2, 6),
                    'min_cluster_size_ratio': [0.05, 0.1, 0.15]
                }
            }),
            ('IsolationForest', {
                'transformer': IsolationForestTransformer,
                'param_space': {
                    'contamination': [0.05, 0.1, 0.15, 0.2],
                    'n_estimators': (50, 200)
                }
            }),
            ('LocalOutlierFactor', {
                'transformer': LocalOutlierFactorTransformer,
                'param_space': {
                    'n_neighbors': (10, 30),
                    'contamination': [0.05, 0.1, 0.15, 0.2]
                }
            }),
            ('FeatureSelection', {
                'transformer': SelectKBest,
                'param_space': {
                    'score_func': [mutual_info_regression, f_regression],
                    'k': (20, min(120, self.X_train.shape[1]))
                }
            }),
            ('QuantileTransform', {
                'transformer': QuantileTransformer,
                'param_space': {
                    'output_distribution': ['uniform', 'normal'],
                    'n_quantiles': (100, min(1000, self.X_train.shape[0] // 2))
                }
            })
        ])
        
        return methods 

    def initialize_main_tree(self, X, y):
        """Initialize the persistent main tree structure"""
        if not HAS_RICH:
            return
            
        # Create the main tree with exact format
        self.main_tree = Tree("⚙️ SYSTEMATIC ML OPTIMIZER v1.2")
        
        # Dataset node
        feature_warning = " ⚠️" if X.shape[1] > 1000 else ""
        self.dataset_node = self.main_tree.add(f"📂 Dataset: Hold-{self.dataset_num} ({X.shape[0]} samples, {X.shape[1]} features{feature_warning})")
        
        # Add empty line node
        self.main_tree.add("")
        
        # Phase nodes
        self.phase1_node = self.main_tree.add("🚩 Phase 1: Model Family Tournament")
        self.main_tree.add("")
        
        self.phase2_node = self.main_tree.add("🚩 Phase 2: Preprocessing Stack Optimization")
        self.main_tree.add("")
        
        self.phase3_node = self.main_tree.add("🚩 Phase 3: Deep Hyperparameter Optimization")
        
        # Initial display
        console.print(self.main_tree)

    def phase_1_model_family_tournament(self, X, y):
        """
        Phase 1: Model Family Tournament on Raw Data
        Test all available model families with NO preprocessing to find the best performer
        """
        model_families = self.get_model_families()
        model_count = len(model_families)
        
        self.logger.info("=" * 60)
        self.logger.info("PHASE 1: Model Family Tournament on Raw Data")
        self.logger.info("=" * 60)
        
        # Split data for training and testing
        test_size = 0.2 if len(X) > 150 else 0.15
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, np.arange(len(X)), test_size=test_size, random_state=42, shuffle=True
        )
        
        # Store data splits
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Configure CV for training data size
        self.configure_cv_for_data_size(len(X_train))
        
        # Save test indices to prevent data leakage
        np.save(self.model_dir / f"hold{self.dataset_num}_test_indices.npy", test_idx)
        np.save(self.model_dir / f"hold{self.dataset_num}_train_indices.npy", train_idx)
        
        # Update Phase 1 node with initial info
        if HAS_RICH and self.phase1_node:
            self.phase1_node.add(f"Models tested: 0/{model_count} ⚠️ (Planned: {model_count})")
            self.phase1_node.add(f"CV: {self.cv_n_splits}-fold × {self.cv_n_repeats} repeats ({self.cv_n_splits * self.cv_n_repeats} folds total)")
            console.print(self.main_tree)
        
        self.logger.info(f"Dataset split: {len(X_train)} training, {len(X_test)} test")
        
        # Get all model families and start testing
        model_families = self.get_model_families()
        
        # Add results node to Phase 1
        if HAS_RICH and self.phase1_node:
            results_node = self.phase1_node.add("📊 Results:")
        
        # Test each model family with live updates
        for i, (family_name, (model_class, default_params)) in enumerate(model_families.items(), 1):
            start_time = time.time()
            
            try:
                # Create model with default parameters
                model = model_class(**default_params)
                
                # Simple pipeline: just variance threshold + model (NO scaling or preprocessing)
                pipeline = Pipeline([
                    ('var_threshold', VarianceThreshold(1e-8)),
                    ('model', model)
                ])
                
                # Cross-validation with 12 threads
                cv_scores = cross_val_score(
                    pipeline, self.X_train, self.y_train, 
                    cv=self.cv, scoring='r2', n_jobs=12
                )
                
                # Handle any invalid scores
                cv_scores = cv_scores[np.isfinite(cv_scores)]
                
                if len(cv_scores) > 0:
                    mean_r2 = np.mean(cv_scores)
                    std_r2 = np.std(cv_scores)
                    elapsed = time.time() - start_time
                    
                    # SANITY CHECK: Flag extreme negative performance
                    if mean_r2 < -50:
                        status_flag = "⚠️ Extreme"
                        self.logger.warning(f"SANITY CHECK FAILED: {family_name} R² = {mean_r2:.4f} (< -50)")
                    elif mean_r2 < -10:
                        status_flag = "⚠️ Extreme"
                        self.logger.warning(f"EXTREME NEGATIVE: {family_name} R² = {mean_r2:.4f} (< -10)")
                    elif mean_r2 < -1:
                        status_flag = "⚠️ Poor"
                    elif mean_r2 < 0:
                        status_flag = "⚠️ Poor"
                    else:
                        status_flag = "✅ Good"
                    
                    # Add to results node
                    if HAS_RICH and self.phase1_node:
                        result_text = f"{family_name}: R²={mean_r2:.3f} ({status_flag})"
                        results_node.add(result_text)
                        console.print(self.main_tree)
                    
                    self.model_family_results[family_name] = {
                        'model_class': model_class,
                        'default_params': default_params,
                        'mean_r2': mean_r2,
                        'std_r2': std_r2,
                        'cv_scores': cv_scores,
                        'time': elapsed,
                        'status': 'success',
                        'performance_flag': status_flag
                    }
                    
                    self.logger.info(f"{family_name}: R² = {mean_r2:.4f} ± {std_r2:.4f} ({elapsed:.1f}s)")
                else:
                    elapsed = time.time() - start_time
                    if HAS_RICH and self.phase1_node:
                        results_node.add(f"{family_name}: NO VALID SCORES")
                        console.print(self.main_tree)
                    self.logger.warning(f"{family_name}: No valid CV scores")
                    
            except Exception as e:
                elapsed = time.time() - start_time
                self.logger.error(f"{family_name}: Failed with error: {e}")
        
        # Find the winner and create results tree
        if self.model_family_results:
            # Sort by mean R² (descending)
            sorted_results = sorted(
                self.model_family_results.items(), 
                key=lambda x: x[1]['mean_r2'], 
                reverse=True
            )
            
            winner_name, winner_data = sorted_results[0]
            self.winning_model_class = winner_data['model_class']
            self.winning_model_params = winner_data['default_params']
            self.current_best_r2 = winner_data['mean_r2']
            
            # Update the main tree with final results  
            if HAS_RICH and self.phase1_node:
                # Update models tested count to final number
                models_tested_text = f"Models tested: {len(self.model_family_results)}/{len(self.get_model_families())} ✅ (Complete)"
                self.phase1_node._children[0]._text = models_tested_text
                console.print(self.main_tree)
            
            self.logger.info(f"Phase 1 Winner: {winner_name} (R² = {self.current_best_r2:.4f})")
            
            return winner_name, self.current_best_r2
        else:
            # Fallback to Ridge if all models failed
            self.winning_model_class = Ridge
            self.winning_model_params = {'alpha': 1.0}
            self.current_best_r2 = 0.0
            
            self.logger.warning("All models failed, falling back to Ridge")
            
            return 'Ridge', 0.0

    def create_comprehensive_status_tree(self, phase="phase1"):
        """Create comprehensive hierarchical tree showing system state across all phases"""
        if not HAS_RICH:
            return
        
        tree = Tree("⚙️ SYSTEMATIC ML OPTIMIZER v1.2")
        
        # Dataset Information
        if hasattr(self, 'X_train') and self.X_train is not None:
            data_node = tree.add(f"📂 Dataset: Hold-{self.dataset_num}")
            total_samples = len(self.X_train) + len(self.X_test)
            data_node.add(f"Total samples: {total_samples}")
            data_node.add(f"Features: {self.X_train.shape[1]} {'(High Dimensionality ⚠️)' if self.X_train.shape[1] > 1000 else ''}")
            data_node.add(f"Feature-to-sample ratio: 1:{self.X_train.shape[1]/len(self.X_train):.1f}")
            data_node.add(f"Train/Test split: {len(self.X_train)}/{len(self.X_test)}")
        
        # Phase 1: Model Family Tournament
        phase1 = tree.add("🚩 Phase 1: Model Family Tournament")
        if hasattr(self, 'model_family_results') and self.model_family_results:
            actual_count = len(self.model_family_results)
            base_count = 11
            optional_count = actual_count - base_count
            
            phase1.add(f"Models tested: {actual_count} (Base: {base_count}, Optional: {optional_count})")
            
            if hasattr(self, 'cv') and self.cv:
                total_folds = self.cv.n_splits * self.cv.n_repeats
                phase1.add(f"Cross-validation: {self.cv.n_splits}-fold × {self.cv.n_repeats} repeats ({total_folds} folds)")
            
            # Results with categorization
            results = phase1.add("📊 Results by Performance:")
            sorted_results = sorted(self.model_family_results.items(), key=lambda x: x[1]['mean_r2'], reverse=True)
            
            # Categorize results
            positive_results = [(n, d) for n, d in sorted_results if d['mean_r2'] >= 0]
            negative_results = [(n, d) for n, d in sorted_results if d['mean_r2'] < 0]
            extreme_results = [(n, d) for n, d in sorted_results if d['mean_r2'] < -10]
            
            if positive_results:
                pos_node = results.add(f"✅ Positive R² ({len(positive_results)} models)")
                for name, data in positive_results[:3]:  # Top 3
                    pos_node.add(f"{name}: R² = {data['mean_r2']:.4f} ± {data['std_r2']:.4f}")
            
            if negative_results and not extreme_results:
                neg_node = results.add(f"❌ Negative R² ({len(negative_results)} models)")
                for name, data in negative_results[:3]:
                    neg_node.add(f"{name}: R² = {data['mean_r2']:.4f}")
            
            if extreme_results:
                extreme_node = results.add(f"🚨 Extreme Negative R² ({len(extreme_results)} models)")
                extreme_node.add("⚠️  Raw data likely requires preprocessing")
                for name, data in extreme_results[:3]:
                    extreme_node.add(f"{name}: R² = {data['mean_r2']:.1f}")
            
            # Winner
            winner_name = sorted_results[0][0]
            winner_r2 = sorted_results[0][1]['mean_r2']
            results.add(f"🏆 Winner: {winner_name} (R² = {winner_r2:.4f})")
            
        else:
            phase1.add("Status: Not started")
        
        # Noise Ceiling Information
        if hasattr(self, 'noise_ceiling') and self.noise_ceiling and hasattr(self, 'ceiling_details'):
            ceiling_node = tree.add("📏 Noise Ceiling Analysis")
            ceiling_node.add(f"RidgeCV baseline: {self.ceiling_details.get('mean', 'N/A'):.4f}")
            ceiling_node.add(f"Calculated ceiling: {self.noise_ceiling:.4f}")
            
            if self.noise_ceiling > 0.9:
                ceiling_node.add("✅ EXCELLENT potential (>0.9)")
            elif self.noise_ceiling > 0.7:
                ceiling_node.add("⚠️  MODERATE potential (0.7-0.9)")
            else:
                ceiling_node.add("❌ LOW potential (<0.7)")
        
        # Phase 2: Iterative Preprocessing Optimization
        phase2 = tree.add("🚩 Phase 2: Iterative Preprocessing Optimization")
        if hasattr(self, 'preprocessing_stack') and len(self.preprocessing_stack) > 0:
            phase2.add(f"Methods in stack: {len(self.preprocessing_stack)}")
            stack_node = phase2.add("📚 Current Stack:")
            for i, (method_name, _) in enumerate(self.preprocessing_stack, 1):
                stack_node.add(f"{i}. {method_name}")
            
            if hasattr(self, 'preprocessing_history') and self.preprocessing_history:
                history_node = phase2.add("📈 Optimization History:")
                for hist in self.preprocessing_history:
                    improvement = hist['improvement']
                    history_node.add(f"Iter {hist['iteration']}: {hist['method']} (+{improvement:+.4f})")
        else:
            phase2.add("Status: Not started")
            phase2.add("Methods to test: 8 preprocessing methods")
            phase2.add("Trials per method: 50")
        
        # Phase 3: Deep Hyperparameter Optimization
        phase3 = tree.add("🚩 Phase 3: Deep Hyperparameter Optimization")
        if hasattr(self, 'final_pipeline') and self.final_pipeline:
            phase3.add(f"Status: Complete")
            phase3.add(f"Final R²: {self.current_best_r2:.4f}")
            if hasattr(self, 'noise_ceiling') and self.noise_ceiling:
                gap = abs(self.current_best_r2 - self.noise_ceiling)
                phase3.add(f"Gap to ceiling: {gap:.4f}")
        else:
            phase3.add("Status: Not started")
            phase3.add("Trials planned: 500")
            if hasattr(self, 'noise_ceiling') and self.noise_ceiling:
                phase3.add(f"Target ceiling: {self.noise_ceiling:.4f}")
        
        console.print("\n")
        console.print(tree)
        console.print("\n")

    def calculate_noise_ceiling(self):
        """Calculate noise ceiling using RidgeCV as a robust baseline estimator"""
        
        # Create noise ceiling calculation tree
        ceiling_tree = Tree(f"{Colors.BOLD}{Colors.CYAN}📏 EXPLICIT NOISE CEILING CALCULATION{Colors.END}")
        
        # Methodology section
        method_node = ceiling_tree.add("🔬 Methodology")
        method_node.add("Estimator: RidgeCV with basic preprocessing")
        method_node.add("Purpose: Theoretical maximum performance baseline")
        method_node.add("Rationale: Establishes upper bound for Phase 3 optimization")
        
        # Pipeline details
        pipeline_node = ceiling_tree.add("⚙️ Pipeline Configuration")
        pipeline_node.add("1. VarianceThreshold(1e-8) - Remove zero-variance features")
        pipeline_node.add("2. RobustScaler() - Scale features robustly")
        pipeline_node.add("3. RidgeCV(alphas=20 values) - Cross-validated ridge regression")
        
        if HAS_RICH:
            console.print(ceiling_tree)
        
        # Use RidgeCV as noise ceiling estimator with robust preprocessing
        ceiling_pipeline = Pipeline([
            ('var_threshold', VarianceThreshold(1e-8)),
            ('scaler', RobustScaler()),
            ('ridge', RidgeCV(alphas=np.logspace(-3, 3, 20)))
        ])
        
        self.logger.info("=" * 50)
        self.logger.info("EXPLICIT NOISE CEILING CALCULATION")
        self.logger.info("=" * 50)
        self.logger.info("Using RidgeCV + RobustScaler as theoretical maximum baseline")
        
        # Create progress tree for calculation
        progress_tree = Tree("🔄 Running Noise Ceiling Calculation")
        progress_tree.add(f"Cross-validation: {self.cv_n_splits}-fold × {self.cv_n_repeats} repeats")
        progress_tree.add("Estimating theoretical maximum performance...")
        
        if HAS_RICH:
            console.print(progress_tree)
        
        try:
            cv_scores = cross_val_score(
                ceiling_pipeline, self.X_train, self.y_train,
                cv=self.cv, scoring='r2', n_jobs=12
            )
            cv_scores = cv_scores[np.isfinite(cv_scores)]
            
            if len(cv_scores) > 0:
                mean_score = np.mean(cv_scores)
                std_score = np.std(cv_scores)
                min_score = np.min(cv_scores)
                max_score = np.max(cv_scores)
                
                # Noise ceiling = mean + 2*std (theoretical maximum with confidence interval)
                self.noise_ceiling = mean_score + 2 * std_score
                
                # Create results tree
                results_tree = Tree("📊 NOISE CEILING CALCULATION RESULTS")
                
                # RidgeCV baseline results
                baseline_node = results_tree.add("🏁 RidgeCV Baseline Performance")
                baseline_node.add(f"Mean R²: {mean_score:.4f}")
                baseline_node.add(f"Std R²: {std_score:.4f}")
                baseline_node.add(f"Min R²: {min_score:.4f}")
                baseline_node.add(f"Max R²: {max_score:.4f}")
                baseline_node.add(f"Range: [{min_score:.4f}, {max_score:.4f}]")
                
                # Calculated noise ceiling
                ceiling_node = results_tree.add(f"🎯 CALCULATED NOISE CEILING: {Colors.BOLD}{self.noise_ceiling:.4f}{Colors.END}")
                ceiling_node.add(f"Formula: mean + 2×std")
                ceiling_node.add(f"Calculation: {mean_score:.4f} + 2×{std_score:.4f} = {self.noise_ceiling:.4f}")
                ceiling_node.add(f"Target for Phase 3: Approach {self.noise_ceiling:.4f}")
                
                # Quality assessment
                assessment_node = results_tree.add("🔍 Quality Assessment")
                if self.noise_ceiling > 0.9:
                    quality_msg = "✅ EXCELLENT (>0.9) - High predictive potential"
                    assessment_node.add(quality_msg)
                    assessment_node.add("Dataset shows strong signal-to-noise ratio")
                elif self.noise_ceiling > 0.7:
                    quality_msg = "⚠️  MODERATE (0.7-0.9) - Good but limited potential"
                    assessment_node.add(quality_msg)
                    assessment_node.add("Dataset has moderate predictive potential")
                elif self.noise_ceiling > 0.5:
                    quality_msg = "❌ LOW (0.5-0.7) - Challenging dataset"
                    assessment_node.add(quality_msg)
                    assessment_node.add("Dataset will be difficult to predict accurately")
                else:
                    quality_msg = "❌ VERY LOW (<0.5) - Extremely difficult dataset"
                    assessment_node.add(quality_msg)
                    assessment_node.add("Dataset has very poor predictive potential")
                
                if HAS_RICH:
                    console.print(results_tree)
                
                # Store detailed ceiling info for later reference
                self.ceiling_details = {
                    'mean': mean_score,
                    'std': std_score,
                    'min': min_score,
                    'max': max_score,
                    'ceiling': self.noise_ceiling
                }
                
                self.logger.info(f"NOISE CEILING CALCULATED: {self.noise_ceiling:.4f}")
                self.logger.info(f"RidgeCV baseline: mean={mean_score:.4f}, std={std_score:.4f}")
                self.logger.info(f"Quality assessment: {quality_msg}")
                
                return self.noise_ceiling
            else:
                # Failure tree
                failure_tree = Tree(f"{Colors.RED}❌ NOISE CEILING CALCULATION FAILED{Colors.END}")
                failure_tree.add("No valid CV scores from RidgeCV baseline")
                failure_tree.add("Using conservative default ceiling: 0.95")
                failure_tree.add("⚠️  This may affect Phase 3 optimization targets")
                
                if HAS_RICH:
                    console.print(failure_tree)
                
                self.noise_ceiling = 0.95
                self.ceiling_details = {'mean': 0.0, 'std': 0.0, 'ceiling': 0.95}
                self.logger.warning("No valid CV scores, using default noise ceiling: 0.95")
                return self.noise_ceiling
                
        except Exception as e:
            # Error tree
            error_tree = Tree(f"{Colors.RED}💥 NOISE CEILING CALCULATION ERROR{Colors.END}")
            error_tree.add(f"Error: {str(e)[:100]}...")
            error_tree.add("Falling back to conservative default: 0.95")
            error_tree.add("⚠️  Phase 3 optimization may be suboptimal")
            
            if HAS_RICH:
                console.print(error_tree)
            
            self.logger.error(f"Noise ceiling calculation failed: {e}")
            self.noise_ceiling = 0.95
            self.ceiling_details = {'mean': 0.0, 'std': 0.0, 'ceiling': 0.95}
            return self.noise_ceiling 

    def phase_2_iterative_preprocessing_optimization(self):
        """
        Phase 2: Iterative Preprocessing Stack Optimization
        Systematically test each of 8 preprocessing methods and add beneficial ones to the stack
        """
        
        # Create Phase 2 tree structure
        phase2_tree = Tree(f"{Colors.BOLD}{Colors.PURPLE}🔧 PHASE 2: ITERATIVE PREPROCESSING STACK OPTIMIZATION{Colors.END}")
        
        # Get all preprocessing methods
        preprocessing_methods = self.get_preprocessing_methods()
        
        # Setup section
        setup_node = phase2_tree.add("⚙️ Setup & Strategy")
        setup_node.add("Strategy: Systematic iterative optimization")
        setup_node.add("Method: Add 1 preprocessing method at a time")
        setup_node.add("Trials per method: 50 with Optuna TPE + Median pruning")
        setup_node.add(f"Baseline R² (raw data): {self.current_best_r2:.4f}")
        
        # Methods to test
        methods_node = phase2_tree.add(f"🧪 Methods to Test ({len(preprocessing_methods)} total)")
        for i, method_name in enumerate(preprocessing_methods.keys(), 1):
            methods_node.add(f"{i}. {method_name}")
        
        # Current stack status
        stack_node = phase2_tree.add(f"📚 Current Stack ({len(self.preprocessing_stack)} methods)")
        if self.preprocessing_stack:
            for i, (method_name, _) in enumerate(self.preprocessing_stack, 1):
                stack_node.add(f"{i}. {method_name}")
        else:
            stack_node.add("Empty - starting with raw data")
        
        if HAS_RICH:
            console.print(phase2_tree)
        
        self.logger.info("=" * 75)
        self.logger.info("PHASE 2: Iterative Preprocessing Stack Optimization")
        self.logger.info("=" * 75)
        
        # Track which methods have been tested
        tested_methods = set()
        iteration = 0
        min_improvement_threshold = 0.005  # Minimum R² improvement to add a method
        
        while len(tested_methods) < len(preprocessing_methods):
            iteration += 1
            
            # Create iteration tree
            iter_tree = Tree(f"🔄 Preprocessing Iteration {iteration}")
            
            status_node = iter_tree.add("📊 Current Status")
            status_node.add(f"Best R²: {self.current_best_r2:.4f}")
            status_node.add(f"Stack size: {len(self.preprocessing_stack)}")
            status_node.add(f"Methods tested: {len(tested_methods)}/{len(preprocessing_methods)}")
            
            if HAS_RICH:
                console.print(iter_tree)
            
            best_candidate = None
            best_candidate_r2 = self.current_best_r2
            best_candidate_params = None
            
            # Test each untested preprocessing method with clear progress tracking
            for i, (method_name, method_config) in enumerate(preprocessing_methods.items(), 1):
                if method_name in tested_methods:
                    continue
                
                # Create method testing tree
                method_tree = Tree(f"🔧 Testing {method_name}")
                
                setup_node = method_tree.add("⚙️ Setup")
                setup_node.add(f"Method {i}/{len(preprocessing_methods)}")
                setup_node.add(f"Trials: {self.max_preprocessing_trials}")
                setup_node.add(f"Current baseline: {self.current_best_r2:.4f}")
                
                if HAS_RICH:
                    console.print(method_tree)
                
                # Create Optuna study for this preprocessing method
                study = optuna.create_study(
                    direction='maximize',
                    pruner=MedianPruner(n_startup_trials=5),
                    sampler=optuna.samplers.TPESampler(seed=42)
                )
                
                def objective(trial):
                    return self._optimize_single_preprocessing_method(
                        trial, method_name, method_config
                    )
                
                # Optimize this preprocessing method
                try:
                    study.optimize(
                        objective,
                        n_trials=self.max_preprocessing_trials,
                        show_progress_bar=False
                    )
                    
                    if len(study.trials) > 0:
                        method_r2 = study.best_value
                        improvement = method_r2 - self.current_best_r2
                        
                        # Create results tree
                        result_tree = Tree(f"📊 {method_name} Results")
                        
                        perf_node = result_tree.add("🎯 Performance")
                        perf_node.add(f"Best R²: {method_r2:.4f}")
                        perf_node.add(f"Improvement: {improvement:+.4f}")
                        
                        if method_r2 > best_candidate_r2:
                            best_candidate = method_name
                            best_candidate_r2 = study.best_value
                            best_candidate_params = study.best_params
                            
                            leader_node = result_tree.add("✨ NEW ITERATION LEADER")
                            leader_node.add(f"Best params: {study.best_params}")
                            
                        elif improvement > 0:
                            result_tree.add("✅ Minor improvement")
                        else:
                            result_tree.add("❌ No improvement")
                        
                        if HAS_RICH:
                            console.print(result_tree)
                    else:
                        fail_tree = Tree(f"❌ {method_name} Failed")
                        fail_tree.add("No trials completed successfully")
                        
                        if HAS_RICH:
                            console.print(fail_tree)
                
                except Exception as e:
                    error_tree = Tree(f"💥 {method_name} Error")
                    error_tree.add(f"Exception: {str(e)[:50]}...")
                    
                    if HAS_RICH:
                        console.print(error_tree)
                    
                    self.logger.error(f"Error optimizing {method_name}: {e}")
                
                # Mark as tested
                tested_methods.add(method_name)
            
            # Check if we found a beneficial preprocessing method
            improvement = best_candidate_r2 - self.current_best_r2
            
            if best_candidate is not None and improvement >= min_improvement_threshold:
                # Add the best preprocessing method to the stack
                method_config = preprocessing_methods[best_candidate]
                
                # Create the transformer with optimal parameters
                transformer = self._create_transformer_with_params(
                    method_config['transformer'], 
                    best_candidate_params,
                    best_candidate
                )
                
                self.preprocessing_stack.append((best_candidate, transformer))
                self.current_best_r2 = best_candidate_r2
                
                self.preprocessing_history.append({
                    'iteration': iteration,
                    'method': best_candidate,
                    'params': best_candidate_params,
                    'r2_before': self.current_best_r2 - improvement,
                    'r2_after': self.current_best_r2,
                    'improvement': improvement
                })
                
                # Create success tree
                success_tree = Tree(f"✅ ADDED TO STACK: {best_candidate}")
                
                improvement_node = success_tree.add("📈 Improvement")
                improvement_node.add(f"R² improvement: {improvement:+.4f}")
                improvement_node.add(f"New best R²: {self.current_best_r2:.4f}")
                
                config_node = success_tree.add("⚙️ Configuration")
                for param, value in best_candidate_params.items():
                    if isinstance(value, float):
                        config_node.add(f"{param}: {value:.4f}")
                    else:
                        config_node.add(f"{param}: {value}")
                
                if HAS_RICH:
                    console.print(success_tree)
                
                self.logger.info(f"Added {best_candidate} to preprocessing stack (R² = {self.current_best_r2:.4f})")
                
                # Reset tested methods to test remaining methods with new stack
                tested_methods = set()
                
            else:
                # No significant improvement found, stop iteration
                stop_tree = Tree("🛑 Stopping Preprocessing Optimization")
                
                if best_candidate is not None:
                    reason_node = stop_tree.add("⚠️ Insufficient Improvement")
                    reason_node.add(f"Best improvement: {improvement:+.4f}")
                    reason_node.add(f"Required threshold: {min_improvement_threshold}")
                else:
                    stop_tree.add("⚠️ No methods provided improvement")
                
                if HAS_RICH:
                    console.print(stop_tree)
                break
        
        # Create final stack summary tree
        final_stack_tree = Tree("📋 FINAL PREPROCESSING STACK")
        
        if self.preprocessing_stack:
            summary_node = final_stack_tree.add(f"✅ Stack Complete: {len(self.preprocessing_stack)} methods")
            summary_node.add(f"Final R²: {self.current_best_r2:.4f}")
            
            baseline_r2 = self.model_family_results[list(self.model_family_results.keys())[0]]['mean_r2']
            total_improvement = self.current_best_r2 - baseline_r2
            summary_node.add(f"Total improvement: {total_improvement:+.4f}")
            
            methods_node = final_stack_tree.add("🔧 Methods in Stack")
            for i, (method_name, _) in enumerate(self.preprocessing_stack, 1):
                methods_node.add(f"{i}. {method_name}")
        else:
            final_stack_tree.add("⚠️ No preprocessing methods added")
            final_stack_tree.add("Raw data was optimal")
        
        if HAS_RICH:
            console.print(final_stack_tree)
        
        self.logger.info(f"Phase 2 complete: {len(self.preprocessing_stack)} methods in stack, R² = {self.current_best_r2:.4f}")
        
        return self.preprocessing_stack

    def _optimize_single_preprocessing_method(self, trial, method_name, method_config):
        """Optimize a single preprocessing method with Optuna"""
        try:
            transformer_class = method_config['transformer']
            param_space = method_config['param_space']
            
            # Build parameter dictionary for this trial
            trial_params = {}
            for param_name, param_config in param_space.items():
                if isinstance(param_config, tuple) and len(param_config) == 2:
                    # Range parameter (min, max)
                    if isinstance(param_config[0], int):
                        trial_params[param_name] = trial.suggest_int(param_name, param_config[0], param_config[1])
                    else:
                        trial_params[param_name] = trial.suggest_float(param_name, param_config[0], param_config[1])
                elif isinstance(param_config, list):
                    # Categorical parameter
                    trial_params[param_name] = trial.suggest_categorical(param_name, param_config)
            
            # Create transformer with trial parameters
            transformer = self._create_transformer_with_params(transformer_class, trial_params, method_name)
            
            # Build pipeline with current preprocessing stack + new method + winning model
            pipeline_steps = [('var_threshold', VarianceThreshold(1e-8))]
            
            # Add existing preprocessing stack
            for step_name, step_transformer in self.preprocessing_stack:
                pipeline_steps.append((f'preprocess_{step_name}', step_transformer))
            
            # Add the new method being tested
            pipeline_steps.append((f'new_{method_name}', transformer))
            
            # Add the winning model
            model = self.winning_model_class(**self.winning_model_params)
            pipeline_steps.append(('model', model))
            
            # Create and evaluate pipeline
            pipeline = Pipeline(pipeline_steps)
            
            # Cross-validation with 12 threads
            cv_scores = cross_val_score(
                pipeline, self.X_train, self.y_train,
                cv=self.cv, scoring='r2', n_jobs=12
            )
            
            cv_scores = cv_scores[np.isfinite(cv_scores)]
            
            if len(cv_scores) == 0:
                raise optuna.exceptions.TrialPruned()
            
            mean_r2 = np.mean(cv_scores)
            
            # Early pruning if performance is worse than current best
            if mean_r2 < self.current_best_r2 - 0.02:  # Allow some tolerance
                raise optuna.exceptions.TrialPruned()
            
            return mean_r2
            
        except Exception as e:
            self.logger.debug(f"Trial failed for {method_name}: {e}")
            raise optuna.exceptions.TrialPruned()

    def _create_transformer_with_params(self, transformer_class, params, method_name):
        """Create transformer instance with given parameters"""
        try:
            # Handle special cases for certain transformers
            if method_name == 'FeatureSelection' and transformer_class == SelectKBest:
                # SelectKBest needs score_func parameter
                return transformer_class(
                    score_func=params.get('score_func', mutual_info_regression),
                    k=params.get('k', 50)
                )
            elif method_name == 'PCA':
                return transformer_class(
                    n_components=params.get('n_components', 20),
                    random_state=42
                )
            elif method_name in ['KMeansOutlier', 'IsolationForest', 'LocalOutlierFactor']:
                # Outlier detection transformers
                return transformer_class(**params)
            elif method_name == 'QuantileTransform':
                return transformer_class(
                    output_distribution=params.get('output_distribution', 'uniform'),
                    n_quantiles=params.get('n_quantiles', 1000),
                    random_state=42
                )
            else:
                # Standard transformers
                return transformer_class(**params)
                
        except Exception as e:
            self.logger.error(f"Error creating transformer {method_name}: {e}")
            # Return a default instance
            if hasattr(transformer_class, '__init__'):
                try:
                    return transformer_class()
                except:
                    return transformer_class(**self.get_preprocessing_methods()[method_name]['param_space']) 

    def phase_3_deep_hyperparameter_optimization(self):
        """
        Phase 3: Deep Model Hyperparameter Optimization
        Extensive hyperparameter tuning of the winning model with optimized preprocessing stack
        """
        
        # Create Phase 3 header tree
        phase3_header = Tree("🔥 PHASE 3: DEEP HYPERPARAMETER OPTIMIZATION")
        
        # System configuration
        config_node = phase3_header.add("⚙️ Configuration")
        config_node.add(f"Target model: {self.winning_model_class.__name__}")
        config_node.add(f"Preprocessing stack: {len(self.preprocessing_stack)} methods")
        config_node.add(f"Current best R²: {self.current_best_r2:.4f}")
        config_node.add(f"Noise ceiling: {self.noise_ceiling:.4f}")
        
        if HAS_RICH:
            console.print(phase3_header)
        
        self.logger.info("=" * 70)
        self.logger.info("PHASE 3: Deep Model Hyperparameter Optimization")
        self.logger.info("=" * 70)
        self.logger.info(f"Target model: {self.winning_model_class.__name__}")
        self.logger.info(f"Preprocessing stack size: {len(self.preprocessing_stack)}")
        self.logger.info(f"Current best R²: {self.current_best_r2:.4f}")
        self.logger.info(f"Noise ceiling: {self.noise_ceiling:.4f}")
        
        # Create Optuna study for deep hyperparameter optimization
        study = optuna.create_study(
            direction='maximize',
            pruner=MedianPruner(n_startup_trials=20),
            sampler=optuna.samplers.TPESampler(seed=42, multivariate=True)
        )
        
        # Create Phase 3 setup tree
        phase3_setup = Tree("🔥 Phase 3: Deep Hyperparameter Optimization Setup")
        
        setup_node = phase3_setup.add("⚙️ Configuration")
        setup_node.add(f"Current R² with preprocessing: {self.current_best_r2:.4f}")
        setup_node.add(f"Noise ceiling target: {self.noise_ceiling:.4f}")
        setup_node.add(f"Maximum trials: {self.max_hyperopt_trials}")
        setup_node.add("Goal: Approach noise ceiling through hyperparameter tuning")
        
        if HAS_RICH:
            console.print(phase3_setup)
        
        # Track progress with explicit stopping criteria
        trials_completed = 0
        best_improvement = 0.0
        near_ceiling_threshold = 0.02  # Within 2% of noise ceiling
        excellent_threshold = 0.01    # Within 1% of noise ceiling
        early_stop_triggered = False
        
        def progress_callback(study, trial):
            nonlocal trials_completed, best_improvement, early_stop_triggered
            trials_completed += 1
            
            if trial.state == optuna.trial.TrialState.COMPLETE:
                improvement = study.best_value - self.current_best_r2
                if improvement > best_improvement:
                    best_improvement = improvement
                
                # Set console title with ceiling proximity
                ceiling_gap = abs(study.best_value - self.noise_ceiling) if self.noise_ceiling else float('inf')
                set_console_title(f"Phase 3 - R²: {study.best_value:.4f} | Gap: {ceiling_gap:.4f} | Trial {trials_completed}/{self.max_hyperopt_trials}")
                
                # Create trial result tree (for significant improvements only)
                if improvement > 0.001:  # Only show tree for meaningful improvements
                    trial_tree = Tree(f"🔥 Trial {trials_completed}: Significant Improvement")
                    
                    result_node = trial_tree.add("📊 Performance")
                    result_node.add(f"Trial R²: {trial.value:.4f}")
                    result_node.add(f"Best R²: {study.best_value:.4f}")
                    result_node.add(f"Improvement: +{improvement:.4f}")
                    
                    # EXPLICIT STOPPING CRITERIA linked to noise ceiling
                    if self.noise_ceiling:
                        ceiling_gap = abs(study.best_value - self.noise_ceiling)
                        
                        ceiling_node = trial_tree.add("🎯 Ceiling Analysis")
                        ceiling_node.add(f"Current R²: {study.best_value:.4f}")
                        ceiling_node.add(f"Noise ceiling: {self.noise_ceiling:.4f}")
                        ceiling_node.add(f"Gap: {ceiling_gap:.4f}")
                        
                        if ceiling_gap <= excellent_threshold:
                            status_node = trial_tree.add("🎉 EXCELLENT: Within 1% of ceiling!")
                            status_node.add("Theoretical maximum nearly achieved")
                            status_node.add("Optimization highly successful")
                            
                        elif ceiling_gap <= near_ceiling_threshold:
                            status_node = trial_tree.add("✅ NEAR CEILING: Within 2% of maximum")
                            status_node.add("Approaching theoretical limits")
                    
                    if HAS_RICH:
                        console.print(trial_tree)
                    
            elif trial.state == optuna.trial.TrialState.PRUNED:
                # Brief status update for pruned trials
                if trials_completed % 10 == 0:  # Every 10th trial
                    status_tree = Tree(f"✂️ Trial {trials_completed}: Pruned (showing every 10th)")
                    if HAS_RICH:
                        console.print(status_tree)
                        
            elif trial.state == optuna.trial.TrialState.FAIL:
                # Show failed trials
                fail_tree = Tree(f"💥 Trial {trials_completed}: Failed")
                if HAS_RICH:
                    console.print(fail_tree)
        
        # Create optimization start tree
        start_tree = Tree("🚀 Starting Deep Hyperparameter Optimization")
        start_tree.add("Initializing Optuna study with TPE sampler")
        start_tree.add("Beginning systematic hyperparameter search")
        
        if HAS_RICH:
            console.print(start_tree)
        
        try:
            study.optimize(
                self._deep_hyperopt_objective,
                n_trials=self.max_hyperopt_trials,
                callbacks=[progress_callback],
                show_progress_bar=False
            )
        except KeyboardInterrupt:
            interrupt_tree = Tree("⚠️ Optimization Interrupted")
            interrupt_tree.add("User requested early termination")
            interrupt_tree.add("Partial results will be analyzed")
            
            if HAS_RICH:
                console.print(interrupt_tree)
            
            self.logger.info("Deep optimization interrupted by user")
        
        # Results analysis
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if completed_trials:
            best_r2 = study.best_value
            improvement = best_r2 - self.current_best_r2
            
            # Create Phase 3 results tree
            results_tree = Tree("📊 PHASE 3 RESULTS")
            
            performance_node = results_tree.add("🎯 Performance Summary")
            performance_node.add(f"Trials completed: {len(completed_trials)}")
            performance_node.add(f"Best R²: {best_r2:.4f}")
            performance_node.add(f"Improvement from preprocessing: {improvement:+.4f}")
            
            params_node = results_tree.add("⚙️ Best Configuration")
            for param, value in study.best_params.items():
                if isinstance(value, float):
                    params_node.add(f"{param}: {value:.4f}")
                else:
                    params_node.add(f"{param}: {value}")
            
            # Build final pipeline
            self.final_pipeline = self._build_final_pipeline(study.best_params)
            
            # Save the final model
            model_path = self.model_dir / f"hold{self.dataset_num}_final_model.pkl"
            joblib.dump(self.final_pipeline, model_path)
            
            save_node = results_tree.add("💾 Model Saved")
            save_node.add(f"Location: {model_path}")
            save_node.add("Pipeline ready for evaluation")
            
            if HAS_RICH:
                console.print(results_tree)
            
            self.logger.info(f"Phase 3 complete: R² = {best_r2:.4f}, improvement = {improvement:+.4f}")
            self.logger.info(f"Final model saved to: {model_path}")
            
            return best_r2, study.best_params
        else:
            # Create failure tree
            failure_tree = Tree("❌ Phase 3 Failed")
            failure_tree.add("No trials completed successfully")
            failure_tree.add("Returning preprocessing baseline")
            
            if HAS_RICH:
                console.print(failure_tree)
            
            self.logger.error("No trials completed successfully in Phase 3")
            return self.current_best_r2, {}

    def _deep_hyperopt_objective(self, trial):
        """Objective function for deep hyperparameter optimization"""
        try:
            # Get hyperparameter suggestions based on winning model class
            model_params = self._suggest_model_hyperparameters(trial)
            
            # Create model with suggested parameters
            model = self.winning_model_class(**model_params)
            
            # Build pipeline with optimized preprocessing stack + model
            pipeline_steps = [('var_threshold', VarianceThreshold(1e-8))]
            
            # Add optimized preprocessing stack
            for step_name, step_transformer in self.preprocessing_stack:
                pipeline_steps.append((f'preprocess_{step_name}', step_transformer))
            
            # Add the model
            pipeline_steps.append(('model', model))
            
            # Create pipeline
            pipeline = Pipeline(pipeline_steps)
            
            # Cross-validation with 12 threads
            cv_scores = cross_val_score(
                pipeline, self.X_train, self.y_train,
                cv=self.cv, scoring='r2', n_jobs=12
            )
            
            cv_scores = cv_scores[np.isfinite(cv_scores)]
            
            if len(cv_scores) == 0:
                raise optuna.exceptions.TrialPruned()
            
            mean_r2 = np.mean(cv_scores)
            
            # Aggressive pruning for deep optimization
            if mean_r2 < self.current_best_r2 - 0.01:
                raise optuna.exceptions.TrialPruned()
            
            return mean_r2
            
        except Exception as e:
            self.logger.debug(f"Deep hyperopt trial failed: {e}")
            raise optuna.exceptions.TrialPruned()

    def _suggest_model_hyperparameters(self, trial):
        """Suggest hyperparameters based on the winning model class"""
        if self.winning_model_class == Ridge:
            return {
                'alpha': trial.suggest_float('alpha', 1e-4, 100, log=True)
            }
        
        elif self.winning_model_class == Lasso:
            return {
                'alpha': trial.suggest_float('alpha', 1e-4, 10, log=True),
                'max_iter': trial.suggest_int('max_iter', 1000, 5000)
            }
        
        elif self.winning_model_class == ElasticNet:
            return {
                'alpha': trial.suggest_float('alpha', 1e-4, 10, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
                'max_iter': trial.suggest_int('max_iter', 1000, 5000)
            }
        
        elif self.winning_model_class == SVR:
            kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'linear'])
            params = {
                'C': trial.suggest_float('C', 0.01, 1000, log=True),
                'kernel': kernel
            }
            if kernel == 'rbf':
                params['gamma'] = trial.suggest_float('gamma', 1e-5, 1, log=True)
            elif kernel == 'poly':
                params['gamma'] = trial.suggest_float('gamma', 1e-5, 1, log=True)
                params['degree'] = trial.suggest_int('degree', 2, 5)
            return params
        
        elif self.winning_model_class == RandomForestRegressor:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42,
                'n_jobs': 1
            }
        
        elif self.winning_model_class == ExtraTreesRegressor:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42,
                'n_jobs': 1
            }
        
        elif self.winning_model_class == GradientBoostingRegressor:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42
            }
        
        elif self.winning_model_class == AdaBoostRegressor:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 2.0),
                'loss': trial.suggest_categorical('loss', ['linear', 'square', 'exponential']),
                'random_state': 42
            }
        
        elif self.winning_model_class == MLPRegressor:
            architecture = trial.suggest_categorical('architecture', [
                (50,), (100,), (128,), (256,),
                (50, 25), (100, 50), (128, 64), (256, 128),
                (100, 50, 25), (128, 64, 32), (256, 128, 64)
            ])
            return {
                'hidden_layer_sizes': architecture,
                'alpha': trial.suggest_float('alpha', 1e-6, 1e-2, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-5, 1e-1, log=True),
                'max_iter': trial.suggest_int('max_iter', 200, 1000),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs']),
                'early_stopping': True,
                'random_state': 42
            }
        
        elif self.winning_model_class == DecisionTreeRegressor:
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': 42
            }
        
        elif HAS_XGB and self.winning_model_class == xgb.XGBRegressor:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42,
                'n_jobs': 1
            }
        
        elif HAS_LGB and self.winning_model_class == lgb.LGBMRegressor:
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.5),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42,
                'n_jobs': 1,
                'verbose': -1
            }
        
        else:
            # Fallback to basic Ridge parameters
            return {'alpha': trial.suggest_float('alpha', 1e-4, 100, log=True)}

    def _build_final_pipeline(self, best_params):
        """Build the final optimized pipeline"""
        # Build pipeline steps
        pipeline_steps = [('var_threshold', VarianceThreshold(1e-8))]
        
        # Add optimized preprocessing stack
        for step_name, step_transformer in self.preprocessing_stack:
            pipeline_steps.append((f'preprocess_{step_name}', step_transformer))
        
        # Add optimized model
        optimized_model = self.winning_model_class(**best_params)
        pipeline_steps.append(('model', optimized_model))
        
        # Create and fit pipeline
        pipeline = Pipeline(pipeline_steps)
        pipeline.fit(self.X_train, self.y_train)
        
        return pipeline 

    def final_evaluation(self):
        """Final evaluation on held-out test set"""
        
        # Create final evaluation tree
        eval_tree = Tree("📊 FINAL EVALUATION ON HELD-OUT TEST SET")
        
        if self.final_pipeline is None:
            error_node = eval_tree.add("❌ Evaluation Failed")
            error_node.add("No final pipeline available")
            error_node.add("Previous phases may have failed")
            
            if HAS_RICH:
                console.print(eval_tree)
            return None
        
        # Evaluate on test set
        y_pred = self.final_pipeline.predict(self.X_test)
        
        # Calculate metrics
        test_r2 = r2_score(self.y_test, y_pred)
        test_mae = mean_absolute_error(self.y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        
        # Test set performance
        performance_node = eval_tree.add("📈 Test Set Performance")
        performance_node.add(f"R²: {test_r2:.4f}")
        performance_node.add(f"MAE: {test_mae:.6f}")
        performance_node.add(f"RMSE: {test_rmse:.6f}")
        
        # Compare with noise ceiling
        ceiling_gap = abs(test_r2 - self.noise_ceiling)
        near_ceiling = ceiling_gap < 0.05
        
        ceiling_node = eval_tree.add("🎯 Ceiling Analysis")
        ceiling_node.add(f"Noise ceiling: {self.noise_ceiling:.4f}")
        ceiling_node.add(f"Gap to ceiling: {ceiling_gap:.4f}")
        ceiling_node.add(f"Near ceiling: {'✅ YES' if near_ceiling else '❌ NO'}")
        
        # Performance assessment
        if near_ceiling:
            assessment_node = eval_tree.add("🎉 EXCELLENT: Near theoretical maximum!")
            assessment_node.add("Optimization highly successful")
        elif test_r2 > 0.8:
            assessment_node = eval_tree.add("✅ GOOD: Strong predictive performance")
            assessment_node.add("Results exceed expectations")
        elif test_r2 > 0.5:
            assessment_node = eval_tree.add("⚠️ MODERATE: Decent performance")
            assessment_node.add("Room for improvement exists")
        else:
            assessment_node = eval_tree.add("❌ POOR: Low predictive performance")
            assessment_node.add("Significant optimization challenges")
        
        if HAS_RICH:
            console.print(eval_tree)
        
        self.logger.info(f"Final evaluation: R² = {test_r2:.4f}, MAE = {test_mae:.6f}, RMSE = {test_rmse:.6f}")
        
        # Save detailed results
        results = {
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'noise_ceiling': self.noise_ceiling,
            'near_ceiling': near_ceiling,
            'winning_model': self.winning_model_class.__name__,
            'preprocessing_stack': [name for name, _ in self.preprocessing_stack],
            'model_family_results': {k: v['mean_r2'] for k, v in self.model_family_results.items()},
            'preprocessing_history': self.preprocessing_history
        }
        
        # Save to file
        results_path = self.model_dir / f"hold{self.dataset_num}_systematic_results.txt"
        with open(results_path, 'w', encoding='utf-8') as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        
        # Output location
        output_node = eval_tree.add("📁 Results Saved")
        output_node.add(f"Location: {results_path}")
        output_node.add("All metrics and configuration stored")
        
        if HAS_RICH:
            console.print(eval_tree)
        
        return results

    def create_comprehensive_summary(self):
        """Create comprehensive summary of all 3 phases"""
        
        # Create comprehensive summary tree
        summary_tree = Tree("📋 SYSTEMATIC OPTIMIZATION SUMMARY")
        
        # Phase 1 Summary
        phase1_node = summary_tree.add("🏟️ Phase 1: Model Family Tournament")
        if self.model_family_results:
            sorted_families = sorted(self.model_family_results.items(), key=lambda x: x[1]['mean_r2'], reverse=True)
            
            results_node = phase1_node.add(f"Results: {len(sorted_families)} models tested")
            for i, (name, data) in enumerate(sorted_families[:5], 1):
                status = " 🏆" if i == 1 else ""
                results_node.add(f"{i}. {name}: {data['mean_r2']:.4f}{status}")
        else:
            phase1_node.add("No results available")
        
        # Phase 2 Summary
        phase2_node = summary_tree.add("🔧 Phase 2: Preprocessing Optimization")
        if self.preprocessing_stack:
            stack_node = phase2_node.add(f"Methods in stack: {len(self.preprocessing_stack)}")
            for i, (method_name, _) in enumerate(self.preprocessing_stack, 1):
                stack_node.add(f"{i}. {method_name}")
            
            if hasattr(self, 'preprocessing_history') and self.preprocessing_history:
                history_node = phase2_node.add("Optimization History")
                for hist in self.preprocessing_history:
                    history_node.add(f"Iter {hist['iteration']}: {hist['method']} (+{hist['improvement']:+.4f})")
        else:
            phase2_node.add("No preprocessing methods added")
            phase2_node.add("Raw data was optimal")
        
        # Phase 3 Summary
        phase3_node = summary_tree.add("🔥 Phase 3: Hyperparameter Optimization")
        phase3_node.add(f"Final model: {self.winning_model_class.__name__}")
        if hasattr(self, 'final_pipeline') and self.final_pipeline:
            phase3_node.add("Pipeline fitted: ✅ YES")
        else:
            phase3_node.add("Pipeline fitted: ❌ NO")
        
        # Overall Performance
        performance_node = summary_tree.add("📊 Overall Performance")
        if self.model_family_results:
            baseline_r2 = list(self.model_family_results.values())[0]['mean_r2']
            total_improvement = self.current_best_r2 - baseline_r2
            performance_node.add(f"Baseline (raw data): {baseline_r2:.4f}")
            performance_node.add(f"Final optimization: {self.current_best_r2:.4f}")
            performance_node.add(f"Total improvement: {total_improvement:+.4f}")
            performance_node.add(f"Noise ceiling: {self.noise_ceiling:.4f}")
        
        if HAS_RICH:
            console.print(summary_tree)

    def run_systematic_optimization(self, X, y):
        """Run the complete 3-phase systematic optimization"""
        
        # Initialize the persistent main tree structure
        self.initialize_main_tree(X, y)
        
        # Set console title
        set_console_title(f"Systematic ML Optimization - Hold-{self.dataset_num}")
        
        start_time = time.time()
        
        try:
            # Phase 1: Model Family Tournament
            winner_name, winner_r2 = self.phase_1_model_family_tournament(X, y)
            
            # Calculate noise ceiling
            self.calculate_noise_ceiling()
            
            # Update Phase 2 with initial setup
            if HAS_RICH and self.phase2_node:
                self.phase2_node.add("Methods tested: 0/8")
                self.phase2_node.add("Trials per method: 50")
                self.phase2_node.add("Stack additions: None yet")
                self.phase2_node.add(f"Current best R²: {self.current_best_r2:.3f} ({self.winning_model_class.__name__})")
                console.print(self.main_tree)
            
            # Phase 2: Iterative Preprocessing Optimization
            preprocessing_stack = self.phase_2_iterative_preprocessing_optimization()
            
            # Update Phase 3 with initial setup
            if HAS_RICH and self.phase3_node:
                if self.noise_ceiling:
                    self.phase3_node.add(f"Noise ceiling: {self.noise_ceiling:.3f}")
                else:
                    self.phase3_node.add("Noise ceiling: Not yet calculated ⚠️")
                self.phase3_node.add("Trials planned: 500")
                console.print(self.main_tree)
            
            # Phase 3: Deep Hyperparameter Optimization  
            final_r2, best_params = self.phase_3_deep_hyperparameter_optimization()
            
            # Final Evaluation
            final_results = self.final_evaluation()
            
            total_time = time.time() - start_time
            
            # Create completion tree
            completion_tree = Tree("✅ SYSTEMATIC OPTIMIZATION COMPLETE")
            
            timing_node = completion_tree.add("⏱️ Execution Summary")
            timing_node.add(f"Total time: {total_time:.1f} seconds")
            timing_node.add(f"Final R²: {final_r2:.4f}")
            timing_node.add(f"Winning model: {winner_name}")
            timing_node.add(f"Preprocessing methods: {len(preprocessing_stack)}")
            
            if HAS_RICH:
                console.print(completion_tree)
            
            # Reset console title
            set_console_title("Systematic Optimization Complete")
            
            return final_results
            
        except Exception as e:
            # Create optimization failure tree
            failure_tree = Tree("❌ OPTIMIZATION FAILED")
            
            error_node = failure_tree.add("💥 Error Details")
            error_node.add(f"Phase: {getattr(self, 'current_phase', 'Unknown')}")
            error_node.add(f"Error: {str(e)[:100]}...")
            
            recovery_node = failure_tree.add("🔄 Recovery Information")
            recovery_node.add("Check logs for detailed traceback")
            recovery_node.add("Partial results may be available")
            
            if HAS_RICH:
                console.print(failure_tree)
            
            self.logger.error(f"Systematic optimization failed: {e}")
            set_console_title("❌ Optimization Failed")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main execution function"""
    
    try:
        # Load data based on dataset selection
        X, y = load_dataset(DATASET)
        print(f"\n{Colors.BOLD}📁 Loaded dataset {DATASET}{Colors.END}")
        
        # Initialize systematic optimizer
        optimizer = SystematicOptimizer(
            dataset_num=DATASET,
            max_preprocessing_trials=50,  # 50 trials per preprocessing method
            max_hyperopt_trials=500      # 500 trials for deep hyperparameter optimization
        )
        
        # Run systematic optimization
        final_results = optimizer.run_systematic_optimization(X, y)
        
        return final_results
        
    except Exception as e:
        set_console_title("❌ Error occurred")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main() 