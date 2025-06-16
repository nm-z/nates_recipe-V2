"""
Configuration Module
===================
Central configuration for all auto_optuna components.
"""

import optuna
from optuna.pruners import MedianPruner

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

# Main configuration dictionary
CONFIG = {
    # Cross-validation settings
    "CV_SPLITS": 5,
    "CV_REPEATS": 3,
    
    # Dataset configuration
    "DATASET": {
        "DEFAULT": "diabetes",  # Default to load_diabetes
        "TEST_SIZE": 0.2,
        "RANDOM_STATE": 42
    },
    
    # Scikit-learn Built-in Datasets
    "SKLEARN_DATASETS": {
        "iris": {"loader": "load_iris", "type": "classification", "name": "Iris"},
        "diabetes": {"loader": "load_diabetes", "type": "regression", "name": "Diabetes"},
        "california_housing": {"loader": "fetch_california_housing", "type": "regression", "name": "California Housing"},
        "wine": {"loader": "load_wine", "type": "classification", "name": "Wine"},
        "breast_cancer": {"loader": "load_breast_cancer", "type": "classification", "name": "Breast Cancer Wisconsin"},
        "diabetes_frame": {"loader": "load_diabetes", "type": "regression", "name": "Diabetes (as_frame=True)"},
        "synthetic_regression": {"loader": "make_regression", "type": "regression", "name": "Synthetic Regression"},
        "airfoil_self_noise": {"loader": "fetch_openml", "type": "regression", "name": "Airfoil Self Noise"},
        "friedman1_small": {"loader": "make_friedman1", "type": "regression", "name": "Friedman1 Small"},
        "friedman2": {"loader": "make_friedman2", "type": "regression", "name": "Friedman2"},
        "friedman3": {"loader": "make_friedman3", "type": "regression", "name": "Friedman3"},
        "california_housing_frame": {"loader": "fetch_california_housing", "type": "regression", "name": "California Housing (as_frame=True)"},
        "energy_efficiency": {"loader": "fetch_openml", "type": "regression", "name": "Energy Efficiency"},
        "openml_42092": {"loader": "fetch_openml", "type": "regression", "name": "OpenML Dataset 42092"},
    },

    # Mapping from integer IDs to dataset keys within SKLEARN_DATASETS
    "DATASET_ID_MAP": {
        1: "diabetes",
        2: "california_housing",
        3: "diabetes",  # Defaulting to diabetes for hold-3
    },
    
    # Optuna optimization parameters
    "OPTUNA": {
        "PREPROCESSING_TRIALS": 50,
        "HYPEROPT_TRIALS": 100,
        "MAX_TRIALS_BATTLE_TESTED": 40,
        "PRUNER_PREPROCESSING": MedianPruner(n_startup_trials=5),
        "PRUNER_HYPEROPT": MedianPruner(n_startup_trials=20),
        "SAMPLER": optuna.samplers.TPESampler(seed=42, multivariate=True),
    },
    
    # Performance thresholds
    "THRESHOLDS": {
        "EXCELLENT": 0.01,      # ≤ 1% gap to ceiling ⇒ "excellent"
        "NEAR_CEILING": 0.02,   # ≤ 2% gap ⇒ "near ceiling"
        "TARGET_R2": 0.93       # Default target R²
    },
    
    # Console and logging settings
    "CONSOLE": {
        "LARGE_DATASET_SIZE": 10_000,
        "REFRESH_INTERVAL": 10.0,  # seconds
    },
    
    # Preprocessing settings
    "PREPROCESSING": {
        "VARIANCE_THRESHOLD": 1e-8,
        "QUANTILE_RANGE": (5, 95),  # For RobustScaler
        "OUTLIER_CONTAMINATION": 0.1,
        "N_NEIGHBORS_LOF": 20,
        "KMEANS_CLUSTERS": 3,
        "MIN_CLUSTER_SIZE_RATIO": 0.1
    },
    
    # Model hyperparameter ranges
    "MODEL_PARAMS": {
        "RIDGE": {
            "alpha_range": (1e-3, 100),
            "alpha_log": True
        },
        "ELASTIC": {
            "alpha_range": (1e-3, 10),
            "alpha_log": True,
            "l1_ratio_range": (0.1, 0.9),
            "max_iter": 2000
        },
        "GBR": {
            "n_estimators_range": (50, 300),
            "learning_rate_range": (0.01, 0.3),
            "max_depth_range": (3, 10)
        },
        "RF": {
            "n_estimators_range": (50, 300),
            "max_depth_range": (5, 20),
            "min_samples_split_range": (2, 10)
        },
        "SVR": {
            "C_range": (0.1, 100),
            "C_log": True,
            "gamma_range": (1e-4, 1),
            "gamma_log": True
        }
    },
    
    # File paths and naming
    "PATHS": {
        "MODEL_DIR_TEMPLATE": "best_model_hold{dataset_num}",
        "LOG_FILE_TEMPLATE": "hold{dataset_num}_training_log.txt",
        "MODEL_FILE_TEMPLATE": "hold{dataset_num}_final_model.pkl",
        "RESULTS_FILE_TEMPLATE": "hold{dataset_num}_results.txt"
    }
}

# Mapping of dataset IDs to friendly names and optional file paths. The
# loaders are handled via the SKLEARN_DATASETS configuration above. File
# paths are left as ``None`` for built-in datasets but the structure allows
# external CSV files to be specified if needed.
DATASET_FILES = {
    1: {
        "name": CONFIG["SKLEARN_DATASETS"]["diabetes"]["name"],
        "dataset": "diabetes",
        "predictors": None,
        "targets": None,
    },
    2: {
        "name": CONFIG["SKLEARN_DATASETS"]["california_housing"]["name"],
        "dataset": "california_housing",
        "predictors": None,
        "targets": None,
    },
    3: {
        "name": CONFIG["SKLEARN_DATASETS"]["diabetes"]["name"],
        "dataset": "diabetes",
        "predictors": None,
        "targets": None,
    },
}
