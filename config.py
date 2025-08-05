"""
Configuration Module
===================
Central configuration for all auto_optuna components.

*** IMPORTANT: LOCAL DATA ONLY ***
This software ONLY USES LOCAL DATA IMPORTED WITHIN THE CODE FOLDER.
No external datasets, APIs, or remote data sources are supported.
All training data must be present as CSV files within the project directory.
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
    
    # Dataset configuration - LOCAL DATA ONLY
    "DATASET": {
        "TEST_SIZE": 0.2,
        "VALIDATION_SIZE": 0.15,  # Additional validation split
        "HOLDOUT_SIZE": 0.1,      # Final holdout for unbiased evaluation
        "RANDOM_STATE": 42
    },
    
    # Data split folders - organized splits for no data leakage
    "DATA_SPLITS": {
        "TRAIN_DIR": "data_splits/train/",
        "TEST_DIR": "data_splits/test/", 
        "VALIDATION_DIR": "data_splits/validation/",
        "HOLDOUT_DIR": "data_splits/holdout/"
    },

    # Optuna optimization parameters
    "OPTUNA": {
        "PREPROCESSING_TRIALS": 50,
        "HYPEROPT_TRIALS": 100,
        "MAX_TRIALS_BATTLE_TESTED": 40,
        "PRUNER_PREPROCESSING": MedianPruner(n_startup_trials=5),
        "PRUNER_HYPEROPT": MedianPruner(n_startup_trials=20),
        "SAMPLER": optuna.samplers.TPESampler(seed=42, warn_independent_sampling=False),
    },
    
    # Performance thresholds
    "THRESHOLDS": {
        "EXCELLENT": 0.01,      # ≤ 1% gap to ceiling ⇒ "excellent"
        "NEAR_CEILING": 0.02,   # ≤ 2% gap ⇒ "near ceiling"
        "TARGET_R2": 0.95       # Updated target R² for D3
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
        "EXTRA": {
            "n_estimators_range": (50, 300),
            "max_depth_range": (5, 20),
            "min_samples_split_range": (2, 10)
        },
        "SVR": {
            "C_range": (0.1, 100),
            "C_log": True,
            "gamma_range": (1e-4, 1),
            "gamma_log": True
        },
        "LASSO": {
            "alpha_range": (1e-4, 10),
            "alpha_log": True,
            "max_iter": 2000
        },
        "DT": {
            "max_depth_range": (3, 20),
            "min_samples_split_range": (2, 10)
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

# LOCAL DATA ONLY - Mapping of dataset IDs to local file paths
# This software ONLY USES LOCAL DATA within the project directory
DATASET_FILES = {
    1: {
        "name": "Engineering Temperature Prediction (Dataset 1)",
        "predictors": "Data/dataset1_X_predictors.csv",
        "targets": "Data/dataset1_y_target.csv",
    },
    2: {
        "name": "Engineering Temperature Prediction (Dataset 2)",
        "predictors": "Data/dataset2_X_predictors.csv",
        "targets": "Data/dataset2_y_target.csv",
    },
    3: {
        "name": "Rodger's Temperature Dataset (VNA Measurements)",
        "predictors_dir": "rodgers_data/raw_vna_data/",
        "targets": [
            "rodgers_data/temperature_data/temp_readings-1.csv",
            "rodgers_data/temperature_data/temp_readings-2.csv", 
            "rodgers_data/temperature_data/temp_readings-3.csv"
        ],
        "predictor_dirs": [
            "rodgers_data/raw_vna_data/Predictors-1/",
            "rodgers_data/raw_vna_data/Predictors-2/",
            "rodgers_data/raw_vna_data/Predictors-3/"
        ]
    },
    4: {
        "name": "D4 Temperature Dataset (VNA Measurements)",
        "predictors_dir": "VNA-D4/",
        "targets": "temp_readings-D4.csv",
        "predictor_dirs": ["VNA-D4/"]
    },
    5: {
        "name": "D4B Temperature Dataset (VNA Measurements - S11/Phase/Xs Only)",
        "predictors_dir": "VNA-D4B/",
        "targets": "temp_readings-D4.csv",
        "predictor_dirs": ["VNA-D4B/"]
    }
}

# Data quality validation settings
DATA_VALIDATION = {
    "MAX_OUTLIER_ZSCORE": 3.0,
    "MIN_SAMPLES_PER_FEATURE": 5,
    "MAX_MISSING_RATIO": 0.1,
    "REQUIRE_TARGET_PREDICTOR_MATCH": True
}
