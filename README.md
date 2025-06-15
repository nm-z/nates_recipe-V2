# Nate's Recipe Optimization - V2

A machine learning pipeline for recipe optimization using advanced hyperparameter tuning with Optuna.

## Project Overview

This project implements a sophisticated machine learning pipeline for recipe optimization, featuring:

- **Advanced Hyperparameter Optimization**: Multiple versions of Optuna-based optimization scripts
- **Cross-Validation**: Robust model validation with configurable threading
- **Rich Logging**: Structured tree-based logging for clear pipeline visualization
- **Model Persistence**: Automated saving and loading of best-performing models
- **Multiple Hold-Out Sets**: Support for multiple validation datasets

## Files Description

### Core Scripts
- `auto_optuna-V1.py` - Initial Optuna optimization implementation
- `auto_optuna-V1.1.py` - Enhanced version with improved logging
- `auto_optuna-V1.2.py` - Advanced version with Rich tree logging
- `auto_optuna-V1.3.py` - Latest version with optimized performance
- `battle_tested_optuna_playbook.py` - Production-ready optimization pipeline

### Data Files
- `9_10_24_Hold_01_targets.csv` - Target values for hold-out set 1
- `hold2_predictor.csv` - Predictor variables for hold-out set 2
- `hold2_target.csv` - Target values for hold-out set 2
- `Predictors_Hold-1_2025-04-14_18-28.csv` - Predictor variables for hold-out set 1

### Model Artifacts
- `best_model_hold1/` - Best models for hold-out set 1
- `best_model_hold2/` - Best models for hold-out set 2
- `best_model_hold3/` - Best models for hold-out set 3

### Testing & Results
- `test_best_model.py` - Model testing and evaluation script
- `results.md` - Detailed results and performance metrics
- `training_output.log` - Training logs and debugging information

## Key Features

### Hyperparameter Optimization
- **Optuna Integration**: Advanced Bayesian optimization for hyperparameter tuning
- **Multi-Phase Optimization**: Progressive refinement through multiple optimization phases
- **Cross-Validation**: 12-thread cross-validation with single-thread models to prevent system crashes
- **No Fixed R² Targets**: Dynamic performance optimization without predetermined targets

### Logging & Monitoring
- **Rich Tree Logging**: Hierarchical, structured logging throughout the entire pipeline
- **Live Progress Updates**: Real-time progress tracking with tqdm integration
- **Comprehensive Error Handling**: Detailed exception capture and reporting
- **Artifact Preservation**: All models, logs, and metrics are automatically saved

### Performance Optimization
- **Parallel Processing**: Optimized threading configuration for maximum performance
- **Memory Management**: Efficient handling of large datasets
- **Reproducible Runs**: Consistent results across multiple executions
- **Fail-Fast Design**: Quick error detection and reporting

## Usage

### Basic Optimization
```bash
python auto_optuna-V1.3.py
```

### Production Pipeline
```bash
python battle_tested_optuna_playbook.py
```

### Model Testing
```bash
python test_best_model.py
```

## Requirements

- Python 3.8+
- Optuna
- scikit-learn
- pandas
- numpy
- rich (for structured logging)
- tqdm (for progress bars)
- joblib (for model persistence)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd nates_recipe-V2
```

2. Create and activate virtual environment:
```bash
python -m venv ml_env
source ml_env/bin/activate  # On Windows: ml_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

The pipeline follows these key principles:
- **12-thread cross-validation** with single-thread individual models
- **Rich tree logging** as the primary logging method
- **No fixed R² targets** - performance optimization approaches noise ceiling naturally
- **Incremental file editing** for maintainability
- **Comprehensive artifact saving** for reproducibility

## Results

Detailed results and performance metrics are available in `results.md`.

## Contributing

When contributing to this project:
1. Maintain the Rich tree logging structure
2. Use 12 threads for cross-validation, 1 thread for individual models
3. Follow incremental editing practices
4. Preserve all existing functionality unless explicitly requested to change
5. Save all artifacts and maintain comprehensive logging

