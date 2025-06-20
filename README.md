# Nate's Recipe Optimization - V2

A modular machine learning pipeline for recipe optimization using Optuna-based hyperparameter tuning.

## Project Overview

This repository exposes reusable components to streamline model training. The key pieces are:

- **SystematicOptimizer** – orchestrates the optimization workflow
- **Transformers** – custom preprocessing and outlier removal utilities
- **Config** – central configuration for datasets and search ranges
- **Utils** – helpers for loading data and saving artifacts
- **Main script** – command line entry point

## File Layout

- `config.py` – dataset definitions and global settings
- `optimizer.py` – core optimization logic
- `transformers.py` – preprocessing transformers
- `utils.py` – data loading and helper functions
- `main.py` – CLI for running optimizations
- `tests/` – basic pytest suite

## Key Features

- **Optuna Integration** for hyperparameter search
- **RepeatedKFold** cross‑validation (5 splits × 3 repeats)
- **Rich tree logging** with optional `rich` console output
- **Automatic artifact saving** of models and preprocessing steps

## Usage

Run an optimization on any supported dataset by ID:

```bash
python main.py --dataset 1 --optimizer systematic
```

Use `--optimizer battle_tested` to run the compatibility wrapper around `SystematicOptimizer`.

## Requirements

See `requirements.txt` for the full list of Python packages. Python 3.8 or newer is recommended.

## Installation

```bash
git clone <repository-url>
cd nates_recipe-V2
python -m venv ml_env
source ml_env/bin/activate  # On Windows use ml_env\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Datasets and search ranges are defined in `config.py`. Adjust the `CONFIG` dictionary to change defaults.

## Contributing

- Preserve the structured logging style
- Keep optimizers and transformers modular
- Submit tests when adding new functionality
