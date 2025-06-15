"""
Utilities Module
===============
Common utility functions for data loading, logging, and file operations.
"""

import pandas as pd
import numpy as np
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from .config import CONFIG, DATASET_FILES, Colors

try:  # Optional Rich integration
    from rich.tree import Tree
    from rich.console import Console

    console = Console()
    HAS_RICH = True
except Exception:  # pragma: no cover - Rich not installed
    Tree = None
    console = None
    HAS_RICH = False

def load_dataset(dataset_id: int):
    """
    Load dataset based on ID.
    
    Args:
        dataset_id: Dataset identifier (1, 2, or 3)
        
    Returns:
        tuple: (X, y) arrays
    """
    if dataset_id not in DATASET_FILES:
        raise ValueError(f"Invalid dataset ID: {dataset_id}. Must be 1, 2, or 3")
    
    files = DATASET_FILES[dataset_id]
    
    try:
        X = pd.read_csv(files["predictors"], header=None).values.astype(np.float32)
        y = pd.read_csv(files["targets"], header=None).values.astype(np.float32).ravel()

        if HAS_RICH:
            tree = Tree(f"✅ Loaded {files['name']} dataset")
            tree.add(f"Predictors: {X.shape}")
            tree.add(f"Targets: {len(y)} samples")
            console.print(tree)
        else:
            print(
                f"{Colors.GREEN}✅ Loaded {files['name']} dataset: {X.shape} features, {len(y)} samples{Colors.END}"
            )
        return X, y
        
    except FileNotFoundError as e:
        print(f"{Colors.RED}❌ Dataset files not found: {e}{Colors.END}")
        print(f"Expected files: {files['predictors']}, {files['targets']}")
        raise


def setup_logging(dataset_num: int, model_dir: Path = None):
    """
    Setup logging configuration.
    
    Args:
        dataset_num: Dataset number for log file naming
        model_dir: Directory for log files (optional)
        
    Returns:
        logger: Configured logger instance
    """
    if model_dir is None:
        model_dir = Path(CONFIG["PATHS"]["MODEL_DIR_TEMPLATE"].format(dataset_num=dataset_num))
    
    model_dir.mkdir(exist_ok=True)
    
    log_file = model_dir / CONFIG["PATHS"]["LOG_FILE_TEMPLATE"].format(dataset_num=dataset_num)
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized for dataset {dataset_num}")
    return logger


def save_model_artifacts(model, preprocessing_components, dataset_num: int, 
                        results: dict = None, model_dir: Path = None):
    """
    Save model and related artifacts.
    
    Args:
        model: Trained model
        preprocessing_components: Dict of preprocessing components
        dataset_num: Dataset number
        results: Results dictionary (optional)
        model_dir: Directory to save to (optional)
    """
    if model_dir is None:
        model_dir = Path(CONFIG["PATHS"]["MODEL_DIR_TEMPLATE"].format(dataset_num=dataset_num))
    
    model_dir.mkdir(exist_ok=True)
    
    # Save model
    model_file = model_dir / CONFIG["PATHS"]["MODEL_FILE_TEMPLATE"].format(dataset_num=dataset_num)
    joblib.dump(model, model_file)

    # Save preprocessing components
    for name, component in preprocessing_components.items():
        component_file = model_dir / f"hold{dataset_num}_{name}.pkl"
        joblib.dump(component, component_file)

    # Save results if provided
    if results:
        results_file = model_dir / CONFIG["PATHS"]["RESULTS_FILE_TEMPLATE"].format(dataset_num=dataset_num)
        with open(results_file, 'w') as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")

    if HAS_RICH:
        tree = Tree("💾 Model artifacts saved")
        tree.add(str(model_file))
        for name in preprocessing_components:
            tree.add(f"{name} saved")
        if results:
            tree.add(str(results_file))
        console.print(tree)
    else:
        print(f"{Colors.GREEN}💾 Model artifacts saved to {model_dir}/{Colors.END}")


def load_model_artifacts(dataset_num: int, model_dir: Path = None):
    """
    Load saved model and preprocessing components.
    
    Args:
        dataset_num: Dataset number
        model_dir: Directory to load from (optional)
        
    Returns:
        dict: Dictionary containing model and preprocessing components
    """
    if model_dir is None:
        model_dir = Path(CONFIG["PATHS"]["MODEL_DIR_TEMPLATE"].format(dataset_num=dataset_num))
    
    artifacts = {}
    
    # Load model
    model_file = model_dir / CONFIG["PATHS"]["MODEL_FILE_TEMPLATE"].format(dataset_num=dataset_num)
    if model_file.exists():
        artifacts['model'] = joblib.load(model_file)
    
    # Load preprocessing components
    for pkl_file in model_dir.glob(f"hold{dataset_num}_*.pkl"):
        if pkl_file.name != f"hold{dataset_num}_final_model.pkl":
            component_name = pkl_file.stem.replace(f"hold{dataset_num}_", "")
            artifacts[component_name] = joblib.load(pkl_file)
    
    return artifacts


def create_diagnostic_plots(y_true, y_pred, study=None, dataset_num: int = 1, 
                           noise_ceiling: float = None, baseline_r2: float = None,
                           model_dir: Path = None):
    """
    Create comprehensive diagnostic plots.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        study: Optuna study object (optional)
        dataset_num: Dataset number
        noise_ceiling: Noise ceiling estimate (optional)
        baseline_r2: Baseline R² score (optional)
        model_dir: Directory to save plots (optional)
    """
    if model_dir is None:
        model_dir = Path(CONFIG["PATHS"]["MODEL_DIR_TEMPLATE"].format(dataset_num=dataset_num))
    
    model_dir.mkdir(exist_ok=True)
    
    from sklearn.metrics import r2_score
    
    # Create comprehensive plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Hold-{dataset_num} Model: Final Evaluation', fontsize=16, fontweight='bold')
    
    # 1. Predicted vs Actual
    ax1.scatter(y_true, y_pred, alpha=0.6, color='blue')
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title('Predicted vs Actual (Test Set)')
    ax1.grid(True, alpha=0.3)
    
    r2_test = r2_score(y_true, y_pred)
    ax1.text(0.05, 0.95, f'R² = {r2_test:.4f}', transform=ax1.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 2. Residuals
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, color='green')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)
    
    # 3. Optimization progress (if study provided)
    if study and len(study.trials) > 0:
        trial_numbers = [t.number for t in study.trials if t.state.name == 'COMPLETE']
        trial_values = [t.value for t in study.trials if t.state.name == 'COMPLETE']
        
        if trial_numbers:
            ax3.plot(trial_numbers, trial_values, 'b-', alpha=0.6)
            if noise_ceiling:
                ax3.axhline(y=noise_ceiling, color='orange', linestyle=':', 
                           label=f'Noise Ceiling = {noise_ceiling:.3f}')
            if baseline_r2:
                ax3.axhline(y=baseline_r2, color='red', linestyle='--',
                           label=f'Baseline = {baseline_r2:.3f}')
            ax3.set_xlabel('Trial Number')
            ax3.set_ylabel('CV R² Score')
            ax3.set_title('Optimization Progress')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    # 4. Distribution comparison
    ax4.hist(y_true, alpha=0.5, label='Actual', bins=20)
    ax4.hist(y_pred, alpha=0.5, label='Predicted', bins=20)
    ax4.set_xlabel('Values')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = model_dir / f"hold{dataset_num}_diagnostic_plots.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    if HAS_RICH:
        console.print(f"📊 Diagnostic plots saved to: {plot_file}")
    else:
        print(f"{Colors.GREEN}📊 Diagnostic plots saved to: {plot_file}{Colors.END}")


def print_results_summary(results: dict, dataset_num: int):
    """
    Print a formatted summary of results.
    
    Args:
        results: Results dictionary
        dataset_num: Dataset number
    """
    if HAS_RICH:
        summary = Tree(f"📋 FINAL SUMMARY - Hold {dataset_num}")
        if 'test_r2' in results:
            summary.add(f"🧪 Test set R²: {results['test_r2']:.4f}")
        if 'cv_best_r2' in results:
            summary.add(f"🏆 Best CV R²: {results['cv_best_r2']:.4f}")
        if 'noise_ceiling' in results:
            summary.add(f"📏 Noise ceiling: {results['noise_ceiling']:.4f}")
        if 'test_mae' in results:
            summary.add(f"📐 Test MAE: {results['test_mae']:.4f}")
        if 'test_rmse' in results:
            summary.add(f"📊 Test RMSE: {results['test_rmse']:.4f}")

        if 'test_r2' in results and 'noise_ceiling' in results:
            gap = abs(results['noise_ceiling'] - results['test_r2'])
            if gap <= CONFIG["THRESHOLDS"]["EXCELLENT"]:
                summary.add("🎉 EXCELLENT - within 1% of ceiling!")
            elif gap <= CONFIG["THRESHOLDS"]["NEAR_CEILING"]:
                summary.add("✅ Near ceiling - within 2%")
            else:
                summary.add("📈 Room for improvement")
        console.print(summary)
    else:
        print(f"\n{Colors.BOLD}{Colors.CYAN}📋 FINAL SUMMARY - Hold {dataset_num}{Colors.END}")
        print("=" * 50)

        if 'test_r2' in results:
            print(f"🧪 Test set R²: {results['test_r2']:.4f}")
        if 'cv_best_r2' in results:
            print(f"🏆 Best CV R²: {results['cv_best_r2']:.4f}")
        if 'noise_ceiling' in results:
            print(f"📏 Noise ceiling: {results['noise_ceiling']:.4f}")
        if 'test_mae' in results:
            print(f"📐 Test MAE: {results['test_mae']:.4f}")
        if 'test_rmse' in results:
            print(f"📊 Test RMSE: {results['test_rmse']:.4f}")

        if 'test_r2' in results and 'noise_ceiling' in results:
            gap = abs(results['noise_ceiling'] - results['test_r2'])
            if gap <= CONFIG["THRESHOLDS"]["EXCELLENT"]:
                print(f"🎉 {Colors.GREEN}EXCELLENT - within 1% of ceiling!{Colors.END}")
            elif gap <= CONFIG["THRESHOLDS"]["NEAR_CEILING"]:
                print(f"✅ {Colors.YELLOW}Near ceiling - within 2%{Colors.END}")
            else:
                print(f"📈 {Colors.BLUE}Room for improvement{Colors.END}")


def validate_dataset_files(dataset_id: int):
    """
    Validate that dataset files exist.
    
    Args:
        dataset_id: Dataset identifier
        
    Returns:
        bool: True if files exist, False otherwise
    """
    if dataset_id not in DATASET_FILES:
        return False
    
    files = DATASET_FILES[dataset_id]
    predictor_exists = Path(files["predictors"]).exists()
    target_exists = Path(files["targets"]).exists()
    
    return predictor_exists and target_exists 