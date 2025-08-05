"""
Main Entry Point
===============
Main script for running ML optimization on different datasets.

*** IMPORTANT: LOCAL DATA ONLY ***
This software ONLY USES LOCAL DATA IMPORTED WITHIN THE CODE FOLDER.
No external datasets, APIs, or remote data sources are supported.
"""

import argparse
import sys
import warnings
import signal
from pathlib import Path

# Suppress optuna warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="optuna")
warnings.filterwarnings("ignore", message=".*experimental.*")

try:
    from .config import CONFIG, Colors, DATASET_FILES
except ImportError:
    from config import CONFIG, Colors, DATASET_FILES

try:
    from .utils import load_dataset, create_data_splits
except ImportError:
    from utils import load_dataset, create_data_splits

try:
    from .optimizer import SystematicOptimizer, BattleTestedOptimizer, NeuralNetworkOptimizer
except ImportError:
    from optimizer import SystematicOptimizer, BattleTestedOptimizer, NeuralNetworkOptimizer

# Global variable to store current optimizer for signal handling
current_optimizer = None


def _signal_handler(signum, frame):
    """Handle Ctrl+C by training the best model found so far."""
    global current_optimizer
    
    print(f"\n{Colors.YELLOW}🛑 Ctrl+C detected! Starting best model training...{Colors.END}")
    
    if current_optimizer is None:
        print(f"{Colors.RED}❌ No optimizer instance available. Exiting gracefully...{Colors.END}")
        sys.exit(0)
    
    # Check if the optimizer has found any parameters yet
    if hasattr(current_optimizer, 'study') and current_optimizer.study is not None and len(current_optimizer.study.trials) > 0:
        try:
            print(f"{Colors.BLUE}🔄 Found {len(current_optimizer.study.trials)} completed trials{Colors.END}")
            print(f"{Colors.BLUE}🏆 Best R² so far: {current_optimizer.study.best_value:.4f}{Colors.END}")
            print(f"{Colors.CYAN}🚀 Training final model with best parameters...{Colors.END}")
            
            # Train final model with best parameters found so far
            if hasattr(current_optimizer, '_train_best_model_from_study'):
                results = current_optimizer._train_best_model_from_study()
            elif hasattr(current_optimizer, 'phase_3_final_evaluation'):
                # For SystematicOptimizer
                current_optimizer.final_pipeline = current_optimizer._build_final_model(current_optimizer.study.best_params)
                
                # Handle outlier detection if needed
                X_train_final = current_optimizer.X_train.copy()
                y_train_final = current_optimizer.y_train.copy()
                best_params = current_optimizer.study.best_params
                
                if best_params.get('use_outlier_detection', False) and best_params.get('outlier_method'):
                    try:
                        from .transformers import KMeansOutlierTransformer, IsolationForestTransformer, LocalOutlierFactorTransformer
                    except ImportError:
                        from transformers import KMeansOutlierTransformer, IsolationForestTransformer, LocalOutlierFactorTransformer
                    
                    outlier_method = best_params['outlier_method']
                    if outlier_method == 'isolation':
                        outlier_detector = IsolationForestTransformer()
                    elif outlier_method == 'lof':
                        outlier_detector = LocalOutlierFactorTransformer()
                    else:
                        outlier_detector = KMeansOutlierTransformer()
                    
                    outlier_detector.fit(X_train_final)
                    outlier_mask = outlier_detector.mask_
                    X_train_final = X_train_final[outlier_mask]
                    y_train_final = y_train_final[outlier_mask]
                    print(f"Outlier detection removed {(~outlier_mask).sum()} samples")
                
                # Train final model
                print(f"Training final model on {X_train_final.shape[0]} samples...")
                current_optimizer.final_pipeline.fit(X_train_final, y_train_final)
                
                # Evaluate final model
                results = current_optimizer.phase_3_final_evaluation()
            elif hasattr(current_optimizer, '_phase_3_final_evaluation'):
                # For NeuralNetworkOptimizer
                results = current_optimizer._phase_3_final_evaluation()
            else:
                print(f"{Colors.RED}❌ Unknown optimizer type. Cannot train best model.{Colors.END}")
                sys.exit(1)
            
            print(f"\n{Colors.BOLD}{Colors.GREEN}🎉 Best Model Training Complete!{Colors.END}")
            print(f"Final Test R²: {results.get('test_r2', 'N/A')}")
            
            sys.exit(0)
            
        except Exception as e:
            print(f"{Colors.RED}❌ Error training best model: {e}{Colors.END}")
            print(f"{Colors.BLUE}Falling back to graceful exit...{Colors.END}")
            sys.exit(1)
    else:
        print(f"{Colors.YELLOW}⚠️  No optimization trials completed yet. Exiting gracefully...{Colors.END}")
        sys.exit(0)


def _validate_csv_datasets(dataset_info):
    """Validate standard CSV datasets."""
    predictors_path = dataset_info.get("predictors")
    targets_path = dataset_info.get("targets")
    return (Path(predictors_path).exists() and Path(targets_path).exists())

def _validate_vna_dataset(dataset_info, file_pattern):
    """Validate VNA dataset with given file pattern."""
    import glob
    predictor_dirs = dataset_info.get("predictor_dirs", [])
    target_files = dataset_info.get("targets", [])
    
    # Check target files exist
    if isinstance(target_files, list):
        for target_file in target_files:
            if not Path(target_file).exists():
                return False
    else:
        if not Path(target_files).exists():
            return False
    
    # Check at least one predictor directory exists and has files
    for pred_dir in predictor_dirs:
        if Path(pred_dir).exists():
            vna_files = glob.glob(str(Path(pred_dir) / file_pattern))
            if len(vna_files) > 0:
                return True
    
    return False

def validate_dataset_files(dataset_id):
    """Validate that local dataset files exist."""
    if dataset_id not in DATASET_FILES:
        return False
    
    dataset_info = DATASET_FILES[dataset_id]
    
    # Check different dataset types
    if dataset_id in [1, 2]:
        return _validate_csv_datasets(dataset_info)
    
    if dataset_id == 3:
        return _validate_vna_dataset(dataset_info, "VNA_*.csv")
    
    if dataset_id == 4:
        return _validate_vna_dataset(dataset_info, "VNA-D4*.csv")
    
    if dataset_id == 5:
        return _validate_vna_dataset(dataset_info, "VNA-D4*.csv")
    
    return False


def _parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Auto Optuna ML Optimization - LOCAL DATA ONLY')
    parser.add_argument(
        '--dataset',
        type=int,
        default=3,  # Default to D3 (Rodger's Temperature Dataset)
        choices=list(DATASET_FILES.keys()),
        help='Dataset ID to use (LOCAL DATA ONLY)'
    )
    parser.add_argument('--optimizer', type=str, default='systematic', 
                       choices=['systematic', 'battle_tested'],
                       help='Optimizer type to use')
    parser.add_argument('--NN', action='store_true',
                       help='Use Neural Network models only for training')
    parser.add_argument('--trials', type=int, default=None,
                       help='Number of optimization trials (if not specified, trains until noise ceiling is reached)')
    parser.add_argument('--target-r2', type=float, default=0.95,
                       help='Target R² for optimization (default: 0.95)')
    parser.add_argument('--create-splits', action='store_true',
                       help='Create organized data splits before training')
    parser.add_argument('--kbest', type=str, default=None,
        help='Feature selection ratio for SelectKBest as NUM:DEN (e.g. 90:100 for top 90%). If not set, disables SelectKBest.')
    
    return parser.parse_args()

def _validate_and_load_dataset(args):
    """Validate dataset and load it."""
    # Check if dataset is available
    if args.dataset not in DATASET_FILES:
        print(f"{Colors.RED}❌ Invalid dataset ID: {args.dataset}{Colors.END}")
        print(f"Available datasets: {list(DATASET_FILES.keys())}")
        sys.exit(1)

    # Validate local dataset files exist
    if not validate_dataset_files(args.dataset):
        print(f"{Colors.RED}❌ Local dataset files not found!{Colors.END}")
        dataset_info = DATASET_FILES[args.dataset]
        print(f"Dataset: {dataset_info['name']}")
        if args.dataset in [1, 2]:
            print(f"Expected files: {dataset_info.get('predictors')} | {dataset_info.get('targets')}")
        if args.dataset == 3:
            print(f"Expected directories: {dataset_info.get('predictor_dirs')}")
            print(f"Expected target files: {dataset_info.get('targets')}")
        sys.exit(1)
    
    # Load dataset
    print(f"{Colors.BOLD}{Colors.CYAN}🚀 Starting Auto Optuna Optimization - LOCAL DATA ONLY{Colors.END}")
    print(f"Dataset: {DATASET_FILES[args.dataset]['name']}")
    
    # Display correct optimizer type
    if args.NN:
        print(f"Optimizer: neural_network")
    else:
        print(f"Optimizer: {args.optimizer}")
    
    print(f"Target R²: {args.target_r2}")

    try:
        X, y = load_dataset(args.dataset)
        print(f"{Colors.GREEN}✅ Successfully loaded dataset: X{X.shape}, y({len(y)},){Colors.END}")
        return X, y
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"{Colors.RED}❌ Error loading dataset: {e}{Colors.END}")
        sys.exit(1)

def main():
    """Main entry point for auto_optuna package - LOCAL DATA ONLY."""
    args = _parse_arguments()
    
    # Set up complete CLI output logging BEFORE any other output
    try:
        from .utils import setup_logging
    except ImportError:
        from utils import setup_logging
    setup_logging(args.dataset)
    
    X, y = _validate_and_load_dataset(args)

    # Create data splits if requested
    if args.create_splits:
        splits = create_data_splits(X, y, args.dataset)
        print(f"{Colors.BOLD}{Colors.GREEN}✅ Data splits created successfully{Colors.END}")
        
        # Use training split for optimization
        X, y = splits['train']
        print(f"{Colors.BLUE}🔄 Using training split for optimization: X{X.shape}, y({len(y)},){Colors.END}")
    
    # Initialize optimizer
    results = None
    if args.NN:
        # Neural Network only mode
        optimizer = NeuralNetworkOptimizer(
            dataset_num=args.dataset,
            max_hyperopt_trials=args.trials,
            kbest_ratio=args.kbest
        )
        results = optimizer.run_neural_network_optimization(X, y)
    elif args.optimizer == 'systematic':
        optimizer = SystematicOptimizer(
            dataset_num=args.dataset,
            max_hyperopt_trials=args.trials
        )
        results = optimizer.run_systematic_optimization(X, y)
    
    elif args.optimizer == 'battle_tested':
        optimizer = BattleTestedOptimizer(
            dataset_num=args.dataset,
            target_r2=args.target_r2,
            max_trials=args.trials  # Pass None to enable auto-stopping at noise ceiling
        )
        results = optimizer.run_optimization(X, y)
    
    if results is None:
        print(f"{Colors.RED}❌ Unknown optimizer: {args.optimizer}{Colors.END}")
        sys.exit(1)
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}🎉 Optimization Complete!{Colors.END}")
    print(f"Final Test R²: {results.get('test_r2', 'N/A')}")
    
    # Check if target achieved
    if 'test_r2' in results and results['test_r2'] >= args.target_r2:
        print(f"{Colors.BOLD}{Colors.GREEN}🎯 TARGET ACHIEVED: R² = {results['test_r2']:.4f} >= {args.target_r2}{Colors.END}")
    else:
        print(f"{Colors.YELLOW}📊 Current R²: {results.get('test_r2', 'N/A'):.4f}, Target: {args.target_r2}{Colors.END}")
    
    return results


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}🛑 Process interrupted by user (Ctrl+C){Colors.END}")
        print(f"{Colors.BLUE}Cleaning up and exiting gracefully...{Colors.END}")
        sys.exit(0)
    except (SystemExit, RuntimeError, ValueError) as e:
        print(f"\n{Colors.RED}❌ Unexpected error: {e}{Colors.END}")
        sys.exit(1) 