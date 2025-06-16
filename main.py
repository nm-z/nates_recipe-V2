"""
Main Entry Point
===============
Main script for running ML optimization on different datasets.
"""

import argparse
import sys
from pathlib import Path

from config import CONFIG, Colors, DATASET_FILES
from utils import load_dataset, validate_dataset_files
from optimizer import SystematicOptimizer, BattleTestedOptimizer


def main():
    """Main entry point for auto_optuna package."""
    parser = argparse.ArgumentParser(description='Auto Optuna ML Optimization')
    parser.add_argument(
        '--dataset',
        type=int,
        default=1,
        choices=list(DATASET_FILES.keys()),
        help='Dataset ID to use'
    )
    parser.add_argument('--optimizer', type=str, default='systematic', 
                       choices=['systematic', 'battle_tested'],
                       help='Optimizer type to use')
    parser.add_argument('--trials', type=int, default=None,
                       help='Number of optimization trials')
    parser.add_argument('--target-r2', type=float, default=0.93,
                       help='Target R² for battle-tested optimizer')
    
    args = parser.parse_args()
    
    dataset_key = CONFIG["DATASET_ID_MAP"].get(args.dataset)
    if dataset_key is None:
        print(f"{Colors.RED}❌ Invalid dataset ID: {args.dataset}{Colors.END}")
        sys.exit(1)

    # Validate dataset availability / files
    if not validate_dataset_files(dataset_key):
        print(f"{Colors.RED}❌ Dataset '{dataset_key}' files not found!{Colors.END}")
        info = DATASET_FILES.get(args.dataset, {})
        if info.get('predictors') or info.get('targets'):
            print(
                f"Expected files: {info.get('predictors')} | {info.get('targets')}"
            )
        sys.exit(1)
    
    # Load dataset
    print(f"{Colors.BOLD}{Colors.CYAN}🚀 Starting Auto Optuna Optimization{Colors.END}")
    print(f"Dataset: {DATASET_FILES[args.dataset]['name']}")
    print(f"Optimizer: {args.optimizer}")

    X, y = load_dataset(dataset_key)
    
    # Initialize optimizer
    if args.optimizer == 'systematic':
        optimizer = SystematicOptimizer(
            dataset_num=args.dataset,
            max_hyperopt_trials=args.trials
        )
        results = optimizer.run_systematic_optimization(X, y)
    
    elif args.optimizer == 'battle_tested':
        optimizer = BattleTestedOptimizer(
            dataset_num=args.dataset,
            target_r2=args.target_r2,
            max_trials=args.trials or CONFIG["OPTUNA"]["MAX_TRIALS_BATTLE_TESTED"]
        )
        results = optimizer.run_optimization(X, y)
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}🎉 Optimization Complete!{Colors.END}")
    return results


def run_hold1():
    """Convenience function to run Hold 1 optimization."""
    # Override sys.argv to run Hold 1
    original_argv = sys.argv
    sys.argv = ['auto_optuna', '--dataset', '1', '--optimizer', 'systematic']
    
    try:
        return main()
    finally:
        sys.argv = original_argv


def run_hold2():
    """Convenience function to run Hold 2 optimization."""
    original_argv = sys.argv
    sys.argv = ['auto_optuna', '--dataset', '2', '--optimizer', 'systematic']
    
    try:
        return main()
    finally:
        sys.argv = original_argv


def run_hold3():
    """Convenience function to run Hold 3 optimization."""
    original_argv = sys.argv
    sys.argv = ['auto_optuna', '--dataset', '3', '--optimizer', 'systematic']
    
    try:
        return main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main() 