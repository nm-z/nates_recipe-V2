#!/usr/bin/env python3
"""
Interactive test script to verify saved models work correctly
Provides comprehensive performance analysis and diagnostics
"""

import pandas as pd
import numpy as np
import joblib
import time
import sys
import platform
from pathlib import Path
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    explained_variance_score, median_absolute_error
)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def get_system_info():
    """Get system information for reproducibility"""
    import sklearn
    import scipy
    return {
        'python_version': sys.version.split()[0],
        'sklearn_version': sklearn.__version__,
        'scipy_version': scipy.__version__,
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'architecture': platform.architecture()[0]
    }

def load_test_data(dataset_choice):
    """Load the appropriate test data based on dataset choice"""
    if dataset_choice == "1":
        # Hold-1: Load full data and use saved test indices to prevent data leakage
        X_full = pd.read_csv('Predictors_Hold-1_2025-04-14_18-28.csv', header=None).values.astype(np.float32)
        y_full = pd.read_csv('9_10_24_Hold_01_targets.csv', header=None).values.astype(np.float32).ravel()
        
        # Load the exact test indices used during training
        test_indices_path = 'best_model_hold1/hold1_test_indices.npy'
        train_indices_path = 'best_model_hold1/hold1_train_indices.npy'
        
        if Path(test_indices_path).exists() and Path(train_indices_path).exists():
            test_idx = np.load(test_indices_path)
            train_idx = np.load(train_indices_path)
            
            X_test = X_full[test_idx]
            y_test = y_full[test_idx]
            X_train = X_full[train_idx]
            y_train = y_full[train_idx]
            
            print(f"✅ Using saved test indices: {len(test_idx)} samples")
            print(f"   Test indices: {test_idx[:3] if len(test_idx) >= 3 else test_idx}...{test_idx[-3:] if len(test_idx) >= 3 else test_idx}")
            print(f"   Train samples: {len(train_idx)}")
        else:
            print("❌ No saved test indices found! Using fallback method (may have data leakage)")
            print("   Please retrain the model to generate proper test indices")
            # Fallback to old method with warning
            X_test = X_full[-81:]
            y_test = y_full[-81:]
            X_train = X_full[:-81]
            y_train = y_full[:-81]
        
        return X_train, X_test, y_train, y_test, "Hold-1"
    else:
        # Hold-2: Load full data and use saved test indices
        X_full = pd.read_csv('hold2_predictor.csv', header=None).values.astype(np.float32)
        y_full = pd.read_csv('hold2_target.csv', header=None).values.astype(np.float32).ravel()
        
        # Load the exact test indices used during training
        test_indices_path = 'best_model_hold2/hold2_test_indices.npy'
        train_indices_path = 'best_model_hold2/hold2_train_indices.npy'
        
        if Path(test_indices_path).exists() and Path(train_indices_path).exists():
            test_idx = np.load(test_indices_path)
            train_idx = np.load(train_indices_path)
            
            X_test = X_full[test_idx]
            y_test = y_full[test_idx]
            X_train = X_full[train_idx]
            y_train = y_full[train_idx]
            
            print(f"✅ Using saved test indices: {len(test_idx)} samples")
            print(f"   Test indices: {test_idx[:3] if len(test_idx) >= 3 else test_idx}...{test_idx[-3:] if len(test_idx) >= 3 else test_idx}")
            print(f"   Train samples: {len(train_idx)}")
        else:
            print("❌ No saved test indices found! Using fallback method (may have data leakage)")
            print("   Please retrain the model to generate proper test indices")
            # Fallback to old method with warning
            X_test = X_full[-22:]
            y_test = y_full[-22:]
            X_train = X_full[:-22]
            y_train = y_full[:-22]
        
        return X_train, X_test, y_train, y_test, "Hold-2"

def calculate_adjusted_r2(r2, n_samples, n_features):
    """Calculate adjusted R²"""
    if n_samples <= n_features + 1:
        return np.nan
    return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    mask = y_true != 0
    if not np.any(mask):
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100)

def calculate_smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    if not np.any(mask):
        return 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask] * 100)

def analyze_residuals(y_true, y_pred):
    """Analyze residuals for diagnostic purposes"""
    residuals = y_true - y_pred
    
    # Normality test (Shapiro-Wilk)
    if len(residuals) >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan
    
    # Homoscedasticity check (Breusch-Pagan test approximation)
    # Correlation between squared residuals and predicted values
    bp_corr = np.corrcoef(residuals**2, y_pred)[0, 1] if len(residuals) > 1 else np.nan
    
    return {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'skewness': stats.skew(residuals),
        'kurtosis': stats.kurtosis(residuals),
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'bp_correlation': bp_corr
    }

def measure_inference_time(model, X_test, n_runs=100):
    """Measure inference time and throughput"""
    # Warm up
    _ = model.predict(X_test[:1])
    
    # Time multiple runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict(X_test)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times)
    per_sample_time = avg_time / len(X_test) * 1000  # ms per sample
    throughput = len(X_test) / avg_time  # samples per second
    
    return {
        'total_time_ms': avg_time * 1000,
        'per_sample_ms': per_sample_time,
        'throughput_samples_per_sec': throughput
    }

def get_model_size(model_path):
    """Get model file size"""
    return Path(model_path).stat().st_size / (1024 * 1024)  # MB

def print_performance_table(results, dataset_name):
    """Print comprehensive performance table"""
    
    print(f"\n{'='*80}")
    print(f"🔬 COMPREHENSIVE MODEL PERFORMANCE REPORT - {dataset_name}")
    print(f"{'='*80}")
    
    # 1. Data & Split Summary
    print(f"\n📊 1. DATA & SPLIT SUMMARY")
    print(f"{'─'*50}")
    print(f"{'Total samples:':<35} {results['total_samples']:>10}")
    print(f"{'Features before preprocessing:':<35} {results['features_before']:>10}")
    print(f"{'Features after preprocessing:':<35} {results['features_after']:>10}")
    print(f"{'Features removed:':<35} {results['features_removed']:>10}")
    print(f"{'Training samples:':<35} {results['train_samples']:>10}")
    print(f"{'Test samples (20% holdout):':<35} {results['test_samples']:>10}")
    
    # 2. Model & Training Details
    print(f"\n⚙️  2. MODEL & TRAINING DETAILS")
    print(f"{'─'*50}")
    print(f"{'Algorithm:':<35} {results['algorithm']:>15}")
    print(f"{'Scikit-learn version:':<35} {results['sklearn_version']:>15}")
    print(f"{'Best hyperparameters:':<35}")
    for param, value in results['hyperparameters'].items():
        print(f"{'  ' + param + ':':<33} {str(value):>17}")
    print(f"{'Cross-validation:':<35} {'5-fold CV':>15}")
    print(f"{'Training time:':<35} {results['training_time']:>15}")
    print(f"{'Random seed:':<35} {'42':>15}")
    
    # 3. Primary Performance Metrics
    print(f"\n🎯 3. PRIMARY PERFORMANCE METRICS")
    print(f"{'─'*50}")
    print(f"{'Metric':<25} {'Test Set':<15} {'Baseline':<15}")
    print(f"{'─'*25} {'─'*15} {'─'*15}")
    print(f"{'R² Score:':<25} {results['r2']:<15.4f} {results['baseline_r2']:<15.4f}")
    print(f"{'Adjusted R²:':<25} {results['adj_r2']:<15.4f} {'N/A':<15}")
    print(f"{'Explained Variance:':<25} {results['explained_var']:<15.4f} {'N/A':<15}")
    print(f"{'MAE:':<25} {results['mae']:<15.6f} {'N/A':<15}")
    print(f"{'MSE:':<25} {results['mse']:<15.6f} {'N/A':<15}")
    print(f"{'RMSE:':<25} {results['rmse']:<15.6f} {'N/A':<15}")
    print(f"{'MAPE (%):':<25} {results['mape']:<15.2f} {'N/A':<15}")
    print(f"{'SMAPE (%):':<25} {results['smape']:<15.2f} {'N/A':<15}")
    print(f"{'Median Abs Error:':<25} {results['median_ae']:<15.6f} {'N/A':<15}")
    
    # 4. Baseline & Noise Ceiling
    print(f"\n📏 4. BASELINE & NOISE CEILING")
    print(f"{'─'*50}")
    print(f"{'Baseline R² (Ridge):':<35} {results['baseline_r2']:>15.4f}")
    print(f"{'Noise ceiling estimate:':<35} {results['noise_ceiling']:>15.4f}")
    print(f"{'Beats baseline:':<35} {results['beats_baseline']:>15}")
    print(f"{'Near ceiling:':<35} {results['near_ceiling']:>15}")
    
    # 5. Error & Diagnostic Analyses
    print(f"\n🔍 5. ERROR & DIAGNOSTIC ANALYSES")
    print(f"{'─'*50}")
    residuals = results['residuals']
    print(f"{'Residual mean (≈0?):':<35} {residuals['mean']:>15.6f}")
    print(f"{'Residual std:':<35} {residuals['std']:>15.6f}")
    print(f"{'Residual skewness:':<35} {residuals['skewness']:>15.4f}")
    print(f"{'Residual kurtosis:':<35} {residuals['kurtosis']:>15.4f}")
    print(f"{'Shapiro-Wilk p-value:':<35} {residuals['shapiro_p']:>15.4f}")
    normality = "✅ Normal" if residuals['shapiro_p'] > 0.05 else "❌ Non-normal"
    print(f"{'Residual normality:':<35} {normality:>15}")
    
    # 6. Robustness Checks
    print(f"\n🛡️  6. ROBUSTNESS CHECKS")
    print(f"{'─'*50}")
    print(f"{'CV R² mean:':<35} {results['cv_mean']:>15.4f}")
    print(f"{'CV R² std:':<35} {results['cv_std']:>15.4f}")
    print(f"{'CV stability:':<35} {results['cv_stability']:>15}")
    
    # 7. Computational & Deployment Metrics
    print(f"\n⚡ 7. COMPUTATIONAL & DEPLOYMENT METRICS")
    print(f"{'─'*50}")
    perf = results['performance']
    print(f"{'Inference time (total):':<35} {perf['total_time_ms']:>12.2f} ms")
    print(f"{'Per-sample latency:':<35} {perf['per_sample_ms']:>12.4f} ms")
    print(f"{'Throughput:':<35} {perf['throughput_samples_per_sec']:>12.0f} samples/sec")
    print(f"{'Model size:':<35} {results['model_size_mb']:>12.2f} MB")
    
    # 8. Reproducibility
    print(f"\n🔄 8. REPRODUCIBILITY")
    print(f"{'─'*50}")
    sys_info = results['system_info']
    print(f"{'Python version:':<35} {sys_info['python_version']:>15}")
    print(f"{'NumPy version:':<35} {sys_info['numpy_version']:>15}")
    print(f"{'Pandas version:':<35} {sys_info['pandas_version']:>15}")
    print(f"{'Platform:':<35} {sys_info['platform'][:15]:>15}")
    print(f"{'Architecture:':<35} {sys_info['architecture']:>15}")
    
    # Summary Assessment
    print(f"\n🏆 FINAL ASSESSMENT")
    print(f"{'─'*50}")
    
    if results['r2'] >= 0.95:
        assessment = "🌟 EXCELLENT"
    elif results['r2'] >= 0.90:
        assessment = "🎯 VERY GOOD"
    elif results['r2'] >= 0.80:
        assessment = "✅ GOOD"
    elif results['r2'] >= 0.70:
        assessment = "⚠️  ACCEPTABLE"
    else:
        assessment = "❌ POOR"
    
    print(f"{'Overall Performance:':<35} {assessment:>15}")
    print(f"{'R² Achievement:':<35} {results['r2']:>15.4f}")
    
    if results['beats_baseline'] and results['near_ceiling']:
        final_verdict = "🏆 MODEL ACCEPTED - At theoretical maximum!"
    elif results['beats_baseline']:
        final_verdict = "✅ MODEL ACCEPTED - Beats baseline"
    else:
        final_verdict = "❌ MODEL REJECTED - Below baseline"
    
    print(f"{'Final Verdict:':<35} {final_verdict}")
    print(f"{'='*80}")

def test_saved_model():
    """Interactive model testing with comprehensive analysis"""
    
    print("🔬 Interactive Model Performance Testing")
    print("=" * 50)
    print("\nAvailable models:")
    print("1. Hold-1 Model (401 samples, 81 test samples)")
    print("2. Hold-2 Model (108 samples, 22 test samples)")
    
    while True:
        choice = input("\nWhich model would you like to test? (1 or 2): ").strip()
        if choice in ["1", "2"]:
            break
        print("Please enter 1 or 2")
    
    # Load data
    print(f"\n📁 Loading test data for {'Hold-1' if choice == '1' else 'Hold-2'}...")
    X_train, X_test, y_train, y_test, dataset_name = load_test_data(choice)
    
    # Load models
    print("📦 Loading saved models...")
    if choice == "1":
        model_dir = "best_model_hold1"
        model_file = "hold1_best_model.pkl"
    else:
        model_dir = "best_model_hold2"
        model_file = "hold2_best_model.pkl"
    
    # Try new naming convention first, fall back to old if needed
    preprocessing_path = f'{model_dir}/hold{choice}_preprocessing_pipeline.pkl'
    if not Path(preprocessing_path).exists():
        preprocessing_path = f'{model_dir}/preprocessing_pipeline.pkl'
    preprocessing_pipeline = joblib.load(preprocessing_path)
    best_model = joblib.load(f'{model_dir}/{model_file}')
    
    # Load evaluation results - try new naming convention first, fall back to old
    eval_results_path = f'{model_dir}/hold{choice}_evaluation_results.txt'
    if not Path(eval_results_path).exists():
        eval_results_path = f'{model_dir}/evaluation_results.txt'
    
    with open(eval_results_path, 'r') as f:
        eval_lines = f.readlines()
    
    eval_results = {}
    for line in eval_lines:
        if ':' in line:
            key, value = line.strip().split(': ', 1)
            try:
                eval_results[key] = float(value)
            except:
                eval_results[key] = value
    
    print("✅ Models loaded successfully")
    
    # Apply preprocessing
    print("🧹 Applying preprocessing...")
    X_train_clean = preprocessing_pipeline.transform(X_train)
    X_test_clean = preprocessing_pipeline.transform(X_test)
    
    # Make predictions
    print("🔮 Making predictions...")
    start_time = time.time()
    y_pred = best_model.predict(X_test_clean)
    prediction_time = time.time() - start_time
    
    # Calculate all metrics
    print("📊 Calculating comprehensive metrics...")
    
    # Basic metrics
    r2 = r2_score(y_test, y_pred)
    adj_r2 = calculate_adjusted_r2(r2, len(y_test), X_test_clean.shape[1])
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    explained_var = explained_variance_score(y_test, y_pred)
    median_ae = median_absolute_error(y_test, y_pred)
    mape = calculate_mape(y_test, y_pred)
    smape = calculate_smape(y_test, y_pred)
    
    # Cross-validation on training data
    cv_scores = cross_val_score(best_model, X_train_clean, y_train, cv=5, scoring='r2')
    
    # Residual analysis
    residuals = analyze_residuals(y_test, y_pred)
    
    # Performance metrics
    performance = measure_inference_time(best_model, X_test_clean)
    
    # Model size
    model_size_mb = get_model_size(f'{model_dir}/{model_file}')
    
    # System info
    system_info = get_system_info()
    
    # Prepare results dictionary
    results = {
        'total_samples': len(X_train) + len(X_test),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features_before': X_train.shape[1],
        'features_after': X_train_clean.shape[1],
        'features_removed': X_train.shape[1] - X_train_clean.shape[1],
        'algorithm': eval_results['best_model_type'].upper(),
        'sklearn_version': system_info['sklearn_version'],
        'hyperparameters': eval(eval_results['best_params']) if isinstance(eval_results['best_params'], str) else eval_results['best_params'],
        'training_time': 'From log',
        'r2': r2,
        'adj_r2': adj_r2,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'explained_var': explained_var,
        'median_ae': median_ae,
        'mape': mape,
        'smape': smape,
        'baseline_r2': eval_results['baseline_r2'],
        'noise_ceiling': eval_results['noise_ceiling'],
        'beats_baseline': "✅ YES" if eval_results['beats_baseline'] else "❌ NO",
        'near_ceiling': "✅ YES" if eval_results['near_ceiling'] else "❌ NO",
        'residuals': residuals,
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores),
        'cv_stability': "✅ Stable" if np.std(cv_scores) < 0.1 else "⚠️ Variable",
        'performance': performance,
        'model_size_mb': model_size_mb,
        'system_info': system_info
    }
    
    # Print comprehensive table
    print_performance_table(results, dataset_name)
    
    # Show sample predictions
    print(f"\n🧪 SAMPLE PREDICTIONS (First 5 test samples)")
    print(f"{'─'*60}")
    print(f"{'Sample':<8} {'Actual':<12} {'Predicted':<12} {'Error':<12} {'Rel Error%':<12}")
    print(f"{'─'*8} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
    
    for i in range(min(5, len(y_test))):
        error = abs(y_test[i] - y_pred[i])
        rel_error = (error / abs(y_test[i])) * 100 if y_test[i] != 0 else np.inf
        print(f"{i+1:<8} {y_test[i]:<12.6f} {y_pred[i]:<12.6f} {error:<12.6f} {rel_error:<12.2f}")
    
    return results

if __name__ == "__main__":
    test_saved_model() 