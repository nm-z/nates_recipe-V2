#!/usr/bin/env python3
"""
Smoke Test - Quick validation that the pipeline is working
==========================================================
Run this before committing changes to catch obvious issues.
"""

import sys
import traceback
import numpy as np
import pandas as pd
from pathlib import Path

def test_imports():
    """Test that all critical imports work"""
    print("🔍 Testing imports...")
    
    try:
        import battle_tested_optuna_playbook
        print("  ✅ battle_tested_optuna_playbook")
        
        from battle_tested_optuna_playbook import BattleTestedOptimizer
        print("  ✅ BattleTestedOptimizer")
        
        import optuna
        print("  ✅ optuna")
        
        import sklearn
        print("  ✅ sklearn")
        
        from rich.tree import Tree
        from rich.console import Console
        print("  ✅ rich")
        
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality with synthetic data"""
    print("\n🔍 Testing basic functionality...")
    
    try:
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(50, 20).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)
        
        # Test outlier transformers
        from battle_tested_optuna_playbook import KMeansOutlierTransformer
        transformer = KMeansOutlierTransformer(n_clusters=2)
        X_transformed = transformer.fit_transform(X)
        print(f"  ✅ KMeansOutlierTransformer: {X.shape} → {X_transformed.shape}")
        
        # Test optimizer initialization
        from unittest.mock import patch
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch('battle_tested_optuna_playbook.Path') as mock_path:
                mock_path.return_value = Path(temp_dir)
                
                from battle_tested_optuna_playbook import BattleTestedOptimizer
                optimizer = BattleTestedOptimizer(dataset_num=1, max_trials=1)
                print("  ✅ BattleTestedOptimizer initialization")
        
        return True
    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading if CSV files exist"""
    print("\n🔍 Testing data loading...")
    
    csv_files = list(Path('.').glob('*.csv'))
    if not csv_files:
        print("  ℹ️  No CSV files found (expected in test environment)")
        return True
    
    try:
        for csv_file in csv_files[:2]:  # Test first 2 files
            df = pd.read_csv(csv_file, header=None, nrows=5)
            print(f"  ✅ {csv_file.name}: {df.shape} (sample)")
        return True
    except Exception as e:
        print(f"  ❌ Data loading failed: {e}")
        return False

def test_configuration_principle():
    """Test that no user configuration is required"""
    print("\n🔍 Testing configuration principle...")
    
    try:
        import battle_tested_optuna_playbook
        
        # Check hardcoded dataset
        if hasattr(battle_tested_optuna_playbook, 'DATASET'):
            dataset_num = battle_tested_optuna_playbook.DATASET
            if isinstance(dataset_num, int) and dataset_num in [1, 2, 3]:
                print(f"  ✅ Hardcoded dataset number: {dataset_num}")
            else:
                print(f"  ❌ Invalid dataset number: {dataset_num}")
                return False
        else:
            print("  ❌ No hardcoded DATASET found")
            return False
        
        # Check no config files required
        config_files = (list(Path('.').glob('*.config')) + 
                       list(Path('.').glob('*.cfg')) + 
                       list(Path('.').glob('config.*')))
        
        if config_files:
            print(f"  ⚠️  Found config files: {[f.name for f in config_files]}")
        else:
            print("  ✅ No config files found")
        
        return True
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        return False

def test_memory_usage():
    """Test basic memory usage"""
    print("\n🔍 Testing memory usage...")
    
    try:
        import psutil
        
        # Check available memory
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        print(f"  ✅ Available memory: {available_gb:.1f} GB")
        
        if available_gb < 1.0:
            print("  ⚠️  Low memory available")
        
        # Test basic array operations
        X = np.random.randn(1000, 100).astype(np.float32)
        y = np.random.randn(1000).astype(np.float32)
        memory_mb = (X.nbytes + y.nbytes) / (1024**2)
        print(f"  ✅ Test arrays created: {memory_mb:.1f} MB")
        
        return True
    except Exception as e:
        print(f"  ❌ Memory test failed: {e}")
        return False

def main():
    """Run all smoke tests"""
    print("🚀 Running Smoke Tests for Nate's Recipe Pipeline")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_data_loading,
        test_configuration_principle,
        test_memory_usage
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All smoke tests passed! Pipeline is ready.")
        return 0
    else:
        print("💥 Some smoke tests failed. Check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 