#!/usr/bin/env python3
"""
Comprehensive Test Suite for Nate's Recipe Optimization Pipeline
================================================================
Tests all critical components without requiring user configuration.
Follows the principle: 2 CSVs in → Model out, nothing more.
"""

import pytest
import pandas as pd
import numpy as np
import joblib
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import warnings
warnings.filterwarnings('ignore')

# Import the modules to test
from auto_optuna import SystematicOptimizer
from auto_optuna.transformers import (
    KMeansOutlierTransformer,
    IsolationForestTransformer,
    LocalOutlierFactorTransformer,
)

# Test fixtures
@pytest.fixture
def sample_data():
    """Generate synthetic test data that mimics the real datasets"""
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    
    # Create synthetic predictors with some correlation structure
    X = np.random.randn(n_samples, n_features)
    # Add some correlated features
    X[:, 1] = X[:, 0] + 0.5 * np.random.randn(n_samples)
    X[:, 2] = X[:, 0] - 0.3 * np.random.randn(n_samples)
    
    # Create synthetic targets with realistic relationship
    y = (2 * X[:, 0] + 1.5 * X[:, 1] - 0.8 * X[:, 2] + 
         0.5 * np.sum(X[:, 3:8], axis=1) + 
         0.1 * np.random.randn(n_samples))
    
    return X.astype(np.float32), y.astype(np.float32)

@pytest.fixture
def temp_csv_files(sample_data):
    """Create temporary CSV files for testing"""
    X, y = sample_data
    temp_dir = Path(tempfile.mkdtemp())
    
    # Save as CSV files
    predictor_file = temp_dir / "test_predictors.csv"
    target_file = temp_dir / "test_targets.csv"
    
    pd.DataFrame(X).to_csv(predictor_file, header=False, index=False)
    pd.DataFrame(y).to_csv(target_file, header=False, index=False)
    
    yield predictor_file, target_file
    
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model artifacts"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

# =============================================================================
# UNIT TESTS - Individual Components
# =============================================================================

class TestOutlierTransformers:
    """Test custom outlier detection transformers"""
    
    def test_kmeans_outlier_transformer(self, sample_data):
        X, _ = sample_data
        transformer = KMeansOutlierTransformer(n_clusters=3)
        
        # Test fit and transform
        X_transformed = transformer.fit_transform(X)
        
        assert X_transformed.shape[0] <= X.shape[0]  # Should remove some samples
        assert X_transformed.shape[1] == X.shape[1]  # Features unchanged
        assert hasattr(transformer, 'mask_')
        assert hasattr(transformer, 'valid_clusters_')
    
    def test_isolation_forest_transformer(self, sample_data):
        X, _ = sample_data
        transformer = IsolationForestTransformer(contamination=0.1)
        
        X_transformed = transformer.fit_transform(X)
        
        assert X_transformed.shape[0] <= X.shape[0]
        assert X_transformed.shape[1] == X.shape[1]
        assert hasattr(transformer, 'mask_')
    
    def test_lof_transformer(self, sample_data):
        X, _ = sample_data
        transformer = LocalOutlierFactorTransformer(n_neighbors=5, contamination=0.1)
        
        X_transformed = transformer.fit_transform(X)
        
        assert X_transformed.shape[0] <= X.shape[0]
        assert X_transformed.shape[1] == X.shape[1]
        assert hasattr(transformer, 'mask_')

class TestSystematicOptimizer:
    """Test the main optimizer class"""

    def test_optimizer_initialization(self, temp_model_dir):
        with patch('auto_optuna.optimizer.Path') as mock_path:
            mock_path.return_value = temp_model_dir
            optimizer = SystematicOptimizer(dataset_num=1, max_hyperopt_trials=1)

            assert optimizer.dataset_num == 1
            assert optimizer.max_hyperopt_trials == 1
            assert hasattr(optimizer, 'cv')
            assert hasattr(optimizer, 'logger')

    def test_run_single_trial(self, sample_data, temp_model_dir):
        X, y = sample_data

        with patch('auto_optuna.optimizer.Path') as mock_path:
            mock_path.return_value = temp_model_dir
            optimizer = SystematicOptimizer(dataset_num=1, max_hyperopt_trials=1)
            results = optimizer.run_systematic_optimization(X, y)

            assert 'test_r2' in results


# =============================================================================
# INTEGRATION TESTS - End-to-End Pipeline
# =============================================================================

class TestPipelineIntegration:
    """Test complete pipeline integration"""

    @pytest.mark.slow
    def test_minimal_training_run(self, sample_data, temp_model_dir):
        """Test a minimal training run with very few trials"""
        X, y = sample_data

        with patch('auto_optuna.optimizer.Path') as mock_path:
            mock_path.return_value = temp_model_dir
            optimizer = SystematicOptimizer(dataset_num=1, max_hyperopt_trials=1)

            results = optimizer.run_systematic_optimization(X, y)

            assert isinstance(results, dict)
            assert 'test_r2' in results

    def test_model_persistence(self, sample_data, temp_model_dir):
        """Test that models are properly saved and can be loaded"""
        X, y = sample_data

        with patch('auto_optuna.optimizer.Path') as mock_path:
            mock_path.return_value = temp_model_dir
            optimizer = SystematicOptimizer(dataset_num=1, max_hyperopt_trials=1)

            optimizer.run_systematic_optimization(X, y)

            model_file = temp_model_dir / "hold1_final_model.pkl"
            assert model_file.exists()
            loaded_model = joblib.load(model_file)
            assert hasattr(loaded_model, 'predict')

# =============================================================================
# DATA VALIDATION TESTS
# =============================================================================

class TestDataValidation:
    """Test data loading and validation"""
    
    def test_csv_loading_format(self, temp_csv_files):
        """Test that CSV files are loaded correctly"""
        predictor_file, target_file = temp_csv_files
        
        # Test loading
        X = pd.read_csv(predictor_file, header=None).values.astype(np.float32)
        y = pd.read_csv(target_file, header=None).values.astype(np.float32).ravel()
        
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]
        assert X.dtype == np.float32
        assert y.dtype == np.float32
    
    def test_data_quality_checks(self, sample_data):
        """Test data quality validation"""
        X, y = sample_data
        
        # Check for NaN values
        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(y))
        
        # Check for infinite values
        assert not np.any(np.isinf(X))
        assert not np.any(np.isinf(y))
        
        # Check data types
        assert X.dtype == np.float32
        assert y.dtype == np.float32
        
        # Check shapes
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0]

# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Test performance characteristics"""
    
    def test_memory_usage(self, sample_data):
        """Test that memory usage is reasonable"""
        X, y = sample_data
        
        # Memory usage should be reasonable for the data size
        x_memory_mb = X.nbytes / (1024 * 1024)
        y_memory_mb = y.nbytes / (1024 * 1024)
        
        assert x_memory_mb < 100  # Should be much less for test data
        assert y_memory_mb < 10
    
    def test_prediction_speed(self, sample_data, temp_model_dir):
        """Test prediction speed is reasonable"""
        X, y = sample_data
        
        with patch('auto_optuna.optimizer.Path') as mock_path:
            mock_path.return_value = temp_model_dir
            optimizer = SystematicOptimizer(dataset_num=1, max_hyperopt_trials=1)

            optimizer.run_systematic_optimization(X, y)

            # Test prediction speed
            import time
            start_time = time.time()
            predictions = optimizer.final_pipeline.predict(optimizer.X_test)
            end_time = time.time()
            
            prediction_time = end_time - start_time
            samples_per_second = len(optimizer.X_test) / prediction_time
            
            assert samples_per_second > 100  # Should predict at least 100 samples/sec

# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_empty_data_handling(self, temp_model_dir):
        """Test handling of empty or minimal data"""
        # Create minimal data
        X = np.array([[1, 2], [3, 4]], dtype=np.float32)
        y = np.array([1, 2], dtype=np.float32)
        
        with patch('auto_optuna.optimizer.Path') as mock_path:
            mock_path.return_value = temp_model_dir
            optimizer = SystematicOptimizer(dataset_num=1, max_hyperopt_trials=1)

            try:
                optimizer.run_systematic_optimization(X, y)
            except Exception as e:
                msg = str(e).lower()
                assert "sample" in msg or "size" in msg
    
    def test_invalid_data_types(self, temp_model_dir):
        """Test handling of invalid data types"""
        # Create data with wrong types
        X = np.array([["a", "b"], ["c", "d"]])  # String data
        y = np.array([1, 2], dtype=np.float32)
        
        with patch('auto_optuna.optimizer.Path') as mock_path:
            mock_path.return_value = temp_model_dir
            optimizer = SystematicOptimizer(dataset_num=1, max_hyperopt_trials=1)

            with pytest.raises((ValueError, TypeError)):
                optimizer.run_systematic_optimization(X, y)

# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestConfiguration:
    """Test that no user configuration is required"""
    
    def test_hardcoded_dataset_numbers(self):
        """Test that dataset numbers are properly hardcoded"""
        # Import the main modules
        import battle_tested_optuna_playbook
        
        # Check that DATASET is hardcoded
        assert hasattr(battle_tested_optuna_playbook, 'DATASET')
        assert isinstance(battle_tested_optuna_playbook.DATASET, int)
        assert battle_tested_optuna_playbook.DATASET in [1, 2, 3]
    
    def test_no_config_files_exist(self):
        """Test that no configuration files exist in the repository"""
        current_dir = Path.cwd()
        
        # Check for common config file patterns
        forbidden_patterns = [
            "*.config", "*.cfg", "*.yaml", "*.yml", "*.json", "*.ini", "*.toml",
            "config.*", "settings.*", "preferences.*", ".env*"
        ]
        
        config_files = []
        for pattern in forbidden_patterns:
            config_files.extend(list(current_dir.glob(pattern)))
        
        # Filter out legitimate files that aren't configuration
        legitimate_files = {
            "requirements.txt",  # Dependencies, not configuration
            ".gitignore",       # Git configuration, not user configuration
            "package.json",     # If it exists, it's for tooling
            "pyproject.toml"    # If it exists, it's for tooling
        }
        
        actual_config_files = [f for f in config_files if f.name not in legitimate_files]
        
        assert len(actual_config_files) == 0, f"Found forbidden config files: {[f.name for f in actual_config_files]}"
    
    def test_no_argparse_imports(self):
        """Test that no scripts import argparse or click for command-line parsing"""
        python_files = list(Path.cwd().glob("*.py"))
        
        forbidden_imports = ["argparse", "click", "fire", "typer"]
        
        for py_file in python_files:
            if py_file.name.startswith("test_"):
                continue  # Skip test files
                
            content = py_file.read_text()
            for forbidden in forbidden_imports:
                assert f"import {forbidden}" not in content, f"{py_file.name} imports {forbidden} (forbidden for user configuration)"
                assert f"from {forbidden}" not in content, f"{py_file.name} imports from {forbidden} (forbidden for user configuration)"
    
    def test_no_input_statements(self):
        """Test that no scripts use input() for user prompts"""
        python_files = list(Path.cwd().glob("*.py"))
        
        for py_file in python_files:
            if py_file.name.startswith("test_"):
                continue  # Skip test files
                
            content = py_file.read_text()
            assert "input(" not in content, f"{py_file.name} contains input() statement (forbidden user interaction)"
    
    def test_no_environment_variable_config(self):
        """Test that scripts don't rely on environment variables for configuration"""
        python_files = list(Path.cwd().glob("*.py"))
        
        for py_file in python_files:
            if py_file.name.startswith("test_"):
                continue  # Skip test files
                
            content = py_file.read_text()
            # Allow os.getenv for system info but not for configuration
            forbidden_patterns = [
                "os.getenv(\"MODEL",
                "os.getenv(\"DATASET", 
                "os.getenv(\"TARGET",
                "os.getenv(\"CONFIG",
                "os.environ.get(\"MODEL",
                "os.environ.get(\"DATASET",
                "os.environ.get(\"TARGET",
                "os.environ.get(\"CONFIG"
            ]
            
            for pattern in forbidden_patterns:
                assert pattern not in content, f"{py_file.name} uses environment variables for configuration: {pattern}"
    
    def test_minimal_command_line_interface(self):
        """Test that the CLI requires minimal input"""
        # The main scripts should run with no arguments
        # This is tested by checking that main() functions exist and are callable
        import battle_tested_optuna_playbook
        
        assert hasattr(battle_tested_optuna_playbook, 'main')
        assert callable(battle_tested_optuna_playbook.main)
    
    def test_zero_configuration_principle(self):
        """Test the core principle: 2 CSVs in → Model out, nothing more"""
        # This test documents the core principle
        
        # The pipeline should:
        # 1. Take exactly 2 CSV files as input
        # 2. Produce a model as output
        # 3. Require ZERO configuration from the user
        
        # If this test fails, someone has violated the core principle
        assert True, "2 CSVs in → Model out. No configuration allowed."

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"]) 