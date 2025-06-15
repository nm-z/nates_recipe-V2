import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from auto_optuna import (
    load_dataset,
    setup_logging,
    SystematicOptimizer,
)
from auto_optuna.transformers import KMeansOutlierCV


def test_load_dataset_shapes():
    X, y = load_dataset(1)
    assert X.shape[0] == len(y)
    assert X.dtype == np.float32
    assert y.dtype == np.float32


def test_setup_logging_creates_file(tmp_path):
    logger = setup_logging(1, model_dir=tmp_path)
    log_file = tmp_path / "hold1_training_log.txt"
    logger.info("test")
    assert log_file.exists()


def test_kmeans_outlier_cv_removes_outliers():
    rs = np.random.RandomState(0)
    inliers = rs.normal(0, 1, (50, 2))
    outliers = rs.normal(5, 1, (5, 2))
    X = np.vstack([inliers, outliers])
    t = KMeansOutlierCV(n_clusters_range=(2, 4), cv_folds=2, min_cluster_size_ratio=0.1)
    t.fit(X)
    X_clean = t.transform(X)
    assert t.best_n_clusters_ >= 2
    assert X_clean.shape[0] <= X.shape[0]


def test_systematic_optimizer_runs(tmp_path):
    X, y = load_dataset(2)
    with patch('auto_optuna.optimizer.Path') as mock_path:
        mock_path.return_value = Path(tmp_path)
        opt = SystematicOptimizer(dataset_num=2, max_hyperopt_trials=1)
        results = opt.run_systematic_optimization(X, y)
    assert 'test_r2' in results
