"""Compatibility wrapper for legacy tests."""

import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from sklearn.cluster import KMeans

from auto_optuna import (
    BattleTestedOptimizer,
    KMeansOutlierTransformer,
    IsolationForestTransformer,
    LocalOutlierFactorTransformer,
    SystematicOptimizer,
)

# Backwards compatibility name
OutlierFilterTransformer = KMeansOutlierTransformer

# Expose for import compatibility
__all__ = [
    "BattleTestedOptimizer",
    "KMeansOutlierTransformer",
    "IsolationForestTransformer",
    "LocalOutlierFactorTransformer",
    "SystematicOptimizer",
    "OutlierFilterTransformer",
    "HSICFeatureSelector",
]

from sklearn.feature_selection import SelectKBest, mutual_info_regression

class HSICFeatureSelector(SelectKBest):
    """Simplified HSIC-based feature selector for testing."""

    def __init__(self, k=10):
        super().__init__(score_func=mutual_info_regression, k=k)

    def fit(self, X, y):  # noqa: D401 - simple passthrough
        import numpy as np

        corrs = [abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])]
        self.scores_ = np.array(corrs)
        self.selected_features_ = np.argsort(self.scores_)[-self.k :][::-1]
        return self

    def transform(self, X):
        return X[:, self.selected_features_]


class OutlierFilterTransformer:
    """Simple wrapper to remove small KMeans clusters."""

    def __init__(self, n_clusters=3, min_cluster_size_ratio=0.1, remove=True):
        self.n_clusters = n_clusters
        self.min_cluster_size_ratio = min_cluster_size_ratio
        self.remove = remove
        self.kmeans = None
        self.outlier_indices_ = None

    def fit(self, X, y=None):
        del y
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(X)
        counts = np.bincount(labels)
        min_size = int(len(X) * self.min_cluster_size_ratio)
        valid = np.where(counts >= min_size)[0]
        self.outlier_indices_ = np.where(~np.isin(labels, valid))[0]
        return self

    def transform(self, X):
        labels = self.kmeans.predict(X)
        mask = np.isin(labels, self.kmeans.labels_[self.outlier_indices_])
        if self.remove:
            return X[~mask]
        return X
