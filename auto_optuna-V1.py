"""Legacy compatibility wrapper for tests.
Provides minimal implementations of transformers and optimizer expected by old tests.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import FunctionTransformer, PowerTransformer

from pathlib import Path as Path

from auto_optuna import (
    BattleTestedOptimizer as _BattleTestedOptimizer,
    IsolationForestTransformer as _IsolationForestTransformer,
    LocalOutlierFactorTransformer as _LocalOutlierFactorTransformer,
)

class OutlierFilterTransformer(BaseEstimator, TransformerMixin):
    """Simple KMeans-based outlier filter used in legacy tests."""

    def __init__(self, n_clusters=3, min_cluster_size_ratio=0.1, remove=True):
        self.n_clusters = n_clusters
        self.min_cluster_size_ratio = min_cluster_size_ratio
        self.remove = remove
        self.kmeans = None
        self.valid_clusters_ = None
        self.outlier_indices_ = None

    def fit(self, X, y=None):
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(X)
        counts = np.bincount(labels)
        min_size = int(len(X) * self.min_cluster_size_ratio)
        self.valid_clusters_ = np.where(counts >= min_size)[0]
        self.outlier_indices_ = np.flatnonzero(~np.isin(labels, self.valid_clusters_))
        return self

    def transform(self, X):
        labels = self.kmeans.predict(X)
        mask = np.isin(labels, self.valid_clusters_)
        return X[mask] if self.remove else X


class HSICFeatureSelector(BaseEstimator, TransformerMixin):
    """Select top-k features based on correlation with the target."""

    def __init__(self, k=10):
        self.k = k
        self.selected_features_ = None

    def fit(self, X, y):
        corr = np.abs(np.corrcoef(X, y, rowvar=False)[-1, :-1])
        self.selected_features_ = np.argsort(corr)[::-1][: self.k]
        return self

    def transform(self, X):
        return X[:, self.selected_features_]


class BattleTestedOptimizer(_BattleTestedOptimizer):
    """Subclass adding create_target_transformer used in legacy tests."""

    def create_target_transformer(self, trial):
        choice = trial.suggest_categorical("y_transform", ["none", "log1p", "power"])
        if choice == "log1p":
            return FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=False)
        if choice == "power":
            return PowerTransformer()
        return None


class IsolationForestTransformer(_IsolationForestTransformer):
    """Wrapper adding legacy parameters and attributes."""

    def __init__(self, contamination=0.1, n_estimators=100, random_state=42, remove=True):
        super().__init__(contamination=contamination, n_estimators=n_estimators, random_state=random_state)
        self.remove = remove
        self.outlier_indices_ = None

    def fit(self, X, y=None):
        super().fit(X, y)
        self.outlier_indices_ = np.flatnonzero(~self.mask_)
        return self

    def transform(self, X):
        labels = self.iforest.predict(X)
        mask = labels != -1
        return X[mask] if self.remove else X


class LocalOutlierFactorTransformer(_LocalOutlierFactorTransformer):
    """Wrapper adding legacy parameters and attributes."""

    def __init__(self, n_neighbors=20, contamination=0.1, remove=True):
        super().__init__(n_neighbors=n_neighbors, contamination=contamination)
        self.remove = remove
        self.outlier_indices_ = None

    def fit(self, X, y=None):
        super().fit(X, y)
        self.outlier_indices_ = np.flatnonzero(~self.mask_)
        return self

    def transform(self, X):
        return X[self.mask_] if self.remove else X

