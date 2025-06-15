"""Shared transformers from early auto_optuna versions."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import FunctionTransformer, PowerTransformer


class OutlierFilterTransformer(BaseEstimator, TransformerMixin):
    """Remove small clusters detected via K-means."""

    def __init__(self, n_clusters: int = 3, min_cluster_size_ratio: float = 0.1, remove: bool = False) -> None:
        self.n_clusters = n_clusters
        self.min_cluster_size_ratio = min_cluster_size_ratio
        self.remove = remove
        self.kmeans: KMeans | None = None
        self.valid_clusters: np.ndarray | None = None
        self.outlier_indices_: np.ndarray | None = None

    def fit(self, X, y=None):
        del y
        try:
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            cluster_labels = self.kmeans.fit_predict(X)
            cluster_counts = np.bincount(cluster_labels)
            min_size = int(len(X) * self.min_cluster_size_ratio)
            self.valid_clusters = np.where(cluster_counts >= min_size)[0]
            valid_mask = np.isin(cluster_labels, self.valid_clusters)
            self.outlier_indices_ = np.where(~valid_mask)[0]
            max_removal_ratio = 0.15 if len(X) <= 200 else 0.2
            if len(self.outlier_indices_) > len(X) * max_removal_ratio:
                self.outlier_indices_ = np.array([])
        except Exception:
            self.outlier_indices_ = np.array([])
            self.kmeans = None
        return self

    def transform(self, X, *, remove: bool | None = None):
        if remove is None:
            remove = self.remove
        if remove and self.outlier_indices_ is not None and self.outlier_indices_.size > 0:
            keep_mask = np.ones(len(X), dtype=bool)
            keep_mask[self.outlier_indices_] = False
            return X[keep_mask]
        return X


class IsolationForestTransformer(BaseEstimator, TransformerMixin):
    """Outlier detection using Isolation Forest."""

    def __init__(self, contamination: float = 0.1, n_estimators: int = 100, random_state: int = 42, remove: bool = False) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.remove = remove
        self.iforest: IsolationForest | None = None
        self.outlier_indices_: np.ndarray | None = None

    def fit(self, X, y=None):
        del y
        try:
            self.iforest = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.iforest.fit(X)
            outlier_labels = self.iforest.predict(X)
            self.outlier_indices_ = np.where(outlier_labels == -1)[0]
        except Exception:
            self.iforest = None
            self.outlier_indices_ = np.array([])
        return self

    def transform(self, X, *, remove: bool | None = None):
        if remove is None:
            remove = self.remove
        if remove and self.outlier_indices_ is not None and self.outlier_indices_.size > 0:
            keep_mask = np.ones(len(X), dtype=bool)
            keep_mask[self.outlier_indices_] = False
            return X[keep_mask]
        return X


class LocalOutlierFactorTransformer(BaseEstimator, TransformerMixin):
    """Outlier detection using Local Outlier Factor."""

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.1, remove: bool = False) -> None:
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.remove = remove
        self.lof: LocalOutlierFactor | None = None
        self.outlier_indices_: np.ndarray | None = None

    def fit(self, X, y=None):
        del y
        try:
            self.lof = LocalOutlierFactor(
                n_neighbors=min(self.n_neighbors, len(X) - 1),
                contamination=self.contamination,
                n_jobs=-1,
            )
            outlier_labels = self.lof.fit_predict(X)
            self.outlier_indices_ = np.where(outlier_labels == -1)[0]
        except Exception:
            self.lof = None
            self.outlier_indices_ = np.array([])
        return self

    def transform(self, X, *, remove: bool | None = None):
        if remove is None:
            remove = self.remove
        if remove and self.outlier_indices_ is not None and self.outlier_indices_.size > 0:
            keep_mask = np.ones(len(X), dtype=bool)
            keep_mask[self.outlier_indices_] = False
            return X[keep_mask]
        return X


class HSICFeatureSelector(BaseEstimator, TransformerMixin):
    """Simplified HSIC-based feature selector."""

    def __init__(self, k: int = 50) -> None:
        self.k = k
        self.selected_features_: np.ndarray | None = None

    def fit(self, X, y):
        try:
            correlations = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
            correlations = np.nan_to_num(correlations)
            self.selected_features_ = np.argsort(correlations)[-self.k:]
        except Exception:
            self.selected_features_ = np.arange(min(self.k, X.shape[1]))
        return self

    def transform(self, X):
        if self.selected_features_ is None:
            return X
        return X[:, self.selected_features_]


class BattleTestedOptimizer:
    """Legacy helper exposing only create_target_transformer used in tests."""

    def create_target_transformer(self, trial):
        y_transform = trial.suggest_categorical("y_transform", ["none", "log1p", "power"])
        if y_transform == "none":
            return None
        if y_transform == "log1p":
            return FunctionTransformer(func=np.log1p, inverse_func=np.expm1, validate=False)
        if y_transform == "power":
            return PowerTransformer(method="yeo-johnson", standardize=False)
        return None

