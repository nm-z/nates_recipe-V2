import sys
import ctypes
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    LIME = '\033[92;1m'
    END = '\033[0m'


def set_console_title(msg: str) -> None:
    """Set console window title if supported."""
    try:
        if sys.platform.startswith("win"):
            ctypes.windll.kernel32.SetConsoleTitleW(msg)
        else:
            sys.stdout.write(f"\33]0;{msg}\a")
            sys.stdout.flush()
    except Exception:
        pass


class OutlierFilterTransformer(BaseEstimator, TransformerMixin):
    """Remove outliers using K-means clustering."""

    def __init__(self, n_clusters: int = 3, min_cluster_size_ratio: float = 0.1, remove: bool = False):
        self.n_clusters = n_clusters
        self.min_cluster_size_ratio = min_cluster_size_ratio
        self.remove = remove
        self.kmeans = None
        self.valid_clusters = None
        self.outlier_indices_ = None

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
        except Exception:
            self.outlier_indices_ = np.array([])
            self.kmeans = None
        return self

    def transform(self, X, *, remove=None):
        if remove is None:
            remove = self.remove
        if remove and self.outlier_indices_ is not None and self.outlier_indices_.size > 0:
            keep_mask = np.ones(len(X), dtype=bool)
            keep_mask[self.outlier_indices_] = False
            return X[keep_mask]
        return X


class IsolationForestTransformer(BaseEstimator, TransformerMixin):
    """Outlier detection using Isolation Forest."""

    def __init__(self, contamination: float = 0.1, n_estimators: int = 100, random_state: int = 42, remove: bool = False):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.remove = remove
        self.iforest = None
        self.outlier_indices_ = None

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

    def transform(self, X, *, remove=None):
        if remove is None:
            remove = self.remove
        if remove and self.outlier_indices_ is not None and self.outlier_indices_.size > 0:
            keep_mask = np.ones(len(X), dtype=bool)
            keep_mask[self.outlier_indices_] = False
            return X[keep_mask]
        return X


class LocalOutlierFactorTransformer(BaseEstimator, TransformerMixin):
    """Outlier detection using Local Outlier Factor."""

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.1, remove: bool = False):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.remove = remove
        self.lof = None
        self.outlier_indices_ = None

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

    def transform(self, X, *, remove=None):
        if remove is None:
            remove = self.remove
        if remove and self.outlier_indices_ is not None and self.outlier_indices_.size > 0:
            keep_mask = np.ones(len(X), dtype=bool)
            keep_mask[self.outlier_indices_] = False
            return X[keep_mask]
        return X


class HSICFeatureSelector(BaseEstimator, TransformerMixin):
    """Simple HSIC-based feature selector (approximation)."""

    def __init__(self, k: int = 50):
        self.k = k
        self.selected_features_ = None

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


def load_dataset(dataset_num: int):
    """Load dataset based on numeric choice."""
    if dataset_num == 1:
        X = pd.read_csv('Predictors_Hold-1_2025-04-14_18-28.csv', header=None).values.astype(np.float32)
        y = pd.read_csv('9_10_24_Hold_01_targets.csv', header=None).values.astype(np.float32).ravel()
    elif dataset_num == 2:
        X = pd.read_csv('hold2_predictor.csv', header=None).values.astype(np.float32)
        y = pd.read_csv('hold2_target.csv', header=None).values.astype(np.float32).ravel()
    elif dataset_num == 3:
        X = pd.read_csv('predictors_Hold 1 Full_20250527_151252.csv', header=None).values.astype(np.float32)
        y = pd.read_csv('targets_Hold 1 Full_20250527_151252.csv', header=None).values.astype(np.float32).ravel()
    else:
        raise ValueError(f"Invalid DATASET value: {dataset_num}. Must be 1, 2, or 3.")
    return X, y


