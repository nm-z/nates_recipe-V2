"""
Custom Transformers Module
==========================
Custom preprocessing transformers for outlier removal and data cleaning.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics.pairwise import pairwise_kernels

class KMeansOutlierTransformer(BaseEstimator, TransformerMixin):
    """Remove outliers via K-means clustering"""
    
    def __init__(self, n_clusters=3, min_cluster_size_ratio=0.1):
        self.n_clusters = n_clusters
        self.min_cluster_size_ratio = min_cluster_size_ratio
        self.kmeans = None
        self.valid_clusters_ = None
        self.mask_ = None
    
    def fit(self, X, y=None):
        del y  # Not used
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, 
            random_state=42, 
            n_init=10
        )
        labels = self.kmeans.fit_predict(X)
        counts = np.bincount(labels)
        min_size = int(len(X) * self.min_cluster_size_ratio)
        self.valid_clusters_ = np.where(counts >= min_size)[0]
        self.mask_ = np.isin(labels, self.valid_clusters_)
        return self
    
    def transform(self, X):
        labels = self.kmeans.predict(X)
        mask = np.isin(labels, self.valid_clusters_)
        return X[mask]
    
    def get_support_mask(self):
        return self.mask_


class OutlierFilterTransformer(BaseEstimator, TransformerMixin):
    """Filter out clusters smaller than a minimum size ratio."""

    def __init__(self, n_clusters=3, min_cluster_size_ratio=0.1, remove=True):
        self.n_clusters = n_clusters
        self.min_cluster_size_ratio = min_cluster_size_ratio
        self.remove = remove
        self.kmeans = None
        self.valid_clusters_ = None
        self.outlier_indices_ = None

    def fit(self, X, y=None):
        del y  # Not used
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(X)
        counts = np.bincount(labels)
        min_size = int(len(X) * self.min_cluster_size_ratio)
        self.valid_clusters_ = np.where(counts >= min_size)[0]
        mask = np.isin(labels, self.valid_clusters_)
        self.outlier_indices_ = np.where(~mask)[0]
        self.mask_ = mask
        return self

    def transform(self, X):
        labels = self.kmeans.predict(X)
        mask = np.isin(labels, self.valid_clusters_)
        if self.remove:
            return X[mask]
        return X

    def get_support_mask(self):
        return self.mask_


class HSICFeatureSelector(BaseEstimator, TransformerMixin):
    """Select top-k features using a simplified HSIC criterion."""

    def __init__(self, k=10, gamma=1.0):
        self.k = k
        self.gamma = gamma
        self.selected_features_ = None
        self.scores_ = None

    def _compute_hsic(self, X, y):
        n = X.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        Ky = pairwise_kernels(y.reshape(-1, 1), metric="rbf", gamma=self.gamma)
        scores = []
        for i in range(X.shape[1]):
            Kx = pairwise_kernels(X[:, i].reshape(-1, 1), metric="rbf", gamma=self.gamma)
            hsic = np.trace(Kx @ H @ Ky @ H) / (n - 1) ** 2
            scores.append(hsic)
        return np.array(scores)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.scores_ = self._compute_hsic(X, y)
        idx = np.argsort(self.scores_)[::-1]
        self.selected_features_ = idx[: self.k]
        return self

    def transform(self, X):
        return X[:, self.selected_features_]



class IsolationForestTransformer(BaseEstimator, TransformerMixin):
    """Remove outliers via Isolation Forest"""

    def __init__(self, contamination=0.1, n_estimators=100, random_state=42, remove=False):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.remove = remove
        self.iforest = None
        self.mask_ = None
    
    def fit(self, X, y=None):
        del y  # Not used
        self.iforest = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        labels = self.iforest.fit_predict(X)
        self.mask_ = labels != -1
        self.outlier_indices_ = np.where(labels == -1)[0]
        return self
    
    def transform(self, X):
        labels = self.iforest.predict(X)
        mask = labels != -1
        return X[mask] if self.remove else X
    
    def get_support_mask(self):
        return self.mask_


class LocalOutlierFactorTransformer(BaseEstimator, TransformerMixin):
    """Remove outliers via Local Outlier Factor"""

    def __init__(self, n_neighbors=20, contamination=0.1, remove=False):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.remove = remove
        self.lof = None
        self.mask_ = None
    
    def fit(self, X, y=None):
        del y  # Not used
        self.lof = LocalOutlierFactor(
            n_neighbors=min(self.n_neighbors, len(X) - 1),
            contamination=self.contamination,
            novelty=False,
            n_jobs=-1,
        )
        labels = self.lof.fit_predict(X)
        self.mask_ = labels != -1
        self.outlier_indices_ = np.where(labels == -1)[0]
        return self
    
    def transform(self, X):
        return X[self.mask_] if self.remove else X
    
    def get_support_mask(self):
        return self.mask_


class KMeansOutlierCV(BaseEstimator, TransformerMixin):
    """Cross-validated KMeans outlier removal"""
    
    def __init__(self, n_clusters_range=(2, 8), cv_folds=3, min_cluster_size_ratio=0.1):
        self.n_clusters_range = n_clusters_range
        self.cv_folds = cv_folds
        self.min_cluster_size_ratio = min_cluster_size_ratio
        self.best_n_clusters_ = None
        self.kmeans = None
        self.valid_clusters_ = None
        self.mask_ = None
    
    def fit(self, X, y=None):
        del y  # Not used
        from sklearn.model_selection import KFold
        from sklearn.metrics import silhouette_score
        
        best_score = -1
        best_n_clusters = self.n_clusters_range[0]
        
        # Cross-validate different numbers of clusters
        for n_clusters in range(self.n_clusters_range[0], self.n_clusters_range[1] + 1):
            scores = []
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            
            for train_idx, val_idx in kf.split(X):
                X_train = X[train_idx]
                if len(X_train) < n_clusters:
                    continue
                    
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_train)
                
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(X_train, labels)
                    scores.append(score)
            
            if scores:
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_n_clusters = n_clusters
        
        self.best_n_clusters_ = best_n_clusters
        
        # Fit final model with best number of clusters
        self.kmeans = KMeans(
            n_clusters=self.best_n_clusters_, 
            random_state=42, 
            n_init=10
        )
        labels = self.kmeans.fit_predict(X)
        counts = np.bincount(labels)
        min_size = int(len(X) * self.min_cluster_size_ratio)
        self.valid_clusters_ = np.where(counts >= min_size)[0]
        self.mask_ = np.isin(labels, self.valid_clusters_)
        
        return self
    
    def transform(self, X):
        labels = self.kmeans.predict(X)
        mask = np.isin(labels, self.valid_clusters_)
        return X[mask]
    
    def get_support_mask(self):
        return self.mask_ 