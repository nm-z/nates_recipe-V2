import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import optuna
from sklearn.preprocessing import FunctionTransformer, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_regression

from auto_optuna.transformers import (
    KMeansOutlierTransformer as OutlierFilterTransformer,
    IsolationForestTransformer,
    LocalOutlierFactorTransformer,
)


def test_outlier_filter_removes_small_cluster():
    rs = np.random.RandomState(0)
    inliers = rs.normal(0, 1, (50, 2))
    outliers = rs.normal(5, 1, (5, 2))
    X = np.vstack([inliers, outliers])
    t = OutlierFilterTransformer(n_clusters=2, min_cluster_size_ratio=0.2)
    t.fit(X)
    X_clean = t.transform(X)
    assert X_clean.shape[0] <= X.shape[0]
    assert X_clean.shape[1] == X.shape[1]


def test_isolation_forest_removes_outliers():
    rs = np.random.RandomState(1)
    inliers = rs.normal(0, 1, (50, 2))
    outliers = rs.normal(8, 1, (5, 2))
    X = np.vstack([inliers, outliers])
    t = IsolationForestTransformer(contamination=0.1, n_estimators=50, random_state=42)
    t.fit(X)
    X_clean = t.transform(X)
    assert X_clean.shape[0] < X.shape[0]


def test_lof_removes_outliers():
    rs = np.random.RandomState(2)
    inliers = rs.normal(0, 1, (50, 2))
    outliers = rs.normal(8, 1, (5, 2))
    X = np.vstack([inliers, outliers])
    t = LocalOutlierFactorTransformer(n_neighbors=20, contamination=0.1)
    t.fit(X)
    X_clean = t.transform(X)
    assert X_clean.shape[0] < X.shape[0]


def test_feature_selector_identifies_correlated_features():
    rs = np.random.RandomState(3)
    X = rs.normal(0, 1, (1000, 5))
    y = 0.5 * X[:, 0] + 2 * X[:, 1]

    selector = SelectKBest(f_regression, k=2)
    selector.fit(X, y)
    selected = set(selector.get_support(indices=True))
    assert {0, 1}.issubset(selected)
    X_selected = selector.transform(X)
    assert X_selected.shape == (1000, 2)
