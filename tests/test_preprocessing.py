import importlib.util
import numpy as np
import optuna
from sklearn.preprocessing import FunctionTransformer, PowerTransformer

# Load module with custom transformers
def load_module():
    spec = importlib.util.spec_from_file_location('auto_optuna_V1', 'auto_optuna-V1.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

mod = load_module()
OutlierFilterTransformer = mod.OutlierFilterTransformer
IsolationForestTransformer = mod.IsolationForestTransformer
LocalOutlierFactorTransformer = mod.LocalOutlierFactorTransformer
HSICFeatureSelector = mod.HSICFeatureSelector
BattleTestedOptimizer = mod.BattleTestedOptimizer


def test_outlier_filter_removes_small_cluster():
    rs = np.random.RandomState(0)
    inliers = rs.normal(0, 1, (50, 2))
    outliers = rs.normal(5, 1, (5, 2))
    X = np.vstack([inliers, outliers])
    t = OutlierFilterTransformer(n_clusters=2, min_cluster_size_ratio=0.2, remove=True)
    t.fit(X)
    X_clean = t.transform(X)
    assert t.outlier_indices_.size == 5
    assert X_clean.shape == (50, 2)


def test_isolation_forest_removes_outliers():
    rs = np.random.RandomState(1)
    inliers = rs.normal(0, 1, (50, 2))
    outliers = rs.normal(8, 1, (5, 2))
    X = np.vstack([inliers, outliers])
    t = IsolationForestTransformer(contamination=0.1, n_estimators=50, random_state=42, remove=True)
    t.fit(X)
    X_clean = t.transform(X)
    assert len(t.outlier_indices_) > 0
    assert X_clean.shape[0] < X.shape[0]


def test_lof_removes_outliers():
    rs = np.random.RandomState(2)
    inliers = rs.normal(0, 1, (50, 2))
    outliers = rs.normal(8, 1, (5, 2))
    X = np.vstack([inliers, outliers])
    t = LocalOutlierFactorTransformer(n_neighbors=20, contamination=0.1, remove=True)
    t.fit(X)
    X_clean = t.transform(X)
    assert len(t.outlier_indices_) > 0
    assert X_clean.shape[0] < X.shape[0]


def test_hsic_feature_selector_identifies_correlated_features():
    rs = np.random.RandomState(3)
    X = rs.normal(0, 1, (1000, 5))
    y = 0.5 * X[:, 0] + 2 * X[:, 1]
    selector = HSICFeatureSelector(k=2)
    selector.fit(X, y)
    assert set(selector.selected_features_) == {0, 1}
    X_selected = selector.transform(X)
    assert X_selected.shape == (1000, 2)


def test_create_target_transformer():
    optimizer = BattleTestedOptimizer.__new__(BattleTestedOptimizer)
    none_trial = optuna.trial.FixedTrial({'y_transform': 'none'})
    log_trial = optuna.trial.FixedTrial({'y_transform': 'log1p'})
    power_trial = optuna.trial.FixedTrial({'y_transform': 'power'})

    assert BattleTestedOptimizer.create_target_transformer(optimizer, none_trial) is None

    log_t = BattleTestedOptimizer.create_target_transformer(optimizer, log_trial)
    assert isinstance(log_t, FunctionTransformer)

    power_t = BattleTestedOptimizer.create_target_transformer(optimizer, power_trial)
    assert isinstance(power_t, PowerTransformer)
