import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pathlib import Path
import joblib
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score


class OutlierFilterTransformer(BaseEstimator, TransformerMixin):
    """Identify outliers using K-Means clusters and optionally remove them."""

    def __init__(self, n_clusters=3, min_cluster_size_ratio=0.1, remove=True):
        self.n_clusters = n_clusters
        self.min_cluster_size_ratio = min_cluster_size_ratio
        self.remove = remove
        self.kmeans = None
        self.valid_clusters_ = None
        self.outlier_indices_ = None

    def fit(self, X, y=None):
        del y
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


class IsolationForestTransformer(BaseEstimator, TransformerMixin):
    """Isolation Forest outlier detection with optional removal."""

    def __init__(self, contamination=0.1, n_estimators=100, random_state=42, remove=True):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.remove = remove
        self.iforest = None
        self.outlier_indices_ = None

    def fit(self, X, y=None):
        del y
        self.iforest = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        labels = self.iforest.fit_predict(X)
        self.outlier_indices_ = np.where(labels == -1)[0]
        self.mask_ = labels != -1
        return self

    def transform(self, X):
        labels = self.iforest.predict(X)
        mask = labels != -1
        if self.remove:
            return X[mask]
        return X


class LocalOutlierFactorTransformer(BaseEstimator, TransformerMixin):
    """Local Outlier Factor detector with optional removal."""

    def __init__(self, n_neighbors=20, contamination=0.1, remove=True):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.remove = remove
        self.lof = None
        self.outlier_indices_ = None

    def fit(self, X, y=None):
        del y
        self.lof = LocalOutlierFactor(
            n_neighbors=min(self.n_neighbors, len(X) - 1),
            contamination=self.contamination,
            novelty=False,
            n_jobs=-1,
        )
        labels = self.lof.fit_predict(X)
        self.outlier_indices_ = np.where(labels == -1)[0]
        self.mask_ = labels != -1
        return self

    def transform(self, X):
        if self.remove:
            return X[self.mask_]
        return X


class HSICFeatureSelector(BaseEstimator, TransformerMixin):
    """Simplified feature selector using correlation as HSIC proxy."""

    def __init__(self, k=10):
        self.k = k
        self.selected_features_ = None

    def fit(self, X, y):
        corrs = [abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])]
        self.selected_features_ = np.argsort(corrs)[-self.k:]
        return self

    def transform(self, X):
        return X[:, self.selected_features_]


class BattleTestedOptimizer:
    """Simplified optimizer used for tests."""

    def __init__(self, dataset_num, target_r2=0.93, max_trials=40):
        import battle_tested_optuna_playbook as bt

        self.dataset_num = dataset_num
        self.target_r2 = target_r2
        self.max_trials = max_trials
        self.model_dir = bt.Path(".")
        self.cv = None
        self.logger = None

    @staticmethod
    def create_target_transformer(trial):
        choice = trial.params.get("y_transform", "none")
        if choice == "none":
            return None
        if choice == "log1p":
            return FunctionTransformer(
                np.log1p, inverse_func=np.expm1, check_inverse=False
            )
        if choice == "power":
            return PowerTransformer()
        raise ValueError(f"Unknown y_transform: {choice}")

    def step_1_pin_down_ceiling(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.X = X_train
        self.y = y_train
        self.X_test = X_test
        self.y_test = y_test

        model = LinearRegression()
        model.fit(X_train, y_train)
        baseline = r2_score(y_test, model.predict(X_test))
        ceiling = min(1.0, baseline + 0.1)

        self.noise_ceiling = ceiling
        self.current_best_r2 = baseline
        return ceiling, baseline

    def step_2_bulletproof_preprocessing(self):
        scaler = StandardScaler().fit(self.X)
        self.preprocessing_pipeline = scaler
        self.X_clean = scaler.transform(self.X)
        self.X_test_clean = scaler.transform(self.X_test)
        return self.X_clean.shape[1]

    def step_3_optuna_search(self):
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        )
        pipeline.fit(self.X_clean, self.y)
        self.best_pipeline = pipeline

    def step_4_lock_in_champion(self):
        import battle_tested_optuna_playbook as bt

        y_pred = self.best_pipeline.predict(self.X_test_clean)
        final_r2 = r2_score(self.y_test, y_pred)
        self.final_pipeline = self.best_pipeline
        model_file = bt.Path(self.model_dir) / "best_model.pkl"
        joblib.dump(self.best_pipeline, model_file)
        return final_r2, {"model": "LinearRegression"}

    def run_optimization(self, X, y):
        self.step_1_pin_down_ceiling(X, y)
        self.step_2_bulletproof_preprocessing()
        self.step_3_optuna_search()
        return self.step_4_lock_in_champion()

