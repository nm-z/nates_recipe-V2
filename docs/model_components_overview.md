## Machine Learning Components

Of course. Based on the contents of the files in this repository, here is a comprehensive list of the models, preprocessors, transformers, and optimizers used in this project.

### 1. Machine Learning Models (Regressors)
These are the predictive models used in the model training and selection phases.

* **Linear Models**
    * `Ridge` and `RidgeCV`: Linear least squares with L2 regularization.
    * `Lasso`: Linear Model trained with L1 prior as regularizer.
    * `ElasticNet`: Linear regression with combined L1 and L2 priors as regularizer.
* **Support Vector Machines**
    * `SVR` (Support Vector Regressor).
* **Tree-Based Ensembles**
    * `DecisionTreeRegressor`.
    * `RandomForestRegressor`.
    * `ExtraTreesRegressor`.
    * `GradientBoostingRegressor`.
    * `AdaBoostRegressor`.
* **Neural Networks**
    * `MLPRegressor` (Multi-layer Perceptron).
* **Optional Advanced Models**
    * `xgboost.XGBRegressor` (if installed).
    * `lightgbm.LGBMRegressor` (if installed).

### 2. Preprocessors and Transformers

#### **Custom Transformers**
These are defined locally within the scripts to perform specialized outlier detection and feature selection.

* `OutlierFilterTransformer`: Uses K-Means clustering to identify and remove outlier data points.
* `IsolationForestTransformer`: Wraps the `IsolationForest` algorithm to detect and remove outliers.
* `LocalOutlierFactorTransformer`: Wraps the `LocalOutlierFactor` algorithm for outlier removal.
* `HSICFeatureSelector`: A custom feature selector that uses correlation as a simplified proxy for the Hilbert-Schmidt Independence Criterion.

#### **Standard `scikit-learn` Preprocessors**
Used within pipeline objects to perform scaling, feature selection, and dimensionality reduction.

* **Scaling:**
    * `StandardScaler`
    * `RobustScaler`
    * `QuantileTransformer`
    * `PowerTransformer`
    * `MinMaxScaler`
* **Feature Selection:**
    * `VarianceThreshold`: Removes features with zero or low variance.
    * `SelectKBest`: Selects features with the best scores from a scoring function.
    * `RFECV` (Recursive Feature Elimination with Cross-Validation).
* **Dimensionality Reduction:**
    * `PCA` (Principal Component Analysis).
    * `KernelPCA`.
    * `TruncatedSVD`.
* **Target Transformation:**
    * `TransformedTargetRegressor`: Used to apply transformations (like `log1p`) to the target variable `y`.

### 3. Optimizers and Utilities

* **Pipeline Orchestrators:**
    * `SystematicOptimizer`: Main class in `auto_optuna-V1.2.py` that runs the 3-phase optimization.
    * `SystematicOptimizerV13`: Updated version in `auto_optuna-V1.3.py` that inherits from and extends the V1.2 optimizer.
    * `BattleTestedOptimizer`: Primary class used in the older `battle_tested_optuna_playbook.py` and `auto_optuna-V1.py` scripts.
* **Hyperparameter Optimization (Optuna):**
    * `optuna.create_study`: Core function to start an optimization session.
    * `MedianPruner`: Used to automatically stop unpromising trials early.
    * `TPESampler`: Default Tree-structured Parzen Estimator sampler for suggesting new hyperparameters. 