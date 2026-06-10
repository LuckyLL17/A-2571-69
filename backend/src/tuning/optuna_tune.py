"""
使用 Optuna 对管道内分类器超参数进行调优。
目标为 5 折交叉验证的加权 F1，支持 SVC / RandomForest / LogisticRegression / KNN / GradientBoosting。
优化点：扩展超参搜索空间，加入特征选择 k_best 的调优。
"""
import logging
from typing import Any, Callable

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score

from src.pipeline.model_pipeline import create_pipeline, SUPPORTED_CLASSIFIERS

logger = logging.getLogger(__name__)


def _suggest_params(
    trial: optuna.Trial,
    classifier: str,
    n_features: int,
) -> dict[str, Any]:
    """
    根据分类器类型，让 Optuna trial 建议一组超参数。

    :param trial: Optuna Trial 对象
    :param classifier: 分类器名称
    :param n_features: 输入特征总数，用于设定 k_best 上界
    :return: 超参字典，包含分类器参数和可选的 k_best
    """
    classifier = classifier.lower()
    params: dict[str, Any] = {}

    # 是否启用特征选择（约 50% 概率开启）
    use_feature_selection = trial.suggest_categorical("use_feature_selection", [True, False])
    if use_feature_selection:
        # k_best 在 [5, n_features] 之间选择
        k_best = trial.suggest_int("k_best", 5, n_features)
        params["k_best"] = k_best

    # 各分类器的超参搜索空间
    if classifier == "svc":
        params.update({
            "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 10.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
        })
    elif classifier == "random_forest":
        params.update({
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        })
    elif classifier == "logistic_regression":
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        # l1 正则仅支持 saga solver；l2 正则支持 lbfgs 和 saga
        if penalty == "l1":
            solver = "saga"
        else:
            solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
        params.update({
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "penalty": penalty,
            "solver": solver,
        })
    elif classifier == "knn":
        params.update({
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 30),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 2),  # 1=曼哈顿, 2=欧氏
        })
    elif classifier == "gradient_boosting":
        params.update({
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        })

    return params


def _objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    classifier: str,
    cv: int,
    scoring: str,
) -> float:
    """Optuna 单次试验：根据 classifier 建议超参，构建管道并做交叉验证。"""
    classifier = classifier.lower()
    if classifier not in SUPPORTED_CLASSIFIERS:
        raise ValueError(
            f"不支持的分类器: {classifier}，可选: {SUPPORTED_CLASSIFIERS}"
        )

    # 获取本次试验的超参建议
    params = _suggest_params(trial, classifier, X.shape[1])

    # 提取 k_best 参数（不传给分类器）
    k_best = params.pop("k_best", None)
    # 移除 use_feature_selection 标记（不传给分类器）
    params.pop("use_feature_selection", None)

    pipe = create_pipeline(classifier=classifier, k_best=k_best, **params)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return float(scores.mean())


def run_study(
    X: np.ndarray,
    y: np.ndarray,
    classifier: str = "svc",
    n_trials: int = 50,
    cv: int = 5,
    scoring: str = "f1_weighted",
    random_state: int = 42,
) -> tuple[dict[str, Any], optuna.Study]:
    """
    运行 Optuna 调参，返回最佳超参与 study 对象。

    :param X: 训练特征
    :param y: 训练标签
    :param classifier: 分类器名称
    :param n_trials: 试验次数（默认提升到 50）
    :param cv: 交叉验证折数
    :param scoring: 优化目标，如 f1_weighted, accuracy
    :param random_state: 随机种子
    :return: (best_params, study)
    """
    if classifier not in SUPPORTED_CLASSIFIERS:
        raise ValueError(
            f"不支持的分类器: {classifier}，可选: {SUPPORTED_CLASSIFIERS}"
        )

    # 使用 TPESampler 进行贝叶斯优化，n_startup_trials 控制初始随机探索次数
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state, n_startup_trials=10),
    )
    objective_with_data: Callable[[optuna.Trial], float] = lambda t: _objective(
        t, X, y, classifier, cv, scoring
    )
    study.optimize(objective_with_data, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    logger.info("Optuna 最佳试验: value=%.4f, params=%s", study.best_value, best_params)
    return best_params, study
