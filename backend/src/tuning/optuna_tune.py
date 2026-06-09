"""
使用 Optuna 对管道内分类器超参数进行调优。
目标为 5 折交叉验证的加权 F1，支持 SVC 与 RandomForest。
"""
import logging
from typing import Any, Callable

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score

from src.pipeline.model_pipeline import create_pipeline, SUPPORTED_CLASSIFIERS

logger = logging.getLogger(__name__)


def _objective(
    trial: optuna.Trial,
    X: np.ndarray,
    y: np.ndarray,
    classifier: str,
    cv: int,
    scoring: str,
) -> float:
    """Optuna 单次试验：根据 classifier 建议超参，构建管道并做交叉验证."""
    classifier = classifier.lower()
    if classifier not in SUPPORTED_CLASSIFIERS:
        raise ValueError(
            f"不支持的分类器: {classifier}，可选: {SUPPORTED_CLASSIFIERS}"
        )
    if classifier == "svc":
        params = {
            "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "poly"]),
        }
    else:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }

    pipe = create_pipeline(classifier=classifier, **params)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1)
    return float(scores.mean())


def run_study(
    X: np.ndarray,
    y: np.ndarray,
    classifier: str = "svc",
    n_trials: int = 30,
    cv: int = 5,
    scoring: str = "f1_weighted",
    random_state: int = 42,
) -> tuple[dict[str, Any], optuna.Study]:
    """
    运行 Optuna 调参，返回最佳超参与 study 对象。

    :param X: 训练特征
    :param y: 训练标签
    :param classifier: 分类器名称
    :param n_trials: 试验次数
    :param cv: 交叉验证折数
    :param scoring: 优化目标，如 f1_weighted, accuracy
    :param random_state: 随机种子
    :return: (best_params, study)
    """
    if classifier not in SUPPORTED_CLASSIFIERS:
        raise ValueError(
            f"不支持的分类器: {classifier}，可选: {SUPPORTED_CLASSIFIERS}"
        )

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state, n_startup_trials=10))
    objective_with_data: Callable[[optuna.Trial], float] = lambda t: _objective(
        t, X, y, classifier, cv, scoring
    )
    study.optimize(objective_with_data, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    logger.info("Optuna 最佳试验: value=%.4f, params=%s", study.best_value, best_params)
    return best_params, study
