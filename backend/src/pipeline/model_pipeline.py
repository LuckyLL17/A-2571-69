"""
管道化模型：StandardScaler + 分类器。
支持多种 sklearn 分类器，便于 Optuna 传入超参构建管道。
"""
import logging
from typing import Any

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

SUPPORTED_CLASSIFIERS = ("svc", "random_forest")


def create_pipeline(
    classifier: str = "svc",
    **kwargs: Any,
) -> Pipeline:
    """
    构建「标准化 + 分类器」管道。

    :param classifier: 分类器名称，支持 svc, random_forest
    :param kwargs: 传入分类器的超参（如 C, gamma, n_estimators 等）
    :return: sklearn Pipeline
    """
    classifier = classifier.lower()
    if classifier not in SUPPORTED_CLASSIFIERS:
        raise ValueError(
            f"不支持的分类器: {classifier}，可选: {SUPPORTED_CLASSIFIERS}"
        )

    if classifier == "svc":
        clf = SVC(probability=True, **kwargs)
    else:
        clf = RandomForestClassifier(**kwargs)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf),
    ])
    logger.debug("管道已创建: StandardScaler + %s", classifier)
    return pipe
