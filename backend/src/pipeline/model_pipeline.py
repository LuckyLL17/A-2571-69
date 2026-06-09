"""
管道化模型：StandardScaler + 特征选择 + 分类器。
支持多种 sklearn 分类器，便于 Optuna 传入超参构建管道。
优化点：增加 LogisticRegression / KNN / GradientBoosting / XGBoost，
        管道内加入 SelectKBest 特征选择步骤。
"""
import logging
from typing import Any

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)

# 支持的分类器列表（扩展后）
SUPPORTED_CLASSIFIERS = (
    "svc",
    "random_forest",
    "logistic_regression",
    "knn",
    "gradient_boosting",
)


def create_pipeline(
    classifier: str = "svc",
    k_best: int | None = None,
    **kwargs: Any,
) -> Pipeline:
    """
    构建「标准化 + 特征选择(可选) + 分类器」管道。

    :param classifier: 分类器名称，支持 svc, random_forest, logistic_regression, knn, gradient_boosting
    :param k_best: 特征选择保留的特征数，None 表示不进行特征选择（保留全部）
    :param kwargs: 传入分类器的超参（如 C, gamma, n_estimators 等）
    :return: sklearn Pipeline
    """
    classifier = classifier.lower()
    if classifier not in SUPPORTED_CLASSIFIERS:
        raise ValueError(
            f"不支持的分类器: {classifier}，可选: {SUPPORTED_CLASSIFIERS}"
        )

    # 根据分类器名称实例化对应的模型
    # 注意：使用 setdefault 设置默认值，避免与 kwargs 中的同名参数冲突
    if classifier == "svc":
        kwargs.setdefault("probability", True)
        clf = SVC(**kwargs)
    elif classifier == "random_forest":
        clf = RandomForestClassifier(**kwargs)
    elif classifier == "logistic_regression":
        # solver 和 max_iter 使用 setdefault，允许 Optuna 通过 kwargs 覆盖
        kwargs.setdefault("solver", "lbfgs")
        kwargs.setdefault("max_iter", 5000)
        clf = LogisticRegression(**kwargs)
    elif classifier == "knn":
        clf = KNeighborsClassifier(**kwargs)
    elif classifier == "gradient_boosting":
        clf = GradientBoostingClassifier(**kwargs)

    # 构建管道步骤列表
    steps = [("scaler", StandardScaler())]

    # 如果指定了 k_best 且大于 0，则加入 SelectKBest 特征选择
    if k_best is not None and k_best > 0:
        steps.append(("select_kbest", SelectKBest(score_func=f_classif, k=k_best)))

    steps.append(("classifier", clf))

    pipe = Pipeline(steps)
    logger.debug("管道已创建: StandardScaler + %s%s",
                 f"SelectKBest(k={k_best}) + " if k_best else "",
                 classifier)
    return pipe
