"""
癌症数据集加载与划分。
使用 sklearn 内置威斯康星乳腺癌数据集，标准化接口便于后续替换为其他癌症数据集。
"""
import logging
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_cancer_data(
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    加载乳腺癌数据集并划分训练/测试集。

    :param test_size: 测试集比例
    :param random_state: 随机种子
    :return: 包含 X_train, X_test, y_train, y_test, feature_names, target_names 的字典
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = list(data.feature_names)
    target_names = list(data.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(
        "数据集加载完成: 样本总数=%d, 训练=%d, 测试=%d, 特征数=%d",
        len(X), len(X_train), len(X_test), X.shape[1],
    )

    return {
        "X_train": np.asarray(X_train),
        "X_test": np.asarray(X_test),
        "y_train": np.asarray(y_train),
        "y_test": np.asarray(y_test),
        "feature_names": feature_names,
        "target_names": target_names,
    }
