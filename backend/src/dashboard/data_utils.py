"""
仪表盘数据处理工具模块。
负责数据加载、模型训练、预测结果生成以及可视化数据准备。
"""
import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
)

from src.data.load_data import load_cancer_data
from src.pipeline.model_pipeline import create_pipeline
from src.tuning.optuna_tune import run_study

logger = logging.getLogger(__name__)


class DashboardDataManager:
    """
    仪表盘数据管理器。
    负责管理数据、模型和计算各种可视化所需的指标。
    """

    def __init__(self, random_state: int = 42, test_size: float = 0.2):
        """
        初始化数据管理器。

        :param random_state: 随机种子
        :param test_size: 测试集比例
        """
        self.random_state = random_state
        self.test_size = test_size
        self.data = None
        self.model = None
        self.predictions = None
        self.probabilities = None
        self.best_params = None
        self.study = None
        self.shap_values = None
        self.shap_summary = None
        self.shap_sample_indices = None  # SHAP计算使用的样本索引

    def load_data(self) -> Dict[str, Any]:
        """
        加载癌症数据集。

        :return: 包含数据集信息的字典
        """
        logger.info("加载癌症数据集...")
        self.data = load_cancer_data(
            test_size=self.test_size,
            random_state=self.random_state,
        )
        logger.info(f"数据集加载完成: 训练集 {len(self.data['X_train'])} 样本, "
                    f"测试集 {len(self.data['X_test'])} 样本")
        return self.data

    def train_model(
        self,
        classifier: str = "svc",
        n_trials: int = 30,
        use_optuna: bool = True,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        训练模型，支持使用Optuna调参或默认参数。

        :param classifier: 分类器类型 (svc, random_forest)
        :param n_trials: Optuna试验次数
        :param use_optuna: 是否使用Optuna调参
        :return: (训练好的模型, 最佳参数字典)
        """
        if self.data is None:
            self.load_data()

        if use_optuna:
            logger.info(f"使用Optuna调参训练 {classifier} 模型，试验次数: {n_trials}")
            self.best_params, self.study = run_study(
                self.data["X_train"],
                self.data["y_train"],
                classifier=classifier,
                n_trials=n_trials,
                cv=5,
                scoring="f1_weighted",
                random_state=self.random_state,
            )
        else:
            logger.info(f"使用默认参数训练 {classifier} 模型")
            self.best_params = {}

        # 创建并训练模型
        self.model = create_pipeline(classifier=classifier, **self.best_params)
        self.model.fit(self.data["X_train"], self.data["y_train"])

        # 生成预测结果
        self._generate_predictions()

        logger.info("模型训练完成")
        return self.model, self.best_params

    def _generate_predictions(self) -> None:
        """
        生成预测结果和概率。
        """
        if self.model is None or self.data is None:
            raise ValueError("模型或数据未初始化")

        self.predictions = self.model.predict(self.data["X_test"])
        self.probabilities = self.model.predict_proba(self.data["X_test"])
        logger.info("预测结果生成完成")

    def get_metrics(self) -> Dict[str, float]:
        """
        计算模型性能指标。

        :return: 包含各项指标的字典
        """
        if self.predictions is None or self.data is None:
            raise ValueError("预测结果或数据未初始化")

        y_test = self.data["y_test"]
        y_pred = self.predictions

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "precision_0": precision_score(y_test, y_pred, pos_label=0),
            "recall_0": recall_score(y_test, y_pred, pos_label=0),
            "f1_0": f1_score(y_test, y_pred, pos_label=0),
            "precision_1": precision_score(y_test, y_pred, pos_label=1),
            "recall_1": recall_score(y_test, y_pred, pos_label=1),
            "f1_1": f1_score(y_test, y_pred, pos_label=1),
        }

        # 添加最佳CV分数（如果使用了Optuna）
        if self.study is not None:
            metrics["best_cv_score"] = self.study.best_value

        return metrics

    def get_confusion_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        获取混淆矩阵数据。

        :return: (混淆矩阵数组, 类别名称列表)
        """
        if self.predictions is None or self.data is None:
            raise ValueError("预测结果或数据未初始化")

        cm = confusion_matrix(self.data["y_test"], self.predictions)
        target_names = self.data["target_names"]
        return cm, target_names

    def get_roc_data(self) -> Dict[str, Any]:
        """
        获取ROC曲线数据。

        :return: 包含ROC曲线数据的字典
        """
        if self.probabilities is None or self.data is None:
            raise ValueError("预测概率或数据未初始化")

        y_test = self.data["y_test"]
        y_score = self.probabilities[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        return {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": roc_auc,
        }

    def get_classification_report(self) -> str:
        """
        获取分类报告文本。

        :return: 分类报告字符串
        """
        if self.predictions is None or self.data is None:
            raise ValueError("预测结果或数据未初始化")

        return classification_report(
            self.data["y_test"],
            self.predictions,
            target_names=self.data["target_names"],
        )

    def get_feature_dataframe(self, dataset: str = "test") -> pd.DataFrame:
        """
        获取特征数据DataFrame，便于可视化。

        :param dataset: 数据集类型 ('train' 或 'test')
        :return: 特征数据DataFrame
        """
        if self.data is None:
            raise ValueError("数据未初始化")

        if dataset == "train":
            X = self.data["X_train"]
            y = self.data["y_train"]
        else:
            X = self.data["X_test"]
            y = self.data["y_test"]

        df = pd.DataFrame(X, columns=self.data["feature_names"])
        df["target"] = y
        df["target_name"] = [self.data["target_names"][int(label)] for label in y]

        return df

    def get_optuna_history(self) -> pd.DataFrame:
        """
        获取Optuna调参历史数据。

        :return: 调参历史DataFrame
        """
        if self.study is None:
            raise ValueError("未进行Optuna调参")

        trials_data = []
        for trial in self.study.trials:
            if trial.state.is_completed():
                trial_dict = {
                    "trial_number": trial.number,
                    "value": trial.value,
                    **trial.params,
                }
                trials_data.append(trial_dict)

        return pd.DataFrame(trials_data)

    def compute_shap_values(self, sample_size: int = 100) -> Tuple[np.ndarray, Dict]:
        """
        计算SHAP值，并记录使用的样本索引。

        :param sample_size: 抽样样本数
        :return: (SHAP值数组, SHAP摘要字典)
        """
        from src.interpretation.shap_explain import explain_model

        if self.model is None or self.data is None:
            raise ValueError("模型或数据未初始化")

        logger.info(f"计算SHAP值，样本数: {sample_size}")
        
        # 记录抽样的样本索引，用于后续单样本解释
        X_test = self.data["X_test"]
        if sample_size is not None and len(X_test) > sample_size:
            rng = np.random.default_rng(42)
            self.shap_sample_indices = rng.choice(len(X_test), size=sample_size, replace=False)
        else:
            self.shap_sample_indices = np.arange(len(X_test))
        
        self.shap_values, self.shap_summary = explain_model(
            self.model,
            self.data["X_test"],
            self.data["feature_names"],
            output_dir="output",
            max_display=15,
            sample_size=sample_size,
        )

        logger.info("SHAP值计算完成")
        return self.shap_values, self.shap_summary

    def get_shap_dataframe(self) -> pd.DataFrame:
        """
        获取SHAP值DataFrame，便于可视化。

        :return: SHAP值DataFrame
        """
        if self.shap_values is None or self.shap_summary is None or self.data is None:
            raise ValueError("SHAP值未计算或数据未初始化")

        # 获取测试集的标准化数据（仅使用有SHAP值的样本）
        X_test = self.data["X_test"][self.shap_sample_indices]
        X_scaled = self.model[:-1].transform(X_test)

        n_samples = len(self.shap_values)

        df_shap = pd.DataFrame(
            self.shap_values,
            columns=self.data["feature_names"],
        )
        df_values = pd.DataFrame(
            X_scaled,
            columns=self.data["feature_names"],
        )

        # 合并SHAP值和特征值，重塑为长格式
        shap_long = df_shap.reset_index().melt(
            id_vars="index",
            var_name="feature",
            value_name="shap_value",
        )
        value_long = df_values.reset_index().melt(
            id_vars="index",
            var_name="feature",
            value_name="feature_value",
        )

        merged = shap_long.merge(value_long, on=["index", "feature"])
        merged["sample_id"] = merged["index"]

        return merged

    def get_single_sample_prediction(self, sample_idx: int) -> Dict[str, Any]:
        """
        获取单样本预测解释数据。

        :param sample_idx: SHAP样本列表中的索引（不是原始测试集索引）
        :return: 包含样本预测信息的字典
        """
        if self.model is None or self.data is None:
            raise ValueError("模型或数据未初始化")

        if self.shap_values is None or self.shap_sample_indices is None:
            self.compute_shap_values()

        # 转换为原始测试集索引
        original_idx = int(self.shap_sample_indices[sample_idx])
        
        X_sample = self.data["X_test"][original_idx:original_idx + 1]
        y_true = self.data["y_test"][original_idx]
        y_pred = self.predictions[original_idx]
        prob = self.probabilities[original_idx]

        shap_sample = self.shap_values[sample_idx]
        feature_names = self.data["feature_names"]

        # 按绝对SHAP值排序
        order = np.argsort(np.abs(shap_sample))[::-1]
        # 确保order是整数数组
        order = np.asarray(order, dtype=int)
        top_features = [feature_names[int(i)] for i in order[:10]]
        top_shap = [float(shap_sample[int(i)]) for i in order[:10]]
        top_values = [float(X_sample[0][int(i)]) for i in order[:10]]

        return {
            "sample_idx": sample_idx,
            "original_idx": original_idx,
            "y_true": y_true,
            "y_true_name": self.data["target_names"][int(y_true)],
            "y_pred": y_pred,
            "y_pred_name": self.data["target_names"][int(y_pred)],
            "probability": prob,
            "probability_benign": float(prob[0]),
            "probability_malignant": float(prob[1]),
            "top_features": top_features,
            "top_shap_values": top_shap,
            "top_feature_values": top_values,
            "all_feature_names": feature_names,
            "all_feature_values": [float(v) for v in X_sample[0].tolist()],
        }
