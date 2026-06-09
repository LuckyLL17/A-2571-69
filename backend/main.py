"""
癌症数据集分类预测主入口。
流程：加载数据 -> 管道化建模(标准化+特征选择+分类器) -> Optuna 调参 -> 训练最佳模型
      -> 评估(含 ROC-AUC、混淆矩阵) -> SHAP 特征解释 -> 结果保存。
优化点：支持多分类器比较，增加 ROC-AUC / 混淆矩阵等评估指标，保存可视化结果供仪表盘使用。
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 将 backend 根目录加入 path，便于以 python main.py 或模块方式运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.load_data import load_cancer_data
from src.pipeline.model_pipeline import create_pipeline, SUPPORTED_CLASSIFIERS
from src.tuning.optuna_tune import run_study
from src.interpretation.shap_explain import explain_model
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="癌症数据集分类：管道 + Optuna + SHAP")
    parser.add_argument(
        "--classifier", type=str, default="svc",
        choices=list(SUPPORTED_CLASSIFIERS),
    )
    parser.add_argument("--n_trials", type=int, default=50, help="Optuna 试验次数")
    parser.add_argument("--output-dir", type=str, default="output", help="结果与图表输出目录")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    # 新增：是否运行全部分类器进行对比
    parser.add_argument("--compare-all", action="store_true",
                        help="运行所有支持的分类器并比较结果")
    return parser.parse_args()


def _json_default(obj):
    """JSON 序列化辅助：处理 numpy 类型。"""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def evaluate_and_save(pipeline, X_test, y_test, feature_names, target_names, output_dir, classifier_name):
    """
    对训练好的管道模型进行全面评估，保存指标和可视化图表。

    :param pipeline: 已 fit 的 Pipeline
    :param X_test: 测试集特征
    :param y_test: 测试集标签
    :param feature_names: 特征名列表
    :param target_names: 目标类别名列表
    :param output_dir: 输出目录
    :param classifier_name: 分类器名称（用于图表标题）
    :return: 评估指标字典
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # 正类概率

    # 基础指标
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred)

    logger.info("[%s] 测试集 Accuracy: %.4f, F1 (weighted): %.4f, ROC-AUC: %.4f",
                classifier_name, accuracy, f1, roc_auc)
    logger.info("[%s] 分类报告:\n%s", classifier_name, report)

    # 保存混淆矩阵图
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=target_names,
        yticklabels=target_names,
        title=f"混淆矩阵 - {classifier_name}",
        ylabel="真实标签",
        xlabel="预测标签",
    )
    # 在每个格子中显示数值
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black")
    fig.tight_layout()
    cm_path = output_dir / f"confusion_matrix_{classifier_name}.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    logger.info("混淆矩阵图已保存: %s", cm_path)

    # 保存 ROC 曲线图
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("假正率 (FPR)")
    ax.set_ylabel("真正率 (TPR)")
    ax.set_title(f"ROC 曲线 - {classifier_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    roc_path = output_dir / f"roc_curve_{classifier_name}.png"
    fig.savefig(roc_path, dpi=120)
    plt.close(fig)
    logger.info("ROC 曲线图已保存: %s", roc_path)

    metrics = {
        "classifier": classifier_name,
        "test_accuracy": accuracy,
        "test_f1_weighted": f1,
        "test_roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }
    return metrics


def run_single_classifier(classifier, data, args, output_dir):
    """
    对单个分类器运行完整流程：Optuna 调参 -> 训练 -> 评估 -> SHAP 解释。

    :return: (pipeline, metrics_dict, study, shap_summary)
    """
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    feature_names = data["feature_names"]
    target_names = data["target_names"]

    logger.info("=== 分类器: %s ===", classifier)
    logger.info("Optuna 调参中...")
    best_params, study = run_study(
        X_train, y_train,
        classifier=classifier,
        n_trials=args.n_trials,
        cv=5,
        scoring="f1_weighted",
        random_state=args.random_state,
    )

    # 从 best_params 中提取 k_best 和 use_feature_selection（不传给分类器）
    pipeline_params = dict(best_params)
    k_best = pipeline_params.pop("k_best", None)
    pipeline_params.pop("use_feature_selection", None)

    logger.info("使用最佳超参训练管道: %s", best_params)
    pipeline = create_pipeline(classifier=classifier, k_best=k_best, **pipeline_params)
    pipeline.fit(X_train, y_train)

    # 评估
    clf_output_dir = output_dir / classifier
    metrics = evaluate_and_save(
        pipeline, X_test, y_test, feature_names, target_names,
        clf_output_dir, classifier,
    )
    metrics["best_params"] = best_params
    metrics["best_cv_value"] = study.best_value

    # SHAP 特征解释
    logger.info("SHAP 特征解释中...")
    shap_values, shap_summary = explain_model(
        pipeline, X_test, feature_names,
        output_dir=clf_output_dir,
        max_display=15,
        sample_size=100,
    )
    # 保存特征重要性顺序
    importance_path = clf_output_dir / "feature_importance_order.json"
    with open(importance_path, "w", encoding="utf-8") as f:
        json.dump(shap_summary["importance_order"], f, indent=2, ensure_ascii=False)

    return pipeline, metrics, study, shap_summary


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("1. 加载癌症数据集")
    data = load_cancer_data(
        test_size=args.test_size,
        random_state=args.random_state,
    )

    all_metrics = {}

    if args.compare_all:
        # 对所有支持的分类器分别运行完整流程
        classifiers_to_run = list(SUPPORTED_CLASSIFIERS)
        logger.info("将依次运行以下分类器: %s", classifiers_to_run)
    else:
        classifiers_to_run = [args.classifier]

    for clf_name in classifiers_to_run:
        pipeline, metrics, study, shap_summary = run_single_classifier(
            clf_name, data, args, output_dir,
        )
        all_metrics[clf_name] = metrics

    # 保存所有分类器的对比指标
    comparison = {}
    for clf_name, m in all_metrics.items():
        comparison[clf_name] = {
            "accuracy": m["test_accuracy"],
            "f1_weighted": m["test_f1_weighted"],
            "roc_auc": m["test_roc_auc"],
            "best_cv_value": m["best_cv_value"],
            "best_params": m["best_params"],
        }

    comparison_path = output_dir / "comparison.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False, default=_json_default)
    logger.info("分类器对比结果已保存: %s", comparison_path)

    # 如果有多个分类器，生成对比柱状图
    if len(all_metrics) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        clf_names = list(all_metrics.keys())
        metric_keys = ["test_accuracy", "test_f1_weighted", "test_roc_auc"]
        metric_labels = ["Accuracy", "F1 (weighted)", "ROC-AUC"]
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

        for ax, key, label in zip(axes, metric_keys, metric_labels):
            values = [all_metrics[c][key] for c in clf_names]
            bars = ax.bar(clf_names, values, color=colors[:len(clf_names)])
            ax.set_ylim([min(values) - 0.05, 1.0])
            ax.set_ylabel(label)
            ax.set_title(label)
            # 在柱子上方显示数值
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=9)
            ax.tick_params(axis="x", rotation=30)

        fig.suptitle("分类器性能对比", fontsize=14, fontweight="bold")
        fig.tight_layout()
        compare_path = output_dir / "classifier_comparison.png"
        fig.savefig(compare_path, dpi=120)
        plt.close(fig)
        logger.info("分类器对比图已保存: %s", compare_path)

    logger.info("全部流程完成，结果目录: %s", output_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
