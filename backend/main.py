"""
癌症数据集分类预测主入口。
流程：加载数据 -> 管道化建模(标准化+分类器) -> Optuna 调参 -> 训练最佳模型 -> 评估 -> SHAP 特征解释。
优化版本：保存更多可视化所需数据，支持交互式仪表盘展示。
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import numpy as np
import optuna
import pandas as pd

# 将 backend 根目录加入 path，便于以 python main.py 或模块方式运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.load_data import load_cancer_data
from src.pipeline.model_pipeline import create_pipeline
from src.tuning.optuna_tune import run_study
from src.interpretation.shap_explain import explain_model
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    precision_score, recall_score
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="癌症数据集分类：管道 + Optuna + SHAP")
    parser.add_argument("--classifier", type=str, default="svc", choices=["svc", "random_forest"])
    parser.add_argument("--n_trials", type=int, default=30, help="Optuna 试验次数")
    parser.add_argument("--output-dir", type=str, default="output", help="结果与图表输出目录")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _json_default(obj):
    """JSON序列化辅助函数，处理numpy类型"""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("1. 加载癌症数据集")
    data = load_cancer_data(
        test_size=args.test_size,
        random_state=args.random_state,
    )
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]
    target_names = data["target_names"]

    # 保存原始数据为DataFrame，便于可视化分析
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df["target"] = y_train
    train_df["target_name"] = [target_names[t] for t in y_train]
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df["target"] = y_test
    test_df["target_name"] = [target_names[t] for t in y_test]
    train_df.to_csv(output_dir / "train_data.csv", index=False)
    test_df.to_csv(output_dir / "test_data.csv", index=False)
    logger.info("训练/测试数据已保存为CSV")

    logger.info("2. Optuna 调参")
    best_params, study = run_study(
        X_train,
        y_train,
        classifier=args.classifier,
        n_trials=args.n_trials,
        cv=5,
        scoring="f1_weighted",
        random_state=args.random_state,
    )

    # 保存Optuna调参历史，用于可视化调参过程
    study_trials = []
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trial_data = {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
            }
            study_trials.append(trial_data)
    with open(output_dir / "optuna_history.json", "w", encoding="utf-8") as f:
        json.dump(study_trials, f, indent=2, ensure_ascii=False, default=_json_default)
    logger.info("Optuna调参历史已保存")

    logger.info("3. 使用最佳超参训练管道并评估")
    pipeline = create_pipeline(classifier=args.classifier, **best_params)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)

    # 计算各项评估指标
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred)

    # 计算ROC曲线数据
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    # 计算PR曲线数据
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
    pr_auc = auc(recall_curve, precision_curve)

    logger.info("测试集 Accuracy: %.4f, F1 (weighted): %.4f, AUC: %.4f", accuracy, f1, roc_auc)
    logger.info("分类报告:\n%s", report)

    # 汇总所有指标
    metrics = {
        "classifier": args.classifier,
        "n_trials": args.n_trials,
        "test_size": args.test_size,
        "best_params": best_params,
        "best_cv_value": study.best_value,
        "test_accuracy": accuracy,
        "test_f1_weighted": f1,
        "test_f1_macro": f1_macro,
        "test_precision": precision,
        "test_recall": recall,
        "test_roc_auc": roc_auc,
        "test_pr_auc": pr_auc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "target_names": target_names,
        "feature_names": feature_names,
    }

    metrics_path = output_dir / "metrics.json"
    to_dump = {k: v for k, v in metrics.items() if k != "classification_report"}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(to_dump, f, indent=2, ensure_ascii=False, default=_json_default)
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("指标已保存: %s", metrics_path)

    # 保存预测结果和概率
    predictions_df = test_df.copy()
    predictions_df["predicted"] = y_pred
    predictions_df["predicted_name"] = [target_names[p] for p in y_pred]
    predictions_df["prob_malignant"] = y_pred_proba[:, 1]
    predictions_df["prob_benign"] = y_pred_proba[:, 0]
    predictions_df["correct"] = (y_pred == y_test).astype(int)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    logger.info("预测结果已保存")

    # 保存ROC和PR曲线数据
    curve_data = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_auc": roc_auc,
        "precision_curve": precision_curve.tolist(),
        "recall_curve": recall_curve.tolist(),
        "pr_auc": pr_auc,
    }
    with open(output_dir / "curve_data.json", "w", encoding="utf-8") as f:
        json.dump(curve_data, f, indent=2)

    # 保存训练好的模型
    joblib.dump(pipeline, output_dir / "model.pkl")
    logger.info("模型已保存")

    logger.info("4. SHAP 特征解释")
    shap_values, summary, shap_sample_idx = explain_model(
        pipeline,
        X_test,
        feature_names,
        output_dir=output_dir,
        max_display=30,
        sample_size=100,
    )
    importance_path = output_dir / "feature_importance_order.json"
    with open(importance_path, "w", encoding="utf-8") as f:
        json.dump(summary["importance_order"], f, indent=2, ensure_ascii=False)

    # 保存SHAP值用于交互式可视化
    np.savez_compressed(
        output_dir / "shap_values.npz",
        shap_values=shap_values,
        mean_abs_shap=np.array(summary["mean_abs_shap"]),
        sample_idx=shap_sample_idx,
    )
    logger.info("SHAP值已保存，特征重要性顺序已保存: %s", importance_path)

    logger.info("全部流程完成，结果目录: %s", output_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
