"""
癌症数据集分类预测主入口。
流程：加载数据 -> 管道化建模(标准化+分类器) -> Optuna 调参 -> 训练最佳模型 -> 评估 -> SHAP 特征解释。
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# 将 backend 根目录加入 path，便于以 python main.py 或模块方式运行
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.load_data import load_cancer_data
from src.pipeline.model_pipeline import create_pipeline
from src.tuning.optuna_tune import run_study
from src.interpretation.shap_explain import explain_model
from sklearn.metrics import accuracy_score, f1_score, classification_report

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

    logger.info("3. 使用最佳超参训练管道并评估")
    pipeline = create_pipeline(classifier=args.classifier, **best_params)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred, target_names=target_names)

    logger.info("测试集 Accuracy: %.4f, F1 (weighted): %.4f", accuracy, f1)
    logger.info("分类报告:\n%s", report)

    metrics = {
        "best_params": best_params,
        "best_cv_value": study.best_value,
        "test_accuracy": accuracy,
        "test_f1_weighted": f1,
        "classification_report": report,
    }
    def _json_default(obj):
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        raise TypeError(type(obj).__name__)

    metrics_path = output_dir / "metrics.json"
    to_dump = {k: v for k, v in metrics.items() if k != "classification_report"}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(to_dump, f, indent=2, ensure_ascii=False, default=_json_default)
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    logger.info("指标已保存: %s", metrics_path)

    logger.info("4. SHAP 特征解释")
    shap_values, summary = explain_model(
        pipeline,
        X_test,
        feature_names,
        output_dir=output_dir,
        max_display=15,
        sample_size=100,
    )
    importance_path = output_dir / "feature_importance_order.json"
    with open(importance_path, "w", encoding="utf-8") as f:
        json.dump(summary["importance_order"], f, indent=2, ensure_ascii=False)
    logger.info("特征重要性顺序已保存: %s", importance_path)

    logger.info("全部流程完成，结果目录: %s", output_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
