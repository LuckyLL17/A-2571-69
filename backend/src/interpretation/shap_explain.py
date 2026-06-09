"""
使用 SHAP 对训练后的管道模型进行特征解释。
根据管道内最终估计器类型选择 TreeExplainer 或 KernelExplainer，并输出摘要图与特征重要性。
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import shap
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def explain_model(
    pipeline: Pipeline,
    X: np.ndarray,
    feature_names: list[str],
    output_dir: str | Path = "output",
    max_display: int = 15,
    sample_size: int | None = 100,
) -> tuple[np.ndarray, dict]:
    """
    对管道模型做 SHAP 解释，并保存图表与汇总统计。

    :param pipeline: 已 fit 的 Pipeline（含 StandardScaler + 分类器）
    :param X: 用于解释的样本（建议用训练集或测试集子集）
    :param feature_names: 特征名列表
    :param output_dir: 图表与结果输出目录
    :param max_display: 摘要图/条形图展示的最大特征数
    :param sample_size: 若提供，则对 X 抽样以加速 KernelExplainer；None 表示不抽样
    :return: (shap_values 数组, 汇总统计 dict)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _shap_log = logging.getLogger("shap")
    _shap_log.setLevel(logging.WARNING)

    clf = pipeline.named_steps["classifier"]
    clf_name = type(clf).__name__

    if sample_size is not None and len(X) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_explain = X[idx]
    else:
        X_explain = X

    X_scaled = pipeline[:-1].transform(X_explain)

    try:
        if "Tree" in clf_name or "Forest" in clf_name:
            explainer = shap.TreeExplainer(clf, X_scaled)
            shap_values = explainer.shap_values(X_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            plot_X = X_scaled
        else:
            # SVC 等非树模型：使用标准化后的数据做解释，与分类器实际输入空间一致
            background_scaled = shap.sample(X_scaled, min(50, len(X_scaled)))
            explainer = shap.KernelExplainer(clf.predict_proba, background_scaled)
            shap_values = explainer.shap_values(X_scaled, nsamples=min(100, len(X_scaled) * 2))
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            plot_X = X_scaled
    except Exception as e:
        logger.warning(
            "SHAP 首选解释器不可用，回退到 KernelExplainer。异常类型: %s，信息: %s",
            type(e).__name__,
            str(e),
            exc_info=True,
        )
        try:
            background_scaled = shap.sample(X_scaled, min(50, len(X_scaled)))
            explainer = shap.KernelExplainer(clf.predict_proba, background_scaled)
            shap_values = explainer.shap_values(X_scaled, nsamples=min(80, len(X_scaled) * 2))
        except Exception as fallback_e:
            logger.exception(
                "KernelExplainer 回退失败。异常类型: %s，信息: %s",
                type(fallback_e).__name__,
                str(fallback_e),
            )
            raise RuntimeError(
                f"SHAP 解释失败: 首选 {type(e).__name__}({e}); 回退 {type(fallback_e).__name__}({fallback_e})"
            ) from fallback_e
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        plot_X = X_scaled

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    if mean_abs_shap.ndim > 1:
        mean_abs_shap = mean_abs_shap.mean(axis=0)
    # 确保是一维数组
    mean_abs_shap = np.asarray(mean_abs_shap).ravel()
    order = np.argsort(mean_abs_shap)[::-1]
    # 确保order是整数类型的一维数组
    order = np.asarray(order, dtype=int).ravel()
    summary = {
        "feature_names": feature_names,
        "mean_abs_shap": mean_abs_shap.tolist(),
        "importance_order": [feature_names[int(i)] for i in order],
    }

    shap.summary_plot(
        shap_values,
        plot_X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    summary_path = output_dir / "shap_summary.png"
    import matplotlib.pyplot as plt
    plt.savefig(summary_path, bbox_inches="tight", dpi=120)
    plt.close()
    logger.info("SHAP 摘要图已保存: %s", summary_path)

    shap.summary_plot(
        shap_values,
        plot_X,
        feature_names=feature_names,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    bar_path = output_dir / "shap_bar.png"
    plt.savefig(bar_path, bbox_inches="tight", dpi=120)
    plt.close()
    logger.info("SHAP 条形图已保存: %s", bar_path)

    return shap_values, summary
