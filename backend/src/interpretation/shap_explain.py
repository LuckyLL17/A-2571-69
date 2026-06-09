"""
使用 SHAP 对训练后的管道模型进行特征解释。
根据管道内最终估计器类型选择 TreeExplainer 或 KernelExplainer，
输出摘要图、条形图、瀑布图、依赖图与特征重要性。
优化点：增加瀑布图和依赖图，返回更丰富的解释数据供仪表盘使用。
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

    # 降低 SHAP 内部日志级别，避免刷屏
    _shap_log = logging.getLogger("shap")
    _shap_log.setLevel(logging.WARNING)

    clf = pipeline.named_steps["classifier"]
    clf_name = type(clf).__name__

    # 对大样本进行随机抽样以加速解释
    if sample_size is not None and len(X) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_explain = X[idx]
    else:
        X_explain = X

    # 使用管道中 scaler（和可选的 select_kbest）对数据进行变换
    X_scaled = pipeline[:-1].transform(X_explain)

    # 如果管道中有特征选择步骤，需要更新 feature_names
    if "select_kbest" in pipeline.named_steps:
        selector = pipeline.named_steps["select_kbest"]
        selected_mask = selector.get_support()
        feature_names = [feature_names[i] for i, selected in enumerate(selected_mask) if selected]

    try:
        if "Tree" in clf_name or "Forest" in clf_name or "GradientBoosting" in clf_name:
            # 树模型使用 TreeExplainer，速度更快
            explainer = shap.TreeExplainer(clf, X_scaled)
            shap_values = explainer.shap_values(X_scaled)
            # 二分类时 shap_values 返回列表，取正类（malignant）的 SHAP 值
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            plot_X = X_scaled
        else:
            # SVC / LogisticRegression / KNN 等非树模型：使用 KernelExplainer
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

    # 计算每个特征的平均绝对 SHAP 值（全局特征重要性）
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    if mean_abs_shap.ndim > 1:
        mean_abs_shap = mean_abs_shap.mean(axis=0)
    order = np.argsort(mean_abs_shap)[::-1]

    summary = {
        "feature_names": feature_names,
        "mean_abs_shap": np.asarray(mean_abs_shap).ravel().tolist(),
        "importance_order": [feature_names[int(i)] for i in order.ravel()],
        "shap_values_sample": np.asarray(shap_values[:20]).tolist(),  # 保存前 20 个样本的 SHAP 值供仪表盘使用
        "X_scaled_sample": np.asarray(plot_X[:20]).tolist(),  # 保存对应的标准化特征值
    }

    import matplotlib.pyplot as plt

    # 1. SHAP 摘要图（蜂群图）
    shap.summary_plot(
        shap_values,
        plot_X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    summary_path = output_dir / "shap_summary.png"
    plt.savefig(summary_path, bbox_inches="tight", dpi=120)
    plt.close()
    logger.info("SHAP 摘要图已保存: %s", summary_path)

    # 2. SHAP 条形图（全局特征重要性）
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

    # 3. SHAP 瀑布图（展示单个样本的特征贡献）
    try:
        plt.figure()
        # 使用 shap.Explanation 对象绘制瀑布图
        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
            data=plot_X[0],
            feature_names=feature_names,
        )
        shap.waterfall_plot(explanation, max_display=max_display, show=False)
        waterfall_path = output_dir / "shap_waterfall.png"
        plt.savefig(waterfall_path, bbox_inches="tight", dpi=120)
        plt.close()
        logger.info("SHAP 瀑布图已保存: %s", waterfall_path)
    except Exception as e:
        logger.warning("瀑布图生成失败: %s", str(e))
        plt.close()

    # 4. SHAP 依赖图（对最重要的前 3 个特征绘制）
    try:
        top_features = order[:3]
        for rank, feat_idx in enumerate(top_features):
            plt.figure()
            shap.dependence_plot(
                int(feat_idx),
                shap_values,
                plot_X,
                feature_names=feature_names,
                show=False,
            )
            dep_path = output_dir / f"shap_dependence_{feature_names[int(feat_idx)].replace(' ', '_')}.png"
            plt.savefig(dep_path, bbox_inches="tight", dpi=120)
            plt.close()
            logger.info("SHAP 依赖图已保存: %s", dep_path)
    except Exception as e:
        logger.warning("依赖图生成失败: %s", str(e))
        plt.close()

    return shap_values, summary
