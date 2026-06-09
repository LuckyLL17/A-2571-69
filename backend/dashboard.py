"""
癌症分类结果交互式可视化仪表盘 (Streamlit + Plotly)。

读取 `main.py` 在 `output/` 目录下生成的结构化产物，提供 5 个交互式面板：
1. 概览 (Overview)：核心指标、最佳超参、分类报告
2. 模型评估 (Evaluation)：混淆矩阵、ROC 曲线、PR 曲线（可调阈值）
3. Optuna 调参 (Tuning)：试验值收敛曲线、超参重要性、超参分布
4. SHAP 特征解释 (Interpretation)：全局特征重要性条形图、依赖蜂群散点、单样本解释
5. 预测明细 (Predictions)：可筛选/排序的测试集逐样本预测表

运行方式：
    cd backend
    pip install -r requirements.txt
    python main.py            # 先生成 output/ 下的产物
    streamlit run dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------------
# 页面基础配置
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="癌症分类结果可视化仪表盘",
    page_icon=":bar_chart:",
    layout="wide",
)

# 默认结果目录：与 main.py 输出目录保持一致；允许用户在侧边栏修改
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


# ---------------------------------------------------------------------------
# 数据加载工具：使用 st.cache_data 缓存，仅在文件变化时重新读取
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_json(path: Path) -> dict | None:
    """读取 JSON 文件，文件不存在时返回 None。"""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _load_text(path: Path) -> str | None:
    """读取纯文本文件。"""
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def _load_csv(path: Path) -> pd.DataFrame | None:
    """读取 CSV 文件。"""
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _load_npz(path: Path) -> dict | None:
    """读取 npz，返回字段字典；失败时返回 None。"""
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# 各面板渲染函数
# ---------------------------------------------------------------------------
def render_overview(metrics: dict, report_txt: str | None) -> None:
    """概览：四个核心指标 + 最佳超参 + 分类报告原文。"""
    st.subheader("模型概览")

    # 用 4 列 metric 展示核心数值
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("分类器", metrics.get("classifier", "-"))
    col2.metric("测试集 Accuracy", f"{metrics.get('test_accuracy', 0):.4f}")
    col3.metric("F1 (weighted)", f"{metrics.get('test_f1_weighted', 0):.4f}")
    col4.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.4f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Average Precision", f"{metrics.get('average_precision', 0):.4f}")
    col6.metric("最佳 CV 值", f"{metrics.get('best_cv_value', 0):.4f}")
    col7.metric("训练集样本数", metrics.get("n_train", "-"))
    col8.metric("测试集样本数", metrics.get("n_test", "-"))

    st.markdown("#### 最佳超参 (Best Params)")
    st.json(metrics.get("best_params", {}))

    if report_txt:
        with st.expander("查看完整分类报告 (classification_report)", expanded=False):
            st.code(report_txt, language="text")


def render_evaluation(
    cm_data: dict | None,
    roc_data: dict | None,
    pr_data: dict | None,
    predictions: pd.DataFrame | None,
) -> None:
    """模型评估：混淆矩阵 + ROC + PR + 阈值调节。"""
    st.subheader("模型评估")

    # ---------- 阈值滑动条：动态重算混淆矩阵与对应指标 ----------
    if predictions is not None and {"y_true", "y_proba"}.issubset(predictions.columns):
        threshold = st.slider(
            "分类阈值 (基于正类概率)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="拖动阈值，可实时查看混淆矩阵的变化。",
        )
        y_true = predictions["y_true"].to_numpy()
        y_proba = predictions["y_proba"].to_numpy()
        y_pred_dyn = (y_proba >= threshold).astype(int)
        labels = cm_data["labels"] if cm_data else ["0", "1"]
        cm = confusion_matrix(y_true, y_pred_dyn)
    else:
        threshold = 0.5
        if cm_data is None:
            st.info("缺少混淆矩阵数据，请先运行 main.py。")
            return
        cm = np.array(cm_data["matrix"])
        labels = cm_data["labels"]

    # ---------- 双列布局：左侧混淆矩阵热力图，右侧 ROC ----------
    left, right = st.columns(2)
    with left:
        st.markdown(f"#### 混淆矩阵 (阈值={threshold:.2f})")
        fig_cm = px.imshow(
            cm,
            x=labels,
            y=labels,
            color_continuous_scale="Blues",
            text_auto=True,
            labels=dict(x="预测标签", y="真实标签", color="计数"),
            aspect="auto",
        )
        fig_cm.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    with right:
        st.markdown("#### ROC 曲线")
        if roc_data is None:
            st.info("缺少 roc_curve.json")
        else:
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=roc_data["fpr"], y=roc_data["tpr"],
                mode="lines", name=f"ROC (AUC={roc_data['auc']:.4f})",
            ))
            # 对角参考线
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                name="随机基线", line=dict(dash="dash", color="gray"),
            ))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

    # ---------- PR 曲线 ----------
    st.markdown("#### Precision-Recall 曲线")
    if pr_data is None:
        st.info("缺少 pr_curve.json")
    else:
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=pr_data["recall"], y=pr_data["precision"],
            mode="lines", name=f"PR (AP={pr_data['average_precision']:.4f})",
        ))
        fig_pr.update_layout(
            xaxis_title="Recall", yaxis_title="Precision",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_pr, use_container_width=True)


def render_tuning(trials: pd.DataFrame | None) -> None:
    """Optuna 调参面板：收敛曲线 + 超参与目标值散点。"""
    st.subheader("Optuna 调参过程")
    if trials is None or trials.empty:
        st.info("缺少 optuna_trials.csv，请先运行 main.py。")
        return

    # 仅保留完成的试验，绘制目标值收敛曲线（累积最大）
    completed = trials[trials["state"] == "COMPLETE"].copy()
    completed["best_so_far"] = completed["value"].cummax()

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 试验目标值与最优收敛")
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=completed["number"], y=completed["value"],
            mode="markers", name="单次试验值",
        ))
        fig_conv.add_trace(go.Scatter(
            x=completed["number"], y=completed["best_so_far"],
            mode="lines", name="累计最优",
        ))
        fig_conv.update_layout(
            xaxis_title="试验编号", yaxis_title="CV 目标值",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_conv, use_container_width=True)

    # 用户可以选择某个超参列与目标值绘制散点
    param_cols = [c for c in completed.columns if c.startswith("params_")]
    with col_b:
        st.markdown("#### 超参与目标值关系")
        if not param_cols:
            st.info("未发现 params_* 列。")
        else:
            selected = st.selectbox("选择要查看的超参", param_cols, index=0)
            fig_sc = px.scatter(
                completed, x=selected, y="value",
                hover_data=["number"], trendline=None,
            )
            fig_sc.update_layout(margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_sc, use_container_width=True)

    with st.expander("查看试验明细表", expanded=False):
        st.dataframe(trials, use_container_width=True)


def render_shap(shap_pack: dict | None, feature_names_meta: list[str] | None) -> None:
    """SHAP 解释面板：全局重要性 + 单样本贡献条形图。"""
    st.subheader("SHAP 特征解释")
    if shap_pack is None:
        st.info("缺少 shap_values.npz，请先运行 main.py。")
        return

    feature_names = list(shap_pack.get("feature_names", []))
    if not feature_names and feature_names_meta:
        feature_names = feature_names_meta

    shap_values = np.asarray(shap_pack["shap_values"])
    plot_X = np.asarray(shap_pack["plot_X"])
    # 二维为 (n_samples, n_features)；三维为 (n_samples, n_features, n_classes)，取正类
    if shap_values.ndim == 3:
        shap_values = shap_values[..., 1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    top_n = st.slider("展示前 N 个重要特征", 5, min(30, len(feature_names)), 15)
    top_idx = order[:top_n]
    df_imp = pd.DataFrame({
        "feature": [feature_names[i] for i in top_idx],
        "mean_abs_shap": mean_abs[top_idx],
    })

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### 全局特征重要性 (mean |SHAP|)")
        fig_bar = px.bar(
            df_imp.iloc[::-1],  # 反转使最大值在顶部
            x="mean_abs_shap", y="feature", orientation="h",
        )
        fig_bar.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown("#### SHAP 蜂群散点 (前 N 特征)")
        # 将每个样本×特征的 (shap, feature_value) 展平作图
        rows = []
        for j in top_idx:
            for i in range(shap_values.shape[0]):
                rows.append({
                    "feature": feature_names[j],
                    "shap_value": float(shap_values[i, j]),
                    "feature_value": float(plot_X[i, j]),
                })
        df_sw = pd.DataFrame(rows)
        fig_sw = px.strip(
            df_sw, x="shap_value", y="feature",
            color="feature_value", color_continuous_scale="RdBu_r",
            stripmode="overlay",
        )
        fig_sw.update_traces(jitter=0.4, marker=dict(size=5, opacity=0.7))
        fig_sw.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_sw, use_container_width=True)

    # ---------- 单样本解释 ----------
    st.markdown("#### 单样本 SHAP 贡献")
    sample_idx = st.number_input(
        "选择样本索引",
        min_value=0, max_value=int(shap_values.shape[0]) - 1, value=0, step=1,
    )
    contrib = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_values[sample_idx],
        "feature_value": plot_X[sample_idx],
    })
    contrib["abs_shap"] = contrib["shap_value"].abs()
    contrib = contrib.sort_values("abs_shap", ascending=False).head(top_n)
    contrib = contrib.iloc[::-1]
    fig_one = px.bar(
        contrib, x="shap_value", y="feature", orientation="h",
        color="shap_value", color_continuous_scale="RdBu_r",
        hover_data=["feature_value"],
    )
    fig_one.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_one, use_container_width=True)


def render_predictions(predictions: pd.DataFrame | None) -> None:
    """预测明细：分类筛选 + 错误样本高亮 + 概率分布。"""
    st.subheader("测试集预测明细")
    if predictions is None:
        st.info("缺少 predictions.csv，请先运行 main.py。")
        return

    df = predictions.copy()
    df["is_correct"] = (df["y_true"] == df["y_pred"])

    col1, col2 = st.columns([1, 2])
    with col1:
        view_mode = st.radio(
            "查看范围",
            ["全部", "仅预测正确", "仅预测错误"],
            index=0,
        )
    if view_mode == "仅预测正确":
        df_view = df[df["is_correct"]]
    elif view_mode == "仅预测错误":
        df_view = df[~df["is_correct"]]
    else:
        df_view = df

    with col2:
        # 概率分布直方图，按 y_true 分组
        fig_hist = px.histogram(
            df, x="y_proba", color=df["y_true"].astype(str),
            nbins=30, barmode="overlay", opacity=0.6,
            labels={"color": "y_true"},
        )
        fig_hist.update_layout(
            xaxis_title="正类预测概率", yaxis_title="样本数",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.dataframe(df_view, use_container_width=True, height=420)


# ---------------------------------------------------------------------------
# 入口：侧边栏选择数据源 + 顶部 Tab 切换
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("癌症分类结果可视化仪表盘")
    st.caption("Pipeline + Optuna + SHAP · 交互式探索 (Streamlit + Plotly)")

    # --------- 侧边栏：选择 output 目录 ---------
    with st.sidebar:
        st.header("数据源")
        output_dir_str = st.text_input(
            "结果目录 (output)",
            value=str(DEFAULT_OUTPUT_DIR),
            help="main.py 写入产物的目录。",
        )
        if st.button("重新加载", use_container_width=True):
            st.cache_data.clear()
        st.markdown("---")
        st.markdown(
            "提示：若目录为空，请先在 backend 下执行：\n\n"
            "`python main.py --classifier svc --n_trials 30`"
        )

    output_dir = Path(output_dir_str)
    if not output_dir.exists():
        st.error(f"结果目录不存在：{output_dir}")
        st.stop()

    # --------- 加载所有产物 ---------
    metrics = _load_json(output_dir / "metrics.json") or {}
    report_txt = _load_text(output_dir / "classification_report.txt")
    cm_data = _load_json(output_dir / "confusion_matrix.json")
    roc_data = _load_json(output_dir / "roc_curve.json")
    pr_data = _load_json(output_dir / "pr_curve.json")
    trials = _load_csv(output_dir / "optuna_trials.csv")
    predictions = _load_csv(output_dir / "predictions.csv")
    shap_pack = _load_npz(output_dir / "shap_values.npz")

    # --------- Tab 切换 ---------
    tabs = st.tabs(["概览", "模型评估", "Optuna 调参", "SHAP 解释", "预测明细"])
    with tabs[0]:
        render_overview(metrics, report_txt)
    with tabs[1]:
        render_evaluation(cm_data, roc_data, pr_data, predictions)
    with tabs[2]:
        render_tuning(trials)
    with tabs[3]:
        render_shap(shap_pack, metrics.get("feature_names"))
    with tabs[4]:
        render_predictions(predictions)


if __name__ == "__main__":
    main()
