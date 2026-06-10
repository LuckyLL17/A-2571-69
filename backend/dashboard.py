"""
癌症分类预测 - 交互式可视化仪表盘。
使用 Streamlit 构建，支持：
  - 选择分类器并实时运行 Optuna 调参
  - 查看模型评估指标（Accuracy / F1 / ROC-AUC）
  - 混淆矩阵与 ROC 曲线可视化
  - SHAP 特征解释（摘要图 / 条形图 / 瀑布图 / 依赖图）
  - Optuna 调参历史可视化
  - 多分类器性能对比
  - 数据集概览
"""
import sys
from pathlib import Path

# 将 backend 根目录加入 path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data.load_data import load_cancer_data
from src.pipeline.model_pipeline import create_pipeline, SUPPORTED_CLASSIFIERS
from src.tuning.optuna_tune import run_study
from src.interpretation.shap_explain import explain_model
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)

# ========== 页面配置 ==========
st.set_page_config(
    page_title="癌症分类预测仪表盘",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========== 侧边栏 ==========
st.sidebar.title("🔬 癌症分类预测")
st.sidebar.markdown("---")

# 分类器选择
classifier_options = list(SUPPORTED_CLASSIFIERS)
classifier_display = {
    "svc": "SVC (支持向量机)",
    "random_forest": "Random Forest (随机森林)",
    "logistic_regression": "Logistic Regression (逻辑回归)",
    "knn": "KNN (K近邻)",
    "gradient_boosting": "Gradient Boosting (梯度提升)",
}

selected_classifier = st.sidebar.selectbox(
    "选择分类器",
    classifier_options,
    format_func=lambda x: classifier_display.get(x, x),
)

# Optuna 试验次数
n_trials = st.sidebar.slider(
    "Optuna 调参试验次数",
    min_value=10,
    max_value=100,
    value=30,
    step=10,
    help="更多试验次数可能找到更好的超参，但耗时更长",
)

# 测试集比例
test_size = st.sidebar.slider(
    "测试集比例",
    min_value=0.1,
    max_value=0.4,
    value=0.2,
    step=0.05,
)

# 是否运行多分类器对比
compare_all = st.sidebar.checkbox("运行所有分类器对比", value=False)

# 运行按钮
run_button = st.sidebar.button("🚀 开始训练", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "本系统使用威斯康星乳腺癌数据集，\n"
    "通过 sklearn 管道 + Optuna 调参 + SHAP 解释\n"
    "进行癌症分类预测与可视化分析。"
)

# ========== 数据加载（缓存） ==========
@st.cache_data
def load_data(test_size=0.2):
    """加载数据集，使用 Streamlit 缓存避免重复加载。"""
    return load_cancer_data(test_size=test_size, random_state=42)


# ========== 训练与评估（缓存） ==========
@st.cache_resource
def train_and_evaluate(classifier, n_trials, test_size):
    """
    对指定分类器运行完整训练流程并返回所有结果。
    使用 st.cache_resource 缓存训练好的模型和结果，避免重复计算。
    """
    data = load_data(test_size=test_size)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    feature_names = data["feature_names"]
    target_names = data["target_names"]

    # Optuna 调参
    best_params, study = run_study(
        X_train, y_train,
        classifier=classifier,
        n_trials=n_trials,
        cv=5,
        scoring="f1_weighted",
        random_state=42,
    )

    # 构建最佳管道
    pipeline_params = dict(best_params)
    k_best = pipeline_params.pop("k_best", None)
    pipeline_params.pop("use_feature_selection", None)
    pipeline = create_pipeline(classifier=classifier, k_best=k_best, **pipeline_params)
    pipeline.fit(X_train, y_train)

    # 评估
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    report = classification_report(y_test, y_pred, target_names=target_names)

    # SHAP 解释
    shap_values, shap_summary = explain_model(
        pipeline, X_test, feature_names,
        output_dir=Path("output") / classifier,
        max_display=15,
        sample_size=100,
    )

    return {
        "classifier_name": classifier,
        "pipeline": pipeline,
        "study": study,
        "best_params": best_params,
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "classification_report": report,
        "shap_values": shap_values,
        "shap_summary": shap_summary,
        "feature_names": feature_names,
        "target_names": target_names,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "data": data,
    }


# ========== 主页面 ==========
if run_button or "results" in st.session_state:
    # 如果点击了运行按钮，执行训练
    if run_button:
        with st.spinner("正在训练模型，请稍候..."):
            if compare_all:
                # 运行所有分类器
                all_results = {}
                progress = st.progress(0)
                for i, clf in enumerate(classifier_options):
                    st.toast(f"正在训练 {classifier_display[clf]}...")
                    all_results[clf] = train_and_evaluate(clf, n_trials, test_size)
                    progress.progress((i + 1) / len(classifier_options))
                st.session_state["all_results"] = all_results
                st.session_state["results"] = all_results[selected_classifier]
                st.session_state["compare_mode"] = True
            else:
                st.session_state["results"] = train_and_evaluate(
                    selected_classifier, n_trials, test_size
                )
                st.session_state["compare_mode"] = False
                st.session_state["all_results"] = {selected_classifier: st.session_state["results"]}

    results = st.session_state.get("results")
    if results is None:
        st.warning("请点击左侧「开始训练」按钮运行模型。")
        st.stop()

    compare_mode = st.session_state.get("compare_mode", False)
    all_results = st.session_state.get("all_results", {})

    # ========== 标签页 ==========
    tab_overview, tab_metrics, tab_cm_roc, tab_shap, tab_optuna, tab_compare, tab_data = st.tabs(
        ["📊 总览", "📈 评估指标", "🔲 混淆矩阵 & ROC", "🔍 SHAP 解释", "⚙️ Optuna 调参", "🏆 分类器对比", "📋 数据集"]
    )

    # ---------- 总览标签页 ----------
    with tab_overview:
        st.header("📊 模型总览")

        # 顶部指标卡片
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{results['accuracy']:.4f}",
                     delta=f"{results['accuracy'] - 0.95:.4f}" if results['accuracy'] > 0.95 else None,
                     delta_color="normal")
        col2.metric("F1 (weighted)", f"{results['f1']:.4f}")
        col3.metric("ROC-AUC", f"{results['roc_auc']:.4f}")
        col4.metric("最佳 CV 得分", f"{results['study'].best_value:.4f}")

        st.markdown("---")

        # 最佳超参数
        st.subheader("最佳超参数")
        params_df = pd.DataFrame(
            [{"参数": k, "值": str(v)} for k, v in results["best_params"].items()]
        )
        st.dataframe(params_df, use_container_width=True, hide_index=True)

        # 分类报告
        st.subheader("分类报告")
        st.text(results["classification_report"])

    # ---------- 评估指标标签页 ----------
    with tab_metrics:
        st.header("📈 评估指标详情")

        # 指标仪表盘（使用 plotly gauge）
        fig_gauge = make_subplots(
            rows=1, cols=3,
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
        )
        fig_gauge.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=results["accuracy"],
                title={"text": "Accuracy"},
                gauge={"axis": {"range": [0.8, 1.0]},
                       "bar": {"color": "#4C72B0"},
                       "steps": [
                           {"range": [0.8, 0.9], "color": "#ffe0e0"},
                           {"range": [0.9, 0.95], "color": "#fff3cd"},
                           {"range": [0.95, 1.0], "color": "#d4edda"},
                       ]},
            ),
            row=1, col=1,
        )
        fig_gauge.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=results["f1"],
                title={"text": "F1 (weighted)"},
                gauge={"axis": {"range": [0.8, 1.0]},
                       "bar": {"color": "#55A868"},
                       "steps": [
                           {"range": [0.8, 0.9], "color": "#ffe0e0"},
                           {"range": [0.9, 0.95], "color": "#fff3cd"},
                           {"range": [0.95, 1.0], "color": "#d4edda"},
                       ]},
            ),
            row=1, col=2,
        )
        fig_gauge.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=results["roc_auc"],
                title={"text": "ROC-AUC"},
                gauge={"axis": {"range": [0.8, 1.0]},
                       "bar": {"color": "#C44E52"},
                       "steps": [
                           {"range": [0.8, 0.9], "color": "#ffe0e0"},
                           {"range": [0.9, 0.95], "color": "#fff3cd"},
                           {"range": [0.95, 1.0], "color": "#d4edda"},
                       ]},
            ),
            row=1, col=3,
        )
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # 预测分布
        st.subheader("预测分布")
        col_pred1, col_pred2 = st.columns(2)
        with col_pred1:
            # 真实标签分布
            unique, counts = np.unique(results["y_test"], return_counts=True)
            fig_true = px.pie(
                values=counts,
                names=[results["target_names"][int(u)] for u in unique],
                title="真实标签分布",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig_true, use_container_width=True)
        with col_pred2:
            # 预测标签分布
            unique_pred, counts_pred = np.unique(results["y_pred"], return_counts=True)
            fig_pred = px.pie(
                values=counts_pred,
                names=[results["target_names"][int(u)] for u in unique_pred],
                title="预测标签分布",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig_pred, use_container_width=True)

    # ---------- 混淆矩阵 & ROC 标签页 ----------
    with tab_cm_roc:
        st.header("🔲 混淆矩阵 & ROC 曲线")

        col_cm, col_roc = st.columns(2)

        with col_cm:
            st.subheader("混淆矩阵")
            cm = results["confusion_matrix"]
            target_names = results["target_names"]

            # 使用 plotly 绘制交互式混淆矩阵热力图
            fig_cm = px.imshow(
                cm,
                x=target_names,
                y=target_names,
                text_auto=True,
                color_continuous_scale="Blues",
                labels={"x": "预测标签", "y": "真实标签"},
                title="混淆矩阵",
            )
            fig_cm.update_layout(height=450)
            st.plotly_chart(fig_cm, use_container_width=True)

            # 混淆矩阵解读
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            st.markdown(f"""
            **解读：**
            - 真阴性 (TN): {tn} — 正确预测为良性
            - 假阳性 (FP): {fp} — 误判为恶性
            - 假阴性 (FN): {fn} — 误判为良性 ⚠️
            - 真阳性 (TP): {tp} — 正确预测为恶性
            """)

        with col_roc:
            st.subheader("ROC 曲线")
            fig_roc = go.Figure()
            fig_roc.add_trace(
                go.Scatter(
                    x=results["fpr"],
                    y=results["tpr"],
                    mode="lines",
                    name=f"ROC (AUC = {results['roc_auc']:.4f})",
                    line=dict(color="darkorange", width=2),
                )
            )
            fig_roc.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines",
                    name="随机基线",
                    line=dict(color="navy", width=1, dash="dash"),
                )
            )
            fig_roc.update_layout(
                title="ROC 曲线",
                xaxis_title="假正率 (FPR)",
                yaxis_title="真正率 (TPR)",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1.05]),
                height=450,
            )
            st.plotly_chart(fig_roc, use_container_width=True)

    # ---------- SHAP 解释标签页 ----------
    with tab_shap:
        st.header("🔍 SHAP 特征解释")

        shap_summary = results["shap_summary"]
        feature_names = shap_summary["feature_names"]
        mean_abs_shap = shap_summary["mean_abs_shap"]
        # 从 results 中获取实际训练的分类器名称，用于定位 SHAP 图片输出目录
        trained_classifier = results.get("classifier_name", selected_classifier)

        # 特征重要性条形图（交互式）
        st.subheader("全局特征重要性 (平均 |SHAP|)")
        importance_df = pd.DataFrame({
            "特征": feature_names,
            "平均 |SHAP 值|": mean_abs_shap,
        }).sort_values("平均 |SHAP 值|", ascending=True)

        fig_importance = px.bar(
            importance_df.tail(20),
            x="平均 |SHAP 值|",
            y="特征",
            orientation="h",
            title="Top 20 特征重要性",
            color="平均 |SHAP 值|",
            color_continuous_scale="Viridis",
        )
        fig_importance.update_layout(height=600)
        st.plotly_chart(fig_importance, use_container_width=True)

        # SHAP 摘要图（蜂群图）- 使用保存的图片
        st.subheader("SHAP 摘要图（蜂群图）")
        shap_summary_path = Path("output") / trained_classifier / "shap_summary.png"
        if shap_summary_path.exists():
            st.image(str(shap_summary_path), use_column_width=True)
        else:
            st.info("SHAP 摘要图未找到，请先运行训练。")

        # SHAP 条形图
        st.subheader("SHAP 条形图")
        shap_bar_path = Path("output") / trained_classifier / "shap_bar.png"
        if shap_bar_path.exists():
            st.image(str(shap_bar_path), use_column_width=True)

        # SHAP 瀑布图
        st.subheader("SHAP 瀑布图（单样本解释）")
        shap_waterfall_path = Path("output") / trained_classifier / "shap_waterfall.png"
        if shap_waterfall_path.exists():
            st.image(str(shap_waterfall_path), use_column_width=True)

        # SHAP 依赖图
        st.subheader("SHAP 依赖图")
        importance_order = shap_summary.get("importance_order", [])
        for feat_name in importance_order[:3]:
            dep_path = Path("output") / trained_classifier / f"shap_dependence_{feat_name.replace(' ', '_')}.png"
            if dep_path.exists():
                st.image(str(dep_path), caption=f"依赖图: {feat_name}", use_column_width=True)

        # 特征重要性排序表
        st.subheader("特征重要性排序")
        order_df = pd.DataFrame({
            "排名": range(1, len(importance_order) + 1),
            "特征名称": importance_order,
        })
        st.dataframe(order_df, use_container_width=True, hide_index=True)

    # ---------- Optuna 调参标签页 ----------
    with tab_optuna:
        st.header("⚙️ Optuna 调参历史")

        study = results["study"]

        # 优化历史图
        st.subheader("优化历史")
        trials_df = study.trials_dataframe()
        fig_history = go.Figure()
        fig_history.add_trace(
            go.Scatter(
                x=trials_df["number"],
                y=trials_df["value"],
                mode="lines+markers",
                name="CV F1 得分",
                line=dict(color="#4C72B0"),
            )
        )
        # 标记最佳试验
        best_trial = study.best_trial
        fig_history.add_trace(
            go.Scatter(
                x=[best_trial.number],
                y=[best_trial.value],
                mode="markers",
                name="最佳试验",
                marker=dict(color="red", size=12, symbol="star"),
            )
        )
        fig_history.update_layout(
            title="Optuna 优化历史",
            xaxis_title="试验序号",
            yaxis_title="CV F1 (weighted)",
            height=400,
        )
        st.plotly_chart(fig_history, use_container_width=True)

        # 超参数重要性图
        st.subheader("超参数重要性")
        try:
            importance = optuna.importance.get_param_importances(study)
            imp_df = pd.DataFrame({
                "超参数": list(importance.keys()),
                "重要性": list(importance.values()),
            })
            fig_imp = px.bar(
                imp_df,
                x="重要性",
                y="超参数",
                orientation="h",
                title="超参数重要性排名",
                color="重要性",
                color_continuous_scale="Blues",
            )
            fig_imp.update_layout(height=400)
            st.plotly_chart(fig_imp, use_container_width=True)
        except Exception as e:
            st.warning(f"无法计算超参数重要性: {e}")

        # 所有试验详情表
        st.subheader("试验详情")
        display_cols = [c for c in trials_df.columns if c not in ("state", "datetime_start", "datetime_complete")]
        st.dataframe(trials_df[display_cols], use_container_width=True, hide_index=True)

    # ---------- 分类器对比标签页 ----------
    with tab_compare:
        st.header("🏆 分类器对比")

        if len(all_results) <= 1 and not compare_mode:
            st.warning("请在侧边栏勾选「运行所有分类器对比」，然后重新训练以查看对比结果。")
        else:
            # 对比指标表格
            comparison_data = []
            for clf_name, res in all_results.items():
                comparison_data.append({
                    "分类器": classifier_display.get(clf_name, clf_name),
                    "Accuracy": res["accuracy"],
                    "F1 (weighted)": res["f1"],
                    "ROC-AUC": res["roc_auc"],
                    "最佳 CV 得分": res["study"].best_value,
                })
            comp_df = pd.DataFrame(comparison_data)

            # 高亮最佳值
            st.dataframe(
                comp_df.style.highlight_max(
                    subset=["Accuracy", "F1 (weighted)", "ROC-AUC", "最佳 CV 得分"],
                    color="lightgreen",
                ),
                use_container_width=True,
                hide_index=True,
            )

            # 交互式对比柱状图
            st.subheader("性能对比图")
            metric_select = st.selectbox(
                "选择对比指标",
                ["Accuracy", "F1 (weighted)", "ROC-AUC", "最佳 CV 得分"],
            )
            fig_comp = px.bar(
                comp_df,
                x="分类器",
                y=metric_select,
                color="分类器",
                title=f"{metric_select} 对比",
                text=metric_select,
            )
            fig_comp.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_comp.update_layout(
                yaxis=dict(range=[min(comp_df[metric_select]) - 0.05, 1.0]),
                height=450,
                showlegend=False,
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            # 多指标雷达图
            st.subheader("雷达图对比")
            fig_radar = go.Figure()
            categories = ["Accuracy", "F1 (weighted)", "ROC-AUC"]
            colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
            for i, (clf_name, res) in enumerate(all_results.items()):
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=[res["accuracy"], res["f1"], res["roc_auc"]],
                        theta=categories,
                        fill="toself",
                        name=classifier_display.get(clf_name, clf_name),
                        opacity=0.6,
                        line=dict(color=colors[i % len(colors)]),
                    )
                )
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0.8, 1.0])),
                showlegend=True,
                height=500,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # 多分类器 ROC 曲线对比
            st.subheader("ROC 曲线对比")
            fig_roc_all = go.Figure()
            for i, (clf_name, res) in enumerate(all_results.items()):
                fig_roc_all.add_trace(
                    go.Scatter(
                        x=res["fpr"],
                        y=res["tpr"],
                        mode="lines",
                        name=f"{classifier_display.get(clf_name, clf_name)} (AUC={res['roc_auc']:.4f})",
                        line=dict(color=colors[i % len(colors)], width=2),
                    )
                )
            fig_roc_all.add_trace(
                go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines",
                    name="随机基线",
                    line=dict(color="gray", width=1, dash="dash"),
                )
            )
            fig_roc_all.update_layout(
                title="ROC 曲线对比",
                xaxis_title="假正率 (FPR)",
                yaxis_title="真正率 (TPR)",
                height=500,
            )
            st.plotly_chart(fig_roc_all, use_container_width=True)

    # ---------- 数据集标签页 ----------
    with tab_data:
        st.header("📋 数据集概览")

        data = results["data"]
        X_train = data["X_train"]
        X_test = data["X_test"]
        feature_names = data["feature_names"]
        target_names = data["target_names"]

        col_d1, col_d2, col_d3 = st.columns(3)
        col_d1.metric("总样本数", f"{len(X_train) + len(X_test)}")
        col_d2.metric("训练集", f"{len(X_train)}")
        col_d3.metric("测试集", f"{len(X_test)}")

        st.markdown(f"**特征数**: {len(feature_names)}")
        st.markdown(f"**目标类别**: {', '.join(target_names)}")

        # 特征统计
        st.subheader("特征统计")
        df_full = pd.DataFrame(
            np.vstack([X_train, X_test]),
            columns=feature_names,
        )
        st.dataframe(df_full.describe().round(4), use_container_width=True)

        # 特征分布（可选择特征查看）
        st.subheader("特征分布")
        selected_feature = st.selectbox("选择特征", feature_names)
        feat_idx = feature_names.index(selected_feature)
        fig_dist = px.histogram(
            df_full,
            x=selected_feature,
            nbins=30,
            title=f"{selected_feature} 分布",
            color_discrete_sequence=["#4C72B0"],
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # 类别分布
        st.subheader("类别分布")
        y_all = np.concatenate([data["y_train"], data["y_test"]])
        unique, counts = np.unique(y_all, return_counts=True)
        fig_class = px.pie(
            values=counts,
            names=[target_names[int(u)] for u in unique],
            title="类别分布",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig_class, use_container_width=True)

else:
    # 未运行时的欢迎页面
    st.title("🔬 癌症分类预测可视化仪表盘")
    st.markdown("---")
    st.markdown("""
    ### 欢迎使用癌症分类预测系统！

    本系统基于 **威斯康星乳腺癌数据集**，提供以下功能：

    | 功能 | 说明 |
    |------|------|
    | 🤖 多分类器支持 | SVC、随机森林、逻辑回归、KNN、梯度提升 |
    | ⚙️ Optuna 自动调参 | 贝叶斯优化搜索最佳超参数 |
    | 📈 全面评估指标 | Accuracy、F1、ROC-AUC、混淆矩阵 |
    | 🔍 SHAP 特征解释 | 全局/局部特征重要性、依赖图 |
    | 🏆 分类器对比 | 多模型性能横向对比 |

    **使用方法：**
    1. 在左侧边栏选择分类器和参数
    2. 点击 🚀「开始训练」按钮
    3. 在各标签页中查看分析结果
    """)
