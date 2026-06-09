"""
乳腺癌分类预测结果 - 交互式可视化仪表盘
使用Streamlit构建，包含模型性能评估、SHAP特征解释、特征分布分析等模块
运行方式: streamlit run dashboard.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="乳腺癌分类预测仪表盘",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 数据加载函数 ====================
@st.cache_data
def load_results(output_dir: str = "output"):
    """
    加载所有模型结果数据，使用Streamlit缓存提高性能
    :param output_dir: 结果目录路径
    :return: 包含所有数据的字典
    """
    output_path = Path(output_dir)
    
    # 加载指标数据
    with open(output_path / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    
    # 加载预测结果
    predictions_df = pd.read_csv(output_path / "predictions.csv")
    
    # 加载训练/测试数据
    train_df = pd.read_csv(output_path / "train_data.csv")
    test_df = pd.read_csv(output_path / "test_data.csv")
    
    # 加载Optuna历史
    with open(output_path / "optuna_history.json", "r", encoding="utf-8") as f:
        optuna_history = json.load(f)
    
    # 加载曲线数据
    with open(output_path / "curve_data.json", "r", encoding="utf-8") as f:
        curve_data = json.load(f)
    
    # 加载特征重要性
    with open(output_path / "feature_importance_order.json", "r", encoding="utf-8") as f:
        feature_importance = json.load(f)
    
    # 加载SHAP值
    shap_data = np.load(output_path / "shap_values.npz")
    shap_values = shap_data["shap_values"]
    mean_abs_shap = shap_data["mean_abs_shap"]
    
    # 加载模型
    model = joblib.load(output_path / "model.pkl")
    
    return {
        "metrics": metrics,
        "predictions_df": predictions_df,
        "train_df": train_df,
        "test_df": test_df,
        "optuna_history": optuna_history,
        "curve_data": curve_data,
        "feature_importance": feature_importance,
        "shap_values": shap_values,
        "mean_abs_shap": mean_abs_shap,
        "model": model,
        "feature_names": metrics["feature_names"],
        "target_names": metrics["target_names"],
    }


# ==================== 侧边栏配置 ====================
with st.sidebar:
    st.title("⚙️ 控制面板")
    output_dir = st.text_input("结果目录", value="output")
    st.markdown("---")
    st.markdown("### 📁 导航")
    page = st.radio(
        "选择页面",
        ["📈 模型概览", "🔍 模型评估", "✨ SHAP解释", "📊 特征分析", "🔮 单样本预测"],
        index=0
    )
    st.markdown("---")
    st.markdown("### ℹ️ 关于")
    st.info("""
    **乳腺癌分类预测仪表盘**
    
    使用 sklearn + Optuna + SHAP 构建的机器学习模型，
    通过 Streamlit 实现交互式可视化展示。
    """)

# 加载数据
try:
    data = load_results(output_dir)
except FileNotFoundError:
    st.error(f"❌ 未找到结果文件，请先运行 `python main.py` 生成结果！")
    st.stop()

# 提取常用变量
metrics = data["metrics"]
predictions_df = data["predictions_df"]
train_df = data["train_df"]
test_df = data["test_df"]
feature_names = data["feature_names"]
target_names = data["target_names"]

# ==================== 页面标题 ====================
st.title("🩺 乳腺癌分类预测 - 结果可视化仪表盘")
st.markdown(f"**分类器**: `{metrics['classifier']}` | **Optuna试验次数**: {metrics['n_trials']} | **测试集比例**: {metrics['test_size']:.0%}")
st.markdown("---")

# ==================== 页面1: 模型概览 ====================
if page == "📈 模型概览":
    st.header("📈 模型性能概览")
    
    # 第一行：关键指标卡片
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="🎯 准确率 (Accuracy)",
            value=f"{metrics['test_accuracy']:.4f}",
            delta=f"{metrics['test_accuracy'] - 0.9:.2%}" if metrics['test_accuracy'] > 0.9 else None
        )
    
    with col2:
        st.metric(
            label="🔢 F1分数 (加权)",
            value=f"{metrics['test_f1_weighted']:.4f}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="⚡ 精确率 (Precision)",
            value=f"{metrics['test_precision']:.4f}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="📞 召回率 (Recall)",
            value=f"{metrics['test_recall']:.4f}",
            delta=None
        )
    
    with col5:
        st.metric(
            label="📊 ROC AUC",
            value=f"{metrics['test_roc_auc']:.4f}",
            delta=None
        )
    
    st.markdown("---")
    
    # 第二行：最佳超参数和数据集统计
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🏆 最佳超参数")
        best_params_df = pd.DataFrame(
            list(metrics["best_params"].items()),
            columns=["参数名", "参数值"]
        )
        st.dataframe(best_params_df, use_container_width=True, hide_index=True)
        
        st.metric(
            label="最佳交叉验证F1分数",
            value=f"{metrics['best_cv_value']:.4f}"
        )
    
    with col2:
        st.subheader("📊 数据集统计")
        dataset_stats = pd.DataFrame({
            "数据集": ["训练集", "测试集", "总计"],
            "样本数": [len(train_df), len(test_df), len(train_df) + len(test_df)],
            "良性(benign)": [
                (train_df["target"] == 0).sum(),
                (test_df["target"] == 0).sum(),
                (train_df["target"] == 0).sum() + (test_df["target"] == 0).sum()
            ],
            "恶性(malignant)": [
                (train_df["target"] == 1).sum(),
                (test_df["target"] == 1).sum(),
                (train_df["target"] == 1).sum() + (test_df["target"] == 1).sum()
            ],
        })
        st.dataframe(dataset_stats, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # 第三行：Optuna调参过程可视化
    st.subheader("🔄 Optuna超参数优化过程")
    optuna_df = pd.DataFrame(data["optuna_history"])
    optuna_df["best_value_so_far"] = optuna_df["value"].cummax()
    
    fig_optuna = go.Figure()
    fig_optuna.add_trace(go.Scatter(
        x=optuna_df["number"],
        y=optuna_df["value"],
        mode="markers",
        name="单次试验F1",
        marker=dict(size=10, color="lightblue"),
        hovertemplate="试验 #%{x}<br>F1: %{y:.4f}<extra></extra>"
    ))
    fig_optuna.add_trace(go.Scatter(
        x=optuna_df["number"],
        y=optuna_df["best_value_so_far"],
        mode="lines+markers",
        name="当前最佳",
        line=dict(color="red", width=3),
        marker=dict(size=8),
        hovertemplate="试验 #%{x}<br>最佳F1: %{y:.4f}<extra></extra>"
    ))
    fig_optuna.update_layout(
        title="Optuna调参历史 (目标: 最大化加权F1)",
        xaxis_title="试验编号",
        yaxis_title="交叉验证F1分数",
        hovermode="x unified",
        height=450
    )
    st.plotly_chart(fig_optuna, use_container_width=True)

# ==================== 页面2: 模型评估 ====================
elif page == "🔍 模型评估":
    st.header("🔍 模型详细评估")
    
    # 第一行：混淆矩阵和分类报告
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 混淆矩阵")
        cm = np.array(metrics["confusion_matrix"])
        
        # 使用Plotly绘制交互式混淆矩阵
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=[f"预测<br>{name}" for name in target_names],
            y=[f"真实<br>{name}" for name in target_names],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 20},
            colorscale="Blues",
            hoverongaps=False,
            hovertemplate="真实: %{y}<br>预测: %{x}<br>数量: %{z}<extra></extra>"
        ))
        fig_cm.update_layout(
            title="混淆矩阵热力图",
            xaxis_title="预测标签",
            yaxis_title="真实标签",
            height=450
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        st.subheader("📋 分类报告")
        with st.expander("查看详细分类报告", expanded=True):
            st.text(metrics["classification_report"])
    
    st.markdown("---")
    
    # 第二行：ROC曲线和PR曲线
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📈 ROC曲线")
        curve_data = data["curve_data"]
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=curve_data["fpr"],
            y=curve_data["tpr"],
            mode="lines",
            name=f"ROC曲线 (AUC = {curve_data['roc_auc']:.4f})",
            line=dict(color="darkorange", width=3),
            fill="tozeroy",
            fillcolor="rgba(255, 165, 0, 0.1)"
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="随机分类器",
            line=dict(color="navy", width=2, dash="dash")
        ))
        fig_roc.update_layout(
            title=f"受试者工作特征曲线 (ROC AUC = {curve_data['roc_auc']:.4f})",
            xaxis_title="假阳性率 (FPR)",
            yaxis_title="真阳性率 (TPR)",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05]),
            height=450,
            legend=dict(x=0.6, y=0.1)
        )
        st.plotly_chart(fig_roc, use_container_width=True)
    
    with col2:
        st.subheader("📉 PR曲线 (精确率-召回率曲线)")
        
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=curve_data["recall_curve"],
            y=curve_data["precision_curve"],
            mode="lines",
            name=f"PR曲线 (AUC = {curve_data['pr_auc']:.4f})",
            line=dict(color="green", width=3),
            fill="tozeroy",
            fillcolor="rgba(0, 128, 0, 0.1)"
        ))
        fig_pr.update_layout(
            title=f"精确率-召回率曲线 (PR AUC = {curve_data['pr_auc']:.4f})",
            xaxis_title="召回率 (Recall)",
            yaxis_title="精确率 (Precision)",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05]),
            height=450,
            legend=dict(x=0.6, y=0.1)
        )
        st.plotly_chart(fig_pr, use_container_width=True)
    
    st.markdown("---")
    
    # 第三行：预测结果分布
    st.subheader("🎯 预测概率分布分析")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 按真实标签分组的预测概率分布
        fig_prob = go.Figure()
        for target in [0, 1]:
            subset = predictions_df[predictions_df["target"] == target]
            fig_prob.add_trace(go.Histogram(
                x=subset["prob_malignant"],
                name=f"真实: {target_names[target]}",
                opacity=0.7,
                nbinsx=30,
                hovertemplate="概率区间: %{x:.2f}<br>样本数: %{y}<extra></extra>"
            ))
        fig_prob.add_vline(
            x=0.5, line_dash="dash", line_color="red",
            annotation_text="分类阈值 (0.5)", annotation_position="top"
        )
        fig_prob.update_layout(
            title="预测为恶性的概率分布",
            xaxis_title="预测为恶性的概率",
            yaxis_title="样本数",
            barmode="overlay",
            height=400
        )
        st.plotly_chart(fig_prob, use_container_width=True)
    
    with col2:
        # 预测正确/错误统计
        correct_counts = predictions_df["correct"].value_counts()
        labels = ["预测正确", "预测错误"]
        values = [correct_counts.get(1, 0), correct_counts.get(0, 0)]
        colors = ["#2ecc71", "#e74c3c"]
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            hole=0.5,
            textinfo="label+percent+value",
            hovertemplate="%{label}<br>数量: %{value}<br>占比: %{percent}<extra></extra>"
        )])
        fig_pie.update_layout(
            title="预测准确率分布",
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # 第四行：预测样本详情表格
    st.subheader("📋 预测样本详情")
    with st.expander("展开查看所有测试集预测结果", expanded=False):
        display_cols = ["target_name", "predicted_name", "prob_benign", "prob_malignant", "correct"]
        st.dataframe(
            predictions_df[display_cols].rename(columns={
                "target_name": "真实标签",
                "predicted_name": "预测标签",
                "prob_benign": "良性概率",
                "prob_malignant": "恶性概率",
                "correct": "预测正确"
            }),
            use_container_width=True
        )

# ==================== 页面3: SHAP特征解释 ====================
elif page == "✨ SHAP解释":
    st.header("✨ SHAP模型可解释性分析")
    
    shap_values = data["shap_values"]
    mean_abs_shap = data["mean_abs_shap"]
    
    # 特征重要性滑块
    top_n = st.slider("显示前N个重要特征", min_value=5, max_value=len(feature_names), value=15, step=1)
    
    # 准备SHAP数据
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=True).tail(top_n)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 特征重要性条形图")
        fig_bar = go.Figure(go.Bar(
            x=shap_df["mean_abs_shap"],
            y=shap_df["feature"],
            orientation="h",
            marker=dict(
                color=shap_df["mean_abs_shap"],
                colorscale="Viridis",
                showscale=True
            ),
            hovertemplate="特征: %{y}<br>平均|SHAP值|: %{x:.4f}<extra></extra>"
        ))
        fig_bar.update_layout(
            title=f"Top {top_n} 特征重要性 (平均|SHAP值|)",
            xaxis_title="平均 |SHAP值| (影响程度)",
            yaxis_title="特征名",
            height=600
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.subheader("🎯 SHAP蜂群图 (Summary Plot)")
        X_test_scaled = data["model"][:-1].transform(test_df[feature_names].values)
        
        top_features = data["feature_importance"][:top_n]
        top_indices = [feature_names.index(f) for f in top_features]
        
        fig_dot = go.Figure()
        
        for i, feat_idx in enumerate(reversed(top_indices)):
            feat_name = feature_names[feat_idx]
            shap_vals = shap_values[:, feat_idx]
            feat_vals = X_test_scaled[:, feat_idx]
            
            fig_dot.add_trace(go.Scatter(
                x=shap_vals,
                y=[feat_name] * len(shap_vals),
                mode="markers",
                marker=dict(
                    color=feat_vals,
                    colorscale="RdBu",
                    size=8,
                    opacity=0.7,
                    showscale=(i == 0),
                    colorbar=dict(title="特征值") if i == 0 else None
                ),
                name=feat_name,
                hovertemplate="SHAP值: %{x:.4f}<br>特征: %{y}<extra></extra>"
            ))
        
        fig_dot.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_dot.update_layout(
            title="SHAP值分布 (每个点代表一个样本)",
            xaxis_title="SHAP值 (对预测的影响)",
            yaxis_title="特征名",
            height=600,
            showlegend=False
        )
        st.plotly_chart(fig_dot, use_container_width=True)
    
    st.markdown("---")
    
    # SHAP依赖图
    st.subheader("🔗 SHAP依赖图 (Dependence Plot)")
    selected_feature = st.selectbox(
        "选择要查看的特征",
        options=data["feature_importance"][:top_n],
        index=0
    )
    
    feat_idx = feature_names.index(selected_feature)
    feat_shap = shap_values[:, feat_idx]
    feat_vals = X_test_scaled[:, feat_idx]
    
    fig_dep = px.scatter(
        x=feat_vals,
        y=feat_shap,
        color=predictions_df["prob_malignant"],
        color_continuous_scale="RdBu_r",
        labels={
            "x": f"{selected_feature} (标准化后)",
            "y": "SHAP值",
            "color": "预测恶性概率"
        },
        title=f"特征 '{selected_feature}' 的SHAP依赖图",
        hover_data={
            "真实标签": predictions_df["target_name"],
            "预测标签": predictions_df["predicted_name"]
        }
    )
    fig_dep.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_dep.update_traces(marker=dict(size=10, opacity=0.8))
    fig_dep.update_layout(height=500)
    st.plotly_chart(fig_dep, use_container_width=True)

# ==================== 页面4: 特征分析 ====================
elif page == "📊 特征分析":
    st.header("📊 特征分布与相关性分析")
    
    # 特征选择
    selected_feature = st.selectbox(
        "选择要分析的特征",
        options=feature_names,
        index=0
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📈 特征分布直方图")
        fig_dist = go.Figure()
        for target in [0, 1]:
            subset = train_df[train_df["target"] == target]
            fig_dist.add_trace(go.Histogram(
                x=subset[selected_feature],
                name=f"{target_names[target]}",
                opacity=0.7,
                nbinsx=25,
                hovertemplate="区间: %{x:.2f}<br>样本数: %{y}<extra></extra>"
            ))
        fig_dist.update_layout(
            title=f"'{selected_feature}' 在不同类别中的分布",
            xaxis_title=selected_feature,
            yaxis_title="样本数",
            barmode="overlay",
            height=450
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.subheader("📦 箱线图")
        fig_box = go.Figure()
        for target in [0, 1]:
            subset = train_df[train_df["target"] == target]
            fig_box.add_trace(go.Box(
                y=subset[selected_feature],
                name=f"{target_names[target]}",
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.8,
                hovertemplate="%{y:.2f}<extra></extra>"
            ))
        fig_box.update_layout(
            title=f"'{selected_feature}' 箱线图",
            yaxis_title=selected_feature,
            height=450
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    st.markdown("---")
    
    # 相关性矩阵
    st.subheader("🔥 特征相关性热力图")
    corr_features = st.multiselect(
        "选择要显示相关性的特征 (默认显示Top 10重要特征)",
        options=feature_names,
        default=data["feature_importance"][:10]
    )
    
    if corr_features:
        corr_matrix = train_df[corr_features].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            range_color=[-1, 1],
            title="特征相关性矩阵"
        )
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("---")
    
    # 散点矩阵
    st.subheader("🔀 特征散点矩阵")
    scatter_features = st.multiselect(
        "选择3-5个特征绘制散点矩阵",
        options=feature_names,
        default=data["feature_importance"][:4],
        max_selections=6
    )
    
    if len(scatter_features) >= 2:
        fig_scatter = px.scatter_matrix(
            train_df,
            dimensions=scatter_features,
            color="target_name",
            color_discrete_map={"benign": "#3498db", "malignant": "#e74c3c"},
            title="特征散点矩阵 (按类别着色)",
            labels={col: col.replace("mean ", "")[:15] for col in scatter_features}
        )
        fig_scatter.update_layout(height=700)
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==================== 页面5: 单样本预测 ====================
elif page == "🔮 单样本预测":
    st.header("🔮 单样本预测与解释")
    
    # 样本选择方式
    select_method = st.radio(
        "选择样本方式",
        ["从测试集选择", "手动输入特征值"],
        horizontal=True
    )
    
    model = data["model"]
    
    if select_method == "从测试集选择":
        sample_idx = st.slider("选择测试集样本索引", 0, len(predictions_df) - 1, 0)
        sample = predictions_df.iloc[sample_idx]
        
        st.subheader(f"样本 #{sample_idx} 信息")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("真实标签", sample["target_name"])
        with col2:
            st.metric("预测标签", sample["predicted_name"])
        with col3:
            st.metric("良性概率", f"{sample['prob_benign']:.4f}")
        with col4:
            st.metric("恶性概率", f"{sample['prob_malignant']:.4f}")
        
        X_sample = test_df[feature_names].iloc[sample_idx:sample_idx+1].values
        sample_shap = data["shap_values"][sample_idx]
    else:
        st.subheader("输入特征值")
        st.info("提示: 特征值范围参考乳腺癌数据集的原始范围")
        
        col1, col2, col3 = st.columns(3)
        input_values = {}
        
        for i, feat in enumerate(feature_names):
            col_idx = i % 3
            with [col1, col2, col3][col_idx]:
                mean_val = train_df[feat].mean()
                std_val = train_df[feat].std()
                input_values[feat] = st.number_input(
                    feat,
                    value=float(mean_val),
                    step=float(std_val / 10),
                    format="%.4f"
                )
        
        X_sample = np.array([list(input_values.values())])
        pred_proba = model.predict_proba(X_sample)[0]
        pred = model.predict(X_sample)[0]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("预测标签", target_names[pred])
        with col2:
            st.metric("良性概率", f"{pred_proba[0]:.4f}")
        with col3:
            st.metric("恶性概率", f"{pred_proba[1]:.4f}")
        
        # 对新样本计算SHAP值（简化版本，使用测试集SHAP的类似模式）
        st.info("手动输入样本的SHAP解释需要重新计算，当前显示为参考模式")
        sample_shap = data["mean_abs_shap"] * 0.5  # 简化处理
    
    st.markdown("---")
    
    if select_method == "从测试集选择":
        st.subheader("🎯 该样本的SHAP特征贡献瀑布图")
        
        # 计算基值和SHAP贡献
        X_scaled = model[:-1].transform(X_sample)
        clf = model.named_steps["classifier"]
        
        if hasattr(clf, "predict_proba"):
            try:
                import shap as shap_lib
                explainer = shap_lib.TreeExplainer(clf) if "Forest" in type(clf).__name__ or "Tree" in type(clf).__name__ else None
                if explainer is not None:
                    base_value = float(explainer.expected_value)
                    if isinstance(base_value, list):
                        base_value = base_value[1]
                else:
                    base_value = 0.5
            except:
                base_value = 0.5
        else:
            base_value = 0.5
        
        # 创建SHAP瀑布图
        sorted_idx = np.argsort(np.abs(sample_shap))[::-1]
        top_n_waterfall = 15
        top_sorted_idx = sorted_idx[:top_n_waterfall]
        
        waterfall_features = [feature_names[i] for i in top_sorted_idx]
        waterfall_shap = [sample_shap[i] for i in top_sorted_idx]
        
        fig_waterfall = go.Figure(go.Waterfall(
            name="SHAP贡献",
            orientation="h",
            measure=["relative"] * len(waterfall_features) + ["total"],
            y=waterfall_features[::-1] + ["最终预测"],
            x=[float(s) for s in waterfall_shap[::-1]] + [float(sample["prob_malignant"] if select_method == "从测试集选择" else pred_proba[1]) - base_value],
            base=base_value,
            hovertemplate="特征: %{y}<br>SHAP贡献: %{x:.4f}<extra></extra>"
        ))
        fig_waterfall.update_layout(
            title=f"SHAP瀑布图: 各特征对预测的贡献 (基值={base_value:.4f})",
            xaxis_title="预测为恶性的概率",
            yaxis_title="特征",
            height=600,
            showlegend=False
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        st.markdown("""
        **💡 瀑布图解读**:
        - **基值**: 所有样本的平均预测概率
        - **红色条**: 该特征值推动预测向"恶性"发展
        - **蓝色条**: 该特征值推动预测向"良性"发展
        - **最终值**: 该样本的预测概率
        """)

# ==================== 底部信息 ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    📊 乳腺癌分类预测可视化仪表盘 | 基于 Streamlit + Plotly 构建
</div>
""", unsafe_allow_html=True)
