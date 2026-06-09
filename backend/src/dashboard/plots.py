"""
可视化图表生成模块。
使用Plotly创建各种交互式图表，包括性能指标、混淆矩阵、ROC曲线、SHAP可视化等。
"""
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def create_metrics_cards(metrics: Dict[str, float]) -> List[go.Figure]:
    """
    创建性能指标卡片。

    :param metrics: 包含性能指标的字典
    :return: 指标卡片图表列表
    """
    card_configs = [
        {"key": "accuracy", "title": "准确率 (Accuracy)", "color": "#2E86AB"},
        {"key": "precision", "title": "精确率 (Precision)", "color": "#A23B72"},
        {"key": "recall", "title": "召回率 (Recall)", "color": "#F18F01"},
        {"key": "f1", "title": "F1 分数", "color": "#C73E1D"},
    ]

    figures = []
    for config in card_configs:
        value = metrics.get(config["key"], 0)
        fig = go.Figure()
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=value * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": config["title"], "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkgray"},
                    "bar": {"color": config["color"]},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "lightgray",
                    "steps": [
                        {"range": [0, 60], "color": "#f8d7da"},
                        {"range": [60, 80], "color": "#fff3cd"},
                        {"range": [80, 100], "color": "#d4edda"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90,
                    },
                },
                number={"suffix": "%", "font": {"size": 36}},
            )
        )
        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="white",
        )
        figures.append(fig)

    return figures


def create_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> go.Figure:
    """
    创建混淆矩阵热力图。

    :param cm: 混淆矩阵数组
    :param class_names: 类别名称列表
    :return: Plotly图表对象
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale="Blues",
            showscale=True,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16, "color": "white"},
            hoverongaps=False,
        )
    )

    # 添加百分比注释
    total = np.sum(cm)
    percentages = (cm / total * 100).round(1)
    annotations = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            annotations.append(
                dict(
                    x=class_names[j],
                    y=class_names[i],
                    text=f"{percentages[i, j]}%",
                    showarrow=False,
                    font=dict(color="white" if cm[i, j] > cm.max() / 2 else "black", size=12),
                    yshift=-20,
                )
            )

    fig.update_layout(
        title={
            "text": "混淆矩阵 (Confusion Matrix)",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18},
        },
        xaxis_title="预测类别",
        yaxis_title="真实类别",
        height=500,
        annotations=annotations,
        margin=dict(l=60, r=60, t=80, b=60),
    )

    fig.update_yaxes(autorange="reversed")

    return fig


def create_roc_curve(roc_data: Dict[str, Any]) -> go.Figure:
    """
    创建ROC曲线图。

    :param roc_data: 包含ROC曲线数据的字典
    :return: Plotly图表对象
    """
    fig = go.Figure()

    # 绘制ROC曲线
    fig.add_trace(
        go.Scatter(
            x=roc_data["fpr"],
            y=roc_data["tpr"],
            mode="lines",
            name=f'ROC 曲线 (AUC = {roc_data["auc"]:.4f})',
            line=dict(color="#2E86AB", width=3),
            fill="tozeroy",
            fillcolor="rgba(46, 134, 171, 0.1)",
        )
    )

    # 绘制对角线（随机猜测）
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="随机猜测",
            line=dict(color="gray", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title={
            "text": "ROC 曲线 (Receiver Operating Characteristic)",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18},
        },
        xaxis_title="假阳性率 (False Positive Rate)",
        yaxis_title="真阳性率 (True Positive Rate)",
        height=500,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        legend=dict(x=0.6, y=0.2),
        margin=dict(l=60, r=60, t=80, b=60),
        hovermode="x unified",
    )

    return fig


def create_shap_summary_plot(shap_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    创建SHAP摘要图（beeswarm风格）。

    :param shap_df: SHAP值DataFrame（长格式）
    :param top_n: 显示的特征数量
    :return: Plotly图表对象
    """
    # 计算每个特征的平均绝对SHAP值，用于排序
    feature_importance = (
        shap_df.groupby("feature")["shap_value"]
        .apply(lambda x: np.mean(np.abs(x)))
        .sort_values(ascending=False)
    )

    # 选择top_n特征
    top_features = feature_importance.head(top_n).index.tolist()
    shap_top = shap_df[shap_df["feature"].isin(top_features)]

    # 按照重要性排序
    shap_top["feature"] = pd.Categorical(
        shap_top["feature"], categories=top_features[::-1], ordered=True
    )

    fig = px.strip(
        shap_top,
        x="shap_value",
        y="feature",
        color="feature_value",
        color_continuous_scale="RdBu_r",
        hover_data=["sample_id", "feature_value", "shap_value"],
    )

    fig.update_layout(
        title={
            "text": "SHAP 摘要图 (特征重要性与影响方向)",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18},
        },
        xaxis_title="SHAP 值 (对预测结果的影响)",
        yaxis_title="特征",
        height=600,
        margin=dict(l=150, r=60, t=80, b=60),
        coloraxis_colorbar=dict(
            title="特征值<br>(标准化)",
            thickness=15,
            len=0.7,
        ),
    )

    # 添加垂直参考线
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="gray")

    return fig


def create_shap_bar_plot(shap_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """
    创建SHAP条形图（平均绝对SHAP值）。

    :param shap_df: SHAP值DataFrame（长格式）
    :param top_n: 显示的特征数量
    :return: Plotly图表对象
    """
    # 计算每个特征的平均绝对SHAP值
    feature_importance = (
        shap_df.groupby("feature")["shap_value"]
        .apply(lambda x: np.mean(np.abs(x)))
        .sort_values(ascending=True)
        .tail(top_n)
    )

    fig = go.Figure()

    colors = [
        "#2E86AB" if val < feature_importance.median() else "#A23B72"
        for val in feature_importance.values
    ]

    fig.add_trace(
        go.Bar(
            x=feature_importance.values,
            y=feature_importance.index,
            orientation="h",
            marker_color=colors,
            text=feature_importance.values.round(4),
            textposition="outside",
            hovertemplate="特征: %{y}<br>平均|SHAP|: %{x:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": "SHAP 特征重要性 (平均绝对 SHAP 值)",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18},
        },
        xaxis_title="平均 |SHAP| 值",
        yaxis_title="特征",
        height=600,
        margin=dict(l=150, r=80, t=80, b=60),
    )

    return fig


def create_optuna_history_plot(history_df: pd.DataFrame) -> go.Figure:
    """
    创建Optuna调参历史图。

    :param history_df: 调参历史DataFrame
    :return: Plotly图表对象
    """
    # 计算累计最佳值
    history_df = history_df.sort_values("trial_number").reset_index(drop=True)
    history_df["best_value"] = history_df["value"].cummax()

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("试验分数", "累计最佳分数"),
        vertical_spacing=0.15,
    )

    # 试验分数散点图
    fig.add_trace(
        go.Scatter(
            x=history_df["trial_number"],
            y=history_df["value"],
            mode="markers+lines",
            name="单次试验",
            marker=dict(color="#2E86AB", size=8),
            line=dict(color="#2E86AB", width=1),
            hovertemplate="试验: %{x}<br>分数: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # 累计最佳值折线图
    fig.add_trace(
        go.Scatter(
            x=history_df["trial_number"],
            y=history_df["best_value"],
            mode="lines+markers",
            name="累计最佳",
            marker=dict(color="#C73E1D", size=6),
            line=dict(color="#C73E1D", width=3),
            hovertemplate="试验: %{x}<br>最佳分数: %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title={
            "text": "Optuna 超参数调优过程",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18},
        },
        height=600,
        margin=dict(l=60, r=60, t=80, b=60),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    fig.update_xaxes(title_text="试验次数", row=2, col=1)
    fig.update_yaxes(title_text="F1 分数 (加权)", row=1, col=1)
    fig.update_yaxes(title_text="最佳 F1 分数", row=2, col=1)

    return fig


def create_feature_distribution_plot(df: pd.DataFrame, feature: str) -> go.Figure:
    """
    创建特征分布图，按类别分组。

    :param df: 特征数据DataFrame
    :param feature: 要显示的特征名称
    :return: Plotly图表对象
    """
    fig = go.Figure()

    for target_name in df["target_name"].unique():
        subset = df[df["target_name"] == target_name]
        fig.add_trace(
            go.Histogram(
                x=subset[feature],
                name=target_name,
                opacity=0.7,
                nbinsx=30,
                hovertemplate=f"{target_name}<br>{feature}: %{{x:.2f}}<br>计数: %{{y}}<extra></extra>",
            )
        )

    fig.update_layout(
        title={
            "text": f"特征分布: {feature}",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18},
        },
        xaxis_title=feature,
        yaxis_title="样本数量",
        height=400,
        barmode="overlay",
        margin=dict(l=60, r=60, t=80, b=60),
        legend=dict(title="类别"),
    )

    return fig


def create_feature_boxplot(df: pd.DataFrame, feature: str) -> go.Figure:
    """
    创建特征箱线图，按类别分组。

    :param df: 特征数据DataFrame
    :param feature: 要显示的特征名称
    :return: Plotly图表对象
    """
    fig = px.box(
        df,
        x="target_name",
        y=feature,
        color="target_name",
        points="outliers",
        hover_data=df.columns,
    )

    fig.update_layout(
        title={
            "text": f"特征箱线图: {feature}",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18},
        },
        xaxis_title="类别",
        yaxis_title=feature,
        height=400,
        margin=dict(l=60, r=60, t=80, b=60),
        showlegend=True,
        legend=dict(title="类别"),
    )

    return fig


def create_single_sample_waterfall(sample_data: Dict[str, Any]) -> go.Figure:
    """
    创建单样本预测解释的瀑布图。

    :param sample_data: 单样本数据字典
    :return: Plotly图表对象
    """
    features = sample_data["top_features"][::-1]
    shap_values = sample_data["top_shap_values"][::-1]

    # 计算基础值（假设为0，实际可以根据explainer.expected_value）
    base_value = 0

    # 构建瀑布图数据
    measure = ["relative"] * len(features)
    text = [f"{v:.4f}" for v in shap_values]

    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            y=features,
            x=shap_values,
            measure=measure,
            text=text,
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#2E86AB"}},
            increasing={"marker": {"color": "#C73E1D"}},
        )
    )

    fig.update_layout(
        title={
            "text": f"样本 #{sample_data['sample_idx']} 预测解释 (SHAP 瀑布图)",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 16},
        },
        xaxis_title="SHAP 值 (对预测结果的贡献)",
        yaxis_title="特征",
        height=500,
        margin=dict(l=150, r=80, t=80, b=60),
        showlegend=False,
    )

    # 添加垂直参考线
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="gray")

    return fig


def create_prediction_probability_gauge(sample_data: Dict[str, Any]) -> go.Figure:
    """
    创建预测概率仪表盘。

    :param sample_data: 单样本数据字典
    :return: Plotly图表对象
    """
    prob_malignant = sample_data["probability_malignant"]
    is_correct = sample_data["y_true"] == sample_data["y_pred"]

    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=prob_malignant * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={
                "text": f"恶性概率预测<br><span style='font-size: 12px; color: {'green' if is_correct else 'red'};'>"
                f"真实: {sample_data['y_true_name']} | 预测: {sample_data['y_pred_name']}</span>",
            },
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkgray"},
                "bar": {"color": "#C73E1D"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "lightgray",
                "steps": [
                    {"range": [0, 30], "color": "#d4edda"},
                    {"range": [30, 70], "color": "#fff3cd"},
                    {"range": [70, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
            number={"suffix": "%", "font": {"size": 36}},
            delta={"reference": 50, "increasing": {"color": "red"}, "decreasing": {"color": "green"}},
        )
    )

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="white",
    )

    return fig


def create_parallel_coordinates_plot(df: pd.DataFrame, features: List[str]) -> go.Figure:
    """
    创建平行坐标图，用于多特征比较。

    :param df: 特征数据DataFrame
    :param features: 要显示的特征列表
    :return: Plotly图表对象
    """
    # 选择数据
    plot_df = df[features + ["target"]].copy()

    # 创建维度
    dimensions = []
    for feature in features:
        dimensions.append(
            dict(
                range=[plot_df[feature].min(), plot_df[feature].max()],
                label=feature,
                values=plot_df[feature],
            )
        )

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=plot_df["target"],
                colorscale=[[0, "#2E86AB"], [1, "#C73E1D"]],
                showscale=True,
                colorbar=dict(title="类别", tickvals=[0, 1], ticktext=["良性", "恶性"]),
            ),
            dimensions=dimensions,
        )
    )

    fig.update_layout(
        title={
            "text": "平行坐标图 (多特征对比)",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18},
        },
        height=500,
        margin=dict(l=60, r=80, t=80, b=60),
    )

    return fig
