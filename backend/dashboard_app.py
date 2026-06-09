"""
癌症分类预测 - 交互式可视化仪表盘
使用Dash框架构建，提供多维度的模型性能、特征重要性和预测解释可视化。
"""
import logging
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go

# 将backend目录加入Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.dashboard.data_utils import DashboardDataManager
from src.dashboard.plots import (
    create_metrics_cards,
    create_confusion_matrix,
    create_roc_curve,
    create_shap_summary_plot,
    create_shap_bar_plot,
    create_optuna_history_plot,
    create_feature_distribution_plot,
    create_feature_boxplot,
    create_single_sample_waterfall,
    create_prediction_probability_gauge,
    create_parallel_coordinates_plot,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 初始化数据管理器
data_manager = DashboardDataManager(random_state=42, test_size=0.2)

# 初始化Dash应用
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="癌症分类预测仪表盘",
    suppress_callback_exceptions=True,
)

# ==================== 页面布局组件 ====================

def create_header():
    """创建页面头部"""
    return dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(html.Img(src="https://plotly.com/~chris/1638/image.png", height="40px")),
                            dbc.Col(dbc.NavbarBrand("癌症分类预测 - 交互式可视化仪表盘", className="ms-2")),
                        ],
                        align="center",
                        className="g-0",
                    ),
                    href="#",
                    style={"textDecoration": "none"},
                ),
                dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                dbc.Collapse(
                    dbc.Nav(
                        [
                            dbc.NavItem(dbc.NavLink("模型性能", href="#performance-section")),
                            dbc.NavItem(dbc.NavLink("特征重要性", href="#shap-section")),
                            dbc.NavItem(dbc.NavLink("调参过程", href="#optuna-section")),
                            dbc.NavItem(dbc.NavLink("数据探索", href="#exploration-section")),
                            dbc.NavItem(dbc.NavLink("单样本解释", href="#sample-section")),
                        ],
                        className="ms-auto",
                        navbar=True,
                    ),
                    id="navbar-collapse",
                    navbar=True,
                ),
            ]
        ),
        color="primary",
        dark=True,
        className="mb-4",
    )


def create_control_panel():
    """创建控制面板"""
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5("模型配置", className="card-title"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("分类器类型"),
                                dcc.Dropdown(
                                    id="classifier-dropdown",
                                    options=[
                                        {"label": "支持向量机 (SVC)", "value": "svc"},
                                        {"label": "随机森林 (Random Forest)", "value": "random_forest"},
                                    ],
                                    value="svc",
                                    clearable=False,
                                ),
                            ],
                            width=4,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Optuna试验次数"),
                                dcc.Slider(
                                    id="n-trials-slider",
                                    min=5,
                                    max=100,
                                    step=5,
                                    value=20,
                                    marks={i: str(i) for i in range(10, 101, 10)},
                                ),
                            ],
                            width=5,
                        ),
                        dbc.Col(
                            [
                                dbc.Label(" "),
                                dbc.Button(
                                    "训练模型",
                                    id="train-button",
                                    color="primary",
                                    n_clicks=0,
                                    className="w-100",
                                ),
                            ],
                            width=3,
                        ),
                    ]
                ),
                dbc.Spinner(
                    id="training-status",
                    color="primary",
                    type="border",
                    fullscreen=False,
                    children=html.Div(id="training-status-text", className="mt-2 text-muted"),
                    spinner_style={"display": "none"},
                ),
            ]
        ),
        className="mb-4",
    )


def create_performance_tab():
    """创建模型性能标签页"""
    return dbc.Card(
        dbc.CardBody(
            [
                html.H4("模型性能指标", className="card-title mb-4", id="performance-section"),
                # 指标卡片行
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id="accuracy-gauge"), width=3),
                        dbc.Col(dcc.Graph(id="precision-gauge"), width=3),
                        dbc.Col(dcc.Graph(id="recall-gauge"), width=3),
                        dbc.Col(dcc.Graph(id="f1-gauge"), width=3),
                    ],
                    className="mb-4",
                ),
                # 混淆矩阵和ROC曲线
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id="confusion-matrix"), width=6),
                        dbc.Col(dcc.Graph(id="roc-curve"), width=6),
                    ]
                ),
                # 分类报告
                dbc.Row(
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("分类报告", className="card-title"),
                                    html.Pre(id="classification-report", style={"background": "#f8f9fa", "padding": "15px", "border-radius": "5px"}),
                                ]
                            ),
                            className="mt-4",
                        ),
                        width=12,
                    )
                ),
            ]
        ),
        className="mb-4",
    )


def create_shap_tab():
    """创建特征重要性（SHAP）标签页"""
    return dbc.Card(
        dbc.CardBody(
            [
                html.H4("特征重要性分析 (SHAP)", className="card-title mb-4", id="shap-section"),
                # 特征数量选择
                dbc.Row(
                    dbc.Col(
                        [
                            dbc.Label("显示特征数量:"),
                            dcc.Slider(
                                id="shap-top-n-slider",
                                min=5,
                                max=30,
                                step=1,
                                value=15,
                                marks={i: str(i) for i in range(5, 31, 5)},
                            ),
                        ],
                        width=12,
                    ),
                    className="mb-4",
                ),
                # SHAP摘要图和条形图
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id="shap-summary-plot"), width=6),
                        dbc.Col(dcc.Graph(id="shap-bar-plot"), width=6),
                    ]
                ),
            ]
        ),
        className="mb-4",
    )


def create_optuna_tab():
    """创建Optuna调参标签页"""
    return dbc.Card(
        dbc.CardBody(
            [
                html.H4("超参数调优过程 (Optuna)", className="card-title mb-4", id="optuna-section"),
                dcc.Graph(id="optuna-history-plot"),
                # 最佳参数展示
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5("最佳超参数", className="card-title"),
                            html.Div(id="best-params-display"),
                        ]
                    ),
                    className="mt-4",
                    color="light",
                ),
            ]
        ),
        className="mb-4",
    )


def create_exploration_tab():
    """创建数据探索标签页"""
    return dbc.Card(
        dbc.CardBody(
            [
                html.H4("数据探索与特征分析", className="card-title mb-4", id="exploration-section"),
                # 特征选择
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("选择特征:"),
                                dcc.Dropdown(
                                    id="feature-dropdown",
                                    value="mean radius",
                                    clearable=False,
                                ),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("数据集:"),
                                dcc.RadioItems(
                                    id="dataset-radio",
                                    options=[
                                        {"label": "训练集", "value": "train"},
                                        {"label": "测试集", "value": "test"},
                                    ],
                                    value="test",
                                    inline=True,
                                    labelStyle={"marginRight": "20px"},
                                ),
                            ],
                            width=6,
                        ),
                    ],
                    className="mb-4",
                ),
                # 分布图和箱线图
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id="feature-distribution"), width=6),
                        dbc.Col(dcc.Graph(id="feature-boxplot"), width=6),
                    ]
                ),
                # 平行坐标图
                dbc.Row(
                    dbc.Col(
                        [
                            html.H5("多特征平行坐标图 (Top 6)", className="mt-4"),
                            dcc.Graph(id="parallel-coordinates"),
                        ],
                        width=12,
                    )
                ),
            ]
        ),
        className="mb-4",
    )


def create_sample_tab():
    """创建单样本解释标签页"""
    return dbc.Card(
        dbc.CardBody(
            [
                html.H4("单样本预测解释", className="card-title mb-4", id="sample-section"),
                # 样本选择
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("选择样本索引:"),
                                dcc.Slider(
                                    id="sample-slider",
                                    min=0,
                                    max=50,
                                    step=1,
                                    value=0,
                                    marks={i: str(i) for i in range(0, 51, 5)},
                                ),
                            ],
                            width=8,
                        ),
                        dbc.Col(
                            [
                                dbc.Label("快速跳转:"),
                                dbc.ButtonGroup(
                                    [
                                        dbc.Button("上一个", id="prev-sample", color="secondary", outline=True),
                                        dbc.Button("下一个", id="next-sample", color="secondary", outline=True),
                                    ],
                                    className="w-100",
                                ),
                            ],
                            width=4,
                        ),
                    ],
                    className="mb-4",
                ),
                # 预测概率和瀑布图
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id="prediction-gauge"), width=4),
                        dbc.Col(dcc.Graph(id="sample-waterfall"), width=8),
                    ]
                ),
                # 样本特征详情
                dbc.Row(
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("样本特征详情", className="card-title"),
                                    html.Div(id="sample-details"),
                                ]
                            ),
                            className="mt-4",
                            color="light",
                        ),
                        width=12,
                    )
                ),
            ]
        ),
        className="mb-4",
    )


def create_footer():
    """创建页脚"""
    return dbc.Container(
        html.Footer(
            html.P(
                "癌症分类预测系统 - 交互式可视化仪表盘 | 基于 sklearn + Optuna + SHAP + Dash 构建",
                className="text-center text-muted mt-4 mb-4",
            ),
            style={"borderTop": "1px solid #dee2e6", "paddingTop": "20px"},
        )
    )


# ==================== 主布局 ====================

app.layout = html.Div(
    [
        create_header(),
        dbc.Container(
            [
                create_control_panel(),
                # 使用Tabs组织内容
                dcc.Tabs(
                    id="main-tabs",
                    value="performance",
                    children=[
                        dcc.Tab(label="模型性能", value="performance", children=[create_performance_tab()]),
                        dcc.Tab(label="特征重要性", value="shap", children=[create_shap_tab()]),
                        dcc.Tab(label="调参过程", value="optuna", children=[create_optuna_tab()]),
                        dcc.Tab(label="数据探索", value="exploration", children=[create_exploration_tab()]),
                        dcc.Tab(label="单样本解释", value="sample", children=[create_sample_tab()]),
                    ],
                    className="mb-4",
                ),
                # 存储数据的隐藏组件
                dcc.Store(id="data-loaded", data=False),
            ],
            fluid=True,
        ),
        create_footer(),
    ]
)


# ==================== 回调函数 ====================

@callback(
    Output("training-status-text", "children"),
    Output("data-loaded", "data"),
    Output("feature-dropdown", "options"),
    Output("sample-slider", "max"),
    Output("sample-slider", "marks"),
    Input("train-button", "n_clicks"),
    State("classifier-dropdown", "value"),
    State("n-trials-slider", "value"),
    prevent_initial_call=False,
)
def train_model(n_clicks, classifier, n_trials):
    """
    训练模型并准备数据。
    初始加载时自动训练一次。
    """
    ctx = dash.callback_context
    
    # 如果是初始加载，自动训练
    if not ctx.triggered:
        logger.info("初始加载，自动训练模型...")
        n_clicks = 1
    
    if n_clicks > 0:
        try:
            status_text = f"正在训练 {classifier} 模型，Optuna试验次数: {n_trials}..."
            logger.info(status_text)
            
            # 加载数据
            data_manager.load_data()
            
            # 训练模型
            data_manager.train_model(
                classifier=classifier,
                n_trials=n_trials,
                use_optuna=True,
            )
            
            # 计算SHAP值
            data_manager.compute_shap_values(sample_size=100)
            
            # 获取特征列表
            feature_names = data_manager.data["feature_names"]
            feature_options = [{"label": name, "value": name} for name in feature_names]
            
            # 获取测试集样本数
            n_samples = len(data_manager.data["y_test"])
            max_sample = min(n_samples - 1, 100)  # 限制最多100个样本可选
            sample_marks = {i: str(i) for i in range(0, max_sample + 1, 10)}
            
            status_text = f"模型训练完成! 分类器: {classifier}, 试验次数: {n_trials}"
            logger.info(status_text)
            
            return status_text, True, feature_options, max_sample, sample_marks
            
        except Exception as e:
            error_msg = f"训练出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    return "", False, [], 0, {}


@callback(
    Output("accuracy-gauge", "figure"),
    Output("precision-gauge", "figure"),
    Output("recall-gauge", "figure"),
    Output("f1-gauge", "figure"),
    Output("confusion-matrix", "figure"),
    Output("roc-curve", "figure"),
    Output("classification-report", "children"),
    Input("data-loaded", "data"),
    prevent_initial_call=True,
)
def update_performance_metrics(data_loaded):
    """更新性能指标图表"""
    if not data_loaded:
        return [go.Figure()] * 6 + [""]
    
    # 获取指标
    metrics = data_manager.get_metrics()
    metric_figures = create_metrics_cards(metrics)
    
    # 获取混淆矩阵
    cm, class_names = data_manager.get_confusion_matrix()
    cm_fig = create_confusion_matrix(cm, class_names)
    
    # 获取ROC曲线
    roc_data = data_manager.get_roc_data()
    roc_fig = create_roc_curve(roc_data)
    
    # 获取分类报告
    report = data_manager.get_classification_report()
    
    return *metric_figures, cm_fig, roc_fig, report


@callback(
    Output("shap-summary-plot", "figure"),
    Output("shap-bar-plot", "figure"),
    Input("data-loaded", "data"),
    Input("shap-top-n-slider", "value"),
    prevent_initial_call=True,
)
def update_shap_plots(data_loaded, top_n):
    """更新SHAP图表"""
    if not data_loaded:
        return go.Figure(), go.Figure()
    
    shap_df = data_manager.get_shap_dataframe()
    summary_fig = create_shap_summary_plot(shap_df, top_n=top_n)
    bar_fig = create_shap_bar_plot(shap_df, top_n=top_n)
    
    return summary_fig, bar_fig


@callback(
    Output("optuna-history-plot", "figure"),
    Output("best-params-display", "children"),
    Input("data-loaded", "data"),
    prevent_initial_call=True,
)
def update_optuna_plots(data_loaded):
    """更新Optuna调参图表"""
    if not data_loaded:
        return go.Figure(), ""
    
    # 获取调参历史
    history_df = data_manager.get_optuna_history()
    history_fig = create_optuna_history_plot(history_df)
    
    # 显示最佳参数
    best_params = data_manager.best_params
    params_list = []
    for key, value in best_params.items():
        if isinstance(value, float):
            display_value = f"{value:.4f}"
        else:
            display_value = str(value)
        params_list.append(
            dbc.Badge(
                f"{key}: {display_value}",
                color="primary",
                className="me-2 mb-2",
                style={"fontSize": "14px", "padding": "8px 12px"},
            )
        )
    
    best_params_div = html.Div(params_list)
    
    return history_fig, best_params_div


@callback(
    Output("feature-distribution", "figure"),
    Output("feature-boxplot", "figure"),
    Output("parallel-coordinates", "figure"),
    Input("data-loaded", "data"),
    Input("feature-dropdown", "value"),
    Input("dataset-radio", "value"),
    prevent_initial_call=True,
)
def update_exploration_plots(data_loaded, feature, dataset):
    """更新数据探索图表"""
    if not data_loaded:
        return go.Figure(), go.Figure(), go.Figure()
    
    df = data_manager.get_feature_dataframe(dataset=dataset)
    
    # 分布图和箱线图
    dist_fig = create_feature_distribution_plot(df, feature)
    box_fig = create_feature_boxplot(df, feature)
    
    # 平行坐标图 - 使用Top 6特征
    if data_manager.shap_summary is not None:
        top_features = data_manager.shap_summary["importance_order"][:6]
    else:
        top_features = list(df.columns[:6])
    
    # 确保不包含target列
    top_features = [f for f in top_features if f not in ["target", "target_name"]]
    
    parallel_fig = create_parallel_coordinates_plot(df, top_features)
    
    return dist_fig, box_fig, parallel_fig


@callback(
    Output("sample-slider", "value"),
    Input("prev-sample", "n_clicks"),
    Input("next-sample", "n_clicks"),
    State("sample-slider", "value"),
    State("sample-slider", "max"),
    prevent_initial_call=True,
)
def navigate_samples(prev_clicks, next_clicks, current_value, max_value):
    """样本导航按钮"""
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_value
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "prev-sample":
        return max(0, current_value - 1)
    elif trigger_id == "next-sample":
        return min(max_value, current_value + 1)
    
    return current_value


@callback(
    Output("prediction-gauge", "figure"),
    Output("sample-waterfall", "figure"),
    Output("sample-details", "children"),
    Input("data-loaded", "data"),
    Input("sample-slider", "value"),
    prevent_initial_call=True,
)
def update_sample_explanation(data_loaded, sample_idx):
    """更新单样本解释"""
    if not data_loaded:
        return go.Figure(), go.Figure(), ""
    
    sample_data = data_manager.get_single_sample_prediction(sample_idx)
    
    # 预测概率仪表盘
    gauge_fig = create_prediction_probability_gauge(sample_data)
    
    # SHAP瀑布图
    waterfall_fig = create_single_sample_waterfall(sample_data)
    
    # 样本特征详情表格
    feature_names = sample_data["all_feature_names"]
    feature_values = sample_data["all_feature_values"]
    
    # 创建表格数据
    table_header = [
        html.Thead(html.Tr([html.Th("特征名称"), html.Th("特征值")]))
    ]
    
    rows = []
    for name, value in zip(feature_names, feature_values):
        rows.append(html.Tr([html.Td(name), html.Td(f"{value:.4f}")]))
    
    table_body = [html.Tbody(rows)]
    
    details_table = dbc.Table(
        table_header + table_body,
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
        size="sm",
        style={"fontSize": "12px"},
    )
    
    return gauge_fig, waterfall_fig, details_table


# ==================== 主函数 ====================

def main():
    """主函数：启动Dash服务器"""
    logger.info("=" * 60)
    logger.info("癌症分类预测 - 交互式可视化仪表盘")
    logger.info("=" * 60)
    logger.info("启动服务器: http://localhost:8050")
    logger.info("=" * 60)
    
    # 启动服务器
    app.run_server(
        host="0.0.0.0",
        port=8050,
        debug=False,
    )


if __name__ == "__main__":
    main()
