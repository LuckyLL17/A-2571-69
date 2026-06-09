# 癌症数据集分类预测（管道 + Optuna + SHAP + 交互式仪表盘）

基于 Python 的癌症分类项目：使用 sklearn 管道（标准化 + 分类器）、Optuna 调参、SHAP 做特征解释，并提供基于 Dash 的交互式可视化仪表盘。

---

## 1. How to Run

### 本地运行（推荐先本地验证）

```bash
cd backend
# 若不存在 .venv 则创建虚拟环境（Linux/macOS）
[ -d .venv ] || python -m venv .venv
# Windows CMD: if not exist .venv python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

可选参数示例：

```bash
python main.py --classifier svc --n_trials 30 --output-dir output
python main.py --classifier random_forest --n_trials 20
```

### 运行交互式仪表盘（推荐）

```bash
cd backend
pip install -r requirements.txt
python dashboard_app.py
```

然后在浏览器中访问 `http://localhost:8050` 即可打开交互式可视化仪表盘。

仪表盘功能包括：
- **模型性能**：准确率、精确率、召回率、F1分数仪表盘，混淆矩阵，ROC曲线
- **特征重要性**：SHAP摘要图（beeswarm）、SHAP条形图
- **调参过程**：Optuna超参数优化历史可视化
- **数据探索**：特征分布图、箱线图、平行坐标图
- **单样本解释**：单样本预测解释瀑布图、概率仪表盘

### Docker 运行

在项目根目录执行：

```bash
docker-compose up --build -d
```

容器会执行一次完整流程（加载数据 → Optuna 调参 → 训练 → SHAP 解释），结果写入 `backend/output/`（通过卷挂载）。查看结果：

```bash
ls backend/output/
# 预期: metrics.json, classification_report.txt, shap_summary.png, shap_bar.png, feature_importance_order.json
```

如需在 ARM 环境（如苹果电脑）验证镜像：

```bash
docker pull --platform linux/arm64 python:3.11-slim
docker-compose build
docker-compose up -d
```

---

## 2. Services

| 服务   | 说明 |
|--------|------|
| backend | Python 批处理：癌症数据集加载、管道建模、Optuna 调参、SHAP 特征解释，无对外端口，结果写入 `backend/output/`。 |

本项目无前端，无 8081/8082 端口映射。

---

## 3. 测试账号

不涉及登录与用户系统，无需测试账号。直接运行 `python main.py` 或 `docker-compose up --build -d` 即可。

---

## 4. 题目内容

针对癌症数据集，使用 sklearn 算法进行预测，将数据集分为若干类，要求采用管道化思想，数据进行标准化操作，使用 Optuna 进行调参，SHAP 模块进行模型特征解释。

---

## 项目结构

```
.
├── README.md                 # 项目说明与运行方式
├── docker-compose.yml        # 编排 backend 服务与卷挂载
├── .gitignore                # 忽略 .venv、output、__pycache__ 等
├── docs/
│   └── project_design.md     # 系统架构、数据流、模块与接口清单
└── backend/
    ├── Dockerfile             # 基于 python:3.11-slim 的镜像构建
    ├── requirements.txt      # 依赖：scikit-learn、optuna、shap、matplotlib、dash、plotly 等
    ├── main.py               # 入口：加载数据 → Optuna 调参 → 训练 → 评估 → SHAP 解释
    ├── dashboard_app.py      # 交互式可视化仪表盘入口
    ├── output/                # 运行后生成（metrics.json、SHAP 图、分类报告等）
    └── src/
        ├── __init__.py
        ├── data/              # 数据加载
        │   ├── __init__.py
        │   └── load_data.py   # load_breast_cancer + train_test_split
        ├── pipeline/          # 管道化模型
        │   ├── __init__.py
        │   └── model_pipeline.py   # StandardScaler + SVC/RandomForest 管道
        ├── tuning/            # Optuna 调参
        │   ├── __init__.py
        │   └── optuna_tune.py # 超参搜索与交叉验证
        ├── interpretation/    # SHAP 可解释性
        │   ├── __init__.py
        │   └── shap_explain.py    # TreeExplainer/KernelExplainer、摘要图与条形图
        └── dashboard/         # 仪表盘模块
            ├── __init__.py
            ├── data_utils.py  # 数据管理与模型集成
            └── plots.py       # 可视化图表生成
```

## 技术栈

- Python 3.10+
- scikit-learn（管道、StandardScaler、SVC/RandomForest）
- Optuna（超参优化）
- SHAP（特征可解释性）
- Dash（交互式Web仪表盘）
- Plotly（交互式可视化图表）
- dash-bootstrap-components（Bootstrap样式组件）

数据源为 `sklearn.datasets.load_breast_cancer`（威斯康星乳腺癌二分类，30 维特征）。
