# 项目结构说明

## 📁 目录结构

```
car-racing-revistied/
├── src/                           # 源代码目录（所有 Python 模块）
│   ├── __init__.py
│   ├── models/                    # 模型定义
│   │   ├── __init__.py
│   │   ├── dann_model.py         # DANN 模型（带域分类器）
│   │   └── baseline_model.py     # Baseline 模型（无域适应）
│   ├── agents/                    # 智能体实现
│   │   ├── __init__.py
│   │   ├── agent.py              # DANN PPO Agent
│   │   └── baseline_agent.py     # Baseline PPO Agent
│   └── utils/                     # 工具函数
│       ├── __init__.py
│       ├── environment.py        # CarRacing 环境封装
│       ├── utils.py              # 工具函数（get_random_buffer, eval）
│       ├── experiment_logger.py  # 实验记录器
│       └── experiment_manager.py # 实验管理器
│
├── notebooks/                     # Jupyter Notebooks
│   ├── car-racing-dann-clean.ipynb    # DANN 训练（推荐使用）
│   ├── car-racing-baseline.ipynb      # Baseline 训练
│   └── view_experiments.ipynb         # 实验查看和管理
│
├── docs/                          # 文档目录
│   ├── README.md                  # 详细项目说明
│   ├── EXPERIMENT_GUIDE.md        # 实验记录系统使用指南
│   ├── COLAB_SETUP.md             # Colab 设置指南
│   └── CLEANUP_GUIDE.md           # 清理指南（旧）
│
├── experiments/                   # 实验数据（运行时自动创建）
│   └── dann_20240101_120000/
│       ├── config.json
│       ├── train/                 # 训练数据（JSON）
│       ├── test/                  # 测试数据（JSON）
│       ├── checkpoints/           # 模型检查点
│       └── logs/                  # 日志文件
│
├── output_r/                      # 输出图像（运行时自动创建）
│
├── README.md                       # 项目主 README
├── PROJECT_STRUCTURE.md           # 本文件
└── .gitignore                     # Git 忽略文件
```

## 📦 模块说明

### src/models/
- **dann_model.py**: DANN 模型，包含域分类器
- **baseline_model.py**: Baseline 模型，无域适应

### src/agents/
- **agent.py**: DANN Agent，支持域适应训练
- **baseline_agent.py**: Baseline Agent，只做 PPO 训练

### src/utils/
- **environment.py**: CarRacing 环境封装，支持不同背景颜色
- **utils.py**: 工具函数（get_random_buffer, eval）
- **experiment_logger.py**: 实验记录器，自动保存训练和测试数据
- **experiment_manager.py**: 实验管理器，查看和对比实验

## 🔄 导入方式

所有模块都通过 `src` 包导入：

```python
import sys
sys.path.append('src')

# 从 utils 导入
from src.utils import Env, eval, ExperimentLogger, ExperimentManager

# 从 models 导入
from src.models import DANN, BaselineModel

# 从 agents 导入
from src.agents import Agent, BaselineAgent
```

## 📝 文件说明

### Notebooks
- **car-racing-dann-clean.ipynb**: DANN 方法训练（推荐）
- **car-racing-baseline.ipynb**: Baseline 方法训练
- **view_experiments.ipynb**: 查看和对比所有实验

### 文档
- **README.md**: 项目主说明
- **docs/README.md**: 详细项目说明
- **docs/EXPERIMENT_GUIDE.md**: 实验记录系统使用指南
- **docs/COLAB_SETUP.md**: Colab 设置说明

## 🗑️ 旧文件

以下文件可以删除（已迁移到新结构）：
- `car-racing-dann.ipynb` (旧版本，使用 `notebooks/car-racing-dann-clean.ipynb`)
- `DANN.ipynb` (旧版本)

## ✅ 优势

新的项目结构：
- ✅ 清晰的模块化组织
- ✅ 易于维护和扩展
- ✅ 符合 Python 项目最佳实践
- ✅ 便于代码复用
- ✅ 实验数据自动组织
