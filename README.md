# Car Racing Domain Adaptation with DANN

使用 DANN (Domain Adversarial Neural Network) 进行强化学习的领域适应项目。

## 📁 项目结构

```
car-racing-revistied/
├── src/                           # 源代码目录（所有 Python 模块）
│   ├── models/                    # 模型定义
│   │   ├── dann_model.py         # DANN 模型
│   │   └── baseline_model.py     # Baseline 模型
│   ├── agents/                    # 智能体实现
│   │   ├── agent.py              # DANN Agent
│   │   └── baseline_agent.py     # Baseline Agent
│   └── utils/                     # 工具函数
│       ├── environment.py        # 环境封装
│       ├── utils.py              # 工具函数
│       ├── experiment_logger.py  # 实验记录器
│       └── experiment_manager.py # 实验管理器
│
├── notebooks/                     # Jupyter Notebooks
│   ├── car-racing-dann-clean.ipynb    # DANN 训练（推荐）
│   ├── car-racing-baseline.ipynb      # Baseline 训练
│   └── view_experiments.ipynb         # 实验查看和管理
│
├── docs/                          # 文档目录
│   ├── README.md                  # 详细项目说明
│   ├── EXPERIMENT_GUIDE.md        # 实验记录系统使用指南
│   └── COLAB_SETUP.md             # Colab 设置指南
│
├── experiments/                   # 实验数据（运行时自动创建）
│   └── dann_20240101_120000/
│       ├── config.json
│       ├── train/                 # 训练数据（JSON）
│       ├── test/                  # 测试数据（JSON）
│       └── checkpoints/           # 模型检查点
│
└── output_r/                      # 输出图像（运行时自动创建）
```

详细结构说明请参考 `PROJECT_STRUCTURE.md`。

## 🚀 快速开始

### 在 Colab 中使用

1. 打开 `notebooks/car-racing-dann-clean.ipynb`（DANN 方法）或 `notebooks/car-racing-baseline.ipynb`（Baseline 方法）
2. 运行第一个 cell 自动克隆仓库并导入模块
3. 按顺序运行所有 cells

### 本地使用

```bash
git clone https://github.com/bochendong/car-racing-revistied.git
cd car-racing-revistied
pip install torch gym[box2d] matplotlib numpy
jupyter notebook notebooks/car-racing-dann-clean.ipynb
```

## 📦 模块导入

所有代码都通过模块化导入：

```python
import sys
sys.path.append('src')

from src.utils import Env, eval, ExperimentLogger
from src.models import DANN, BaselineModel
from src.agents import Agent, BaselineAgent
```

## 📊 实验记录

- 所有训练和测试数据自动保存到 `experiments/` 目录
- 测试数据保存为 JSON 格式
- 使用 `notebooks/view_experiments.ipynb` 查看和对比实验

详细说明请参考 `docs/EXPERIMENT_GUIDE.md`。

## 📚 文档

- `docs/README.md`: 详细项目说明
- `docs/EXPERIMENT_GUIDE.md`: 实验记录系统使用指南
- `docs/COLAB_SETUP.md`: Colab 设置说明

## 📄 许可证

MIT License
