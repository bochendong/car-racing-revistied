# Car Racing Domain Adaptation with DANN

使用 DANN (Domain Adversarial Neural Network) 进行强化学习的领域适应项目，让在绿色背景（源域）训练的智能体能够适应到不同背景颜色（目标域）的 CarRacing 环境。

## 📁 项目结构

```
car-racing-revistied/
├── src/                           # 源代码目录
│   ├── models/                    # 模型定义
│   │   ├── __init__.py
│   │   ├── dann_model.py         # DANN 模型
│   │   └── baseline_model.py     # Baseline 模型
│   ├── agents/                    # 智能体实现
│   │   ├── __init__.py
│   │   ├── agent.py              # DANN Agent
│   │   └── baseline_agent.py     # Baseline Agent
│   └── utils/                     # 工具函数
│       ├── __init__.py
│       ├── environment.py       # 环境封装
│       ├── utils.py              # 工具函数
│       ├── experiment_logger.py  # 实验记录器
│       └── experiment_manager.py # 实验管理器
├── notebooks/                     # Jupyter Notebooks
│   ├── car-racing-dann-clean.ipynb    # DANN 训练（推荐）
│   ├── car-racing-baseline.ipynb      # Baseline 训练
│   └── view_experiments.ipynb         # 实验查看和管理
├── docs/                          # 文档
│   ├── README.md                  # 项目说明（本文件）
│   ├── EXPERIMENT_GUIDE.md        # 实验记录系统使用指南
│   └── COLAB_SETUP.md             # Colab 设置指南
├── experiments/                   # 实验数据（运行时自动创建）
│   └── dann_20240101_120000/
│       ├── config.json
│       ├── train/
│       ├── test/
│       └── checkpoints/
└── output_r/                      # 输出图像（运行时自动创建）
```

## 🚀 快速开始

### 在 Colab 中使用

1. **打开训练 Notebook**
   - DANN 方法：`notebooks/car-racing-dann-clean.ipynb`
   - Baseline 方法：`notebooks/car-racing-baseline.ipynb`

2. **运行第一个 Cell**（自动克隆仓库并导入模块）

3. **按顺序运行所有 Cells**

### 本地使用

```bash
# 克隆仓库
git clone https://github.com/bochendong/car-racing-revistied.git
cd car-racing-revistied

# 安装依赖
pip install torch gym[box2d] matplotlib numpy

# 运行 notebook
jupyter notebook notebooks/car-racing-dann-clean.ipynb
```

## 📦 模块说明

### 模型 (src/models/)
- **DANN**: Domain Adversarial Neural Network 模型（带域分类器）
- **BaselineModel**: Baseline 模型（无域适应）

### 智能体 (src/agents/)
- **Agent**: DANN PPO 智能体（支持域适应）
- **BaselineAgent**: Baseline PPO 智能体（无域适应）

### 工具 (src/utils/)
- **Env**: CarRacing 环境封装，支持不同背景颜色
- **get_random_buffer**: 生成随机背景颜色（避免重复）
- **eval**: 评估智能体性能
- **ExperimentLogger**: 实验记录器（自动保存训练和测试数据）
- **ExperimentManager**: 实验管理器（查看和对比实验）

## 📊 实验记录系统

所有训练和测试数据会自动保存到 `experiments/` 目录：

- **训练数据**: 每 100 个 episode 保存一次（JSON 格式）
- **测试数据**: 每 15 个 episode 保存一次（JSON 格式）
- **检查点**: 每 100 个 episode 保存一次

使用 `notebooks/view_experiments.ipynb` 可以查看和对比所有实验。

详细说明请参考 `docs/EXPERIMENT_GUIDE.md`。

## 🔧 环境设置

### Colab 环境

运行 notebook 中的环境设置 cell，会自动安装所有依赖。

### 本地环境

```bash
pip install torch torchvision
pip install gym[box2d]
pip install matplotlib numpy
```

## 📝 使用方法

### 1. 训练 DANN 模型

```python
from src.utils import Env, ExperimentLogger
from src.models import DANN
from src.agents import Agent

# 创建环境和模型
source_env = Env(color='g', seed=0)
net = DANN(num_out=2).double().to(device)
agent = Agent(net=net, ...)

# 训练（详见 notebook）
```

### 2. 训练 Baseline 模型

```python
from src.models import BaselineModel
from src.agents import BaselineAgent

net = BaselineModel().double().to(device)
agent = BaselineAgent(net=net, ...)
```

### 3. 查看实验

```python
from src.utils import ExperimentManager

manager = ExperimentManager(base_dir="./experiments")
experiments = manager.list_experiments()
comparison = manager.compare_experiments(["dann_exp1", "baseline_exp1"])
```

## 🎯 技术栈

- **PyTorch**: 深度学习框架
- **Gym**: CarRacing-v2 环境
- **DANN**: Domain Adversarial Neural Network
- **PPO**: Proximal Policy Optimization

## 📚 文档

- `docs/README.md`: 项目说明（本文件）
- `docs/EXPERIMENT_GUIDE.md`: 实验记录系统详细指南
- `docs/COLAB_SETUP.md`: Colab 设置说明

## ⚠️ 注意事项

- 确保在 Colab 中启用 GPU 加速
- 训练时间较长，建议使用 Colab Pro 或本地 GPU
- 所有实验数据保存在 `experiments/` 目录
- 输出图像保存在 `output_r/` 目录

## 📄 许可证

MIT License
