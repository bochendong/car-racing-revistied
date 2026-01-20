# 实验记录系统使用指南

## 概述

实验记录系统会自动保存所有训练和测试数据，便于管理大量实验和快速验证想法。

## 目录结构

每次运行训练，系统会自动创建如下目录结构：

```
experiments/
└── dann_20240101_120000/          # 实验目录（自动命名：方法_日期_时间）
    ├── config.json                 # 实验配置（超参数等）
    ├── train/                      # 训练数据目录
    │   ├── training_ep0100.json    # 每100个episode保存一次
    │   ├── training_ep0200.json
    │   └── training_final.json     # 最终训练数据
    ├── test/                       # 测试数据目录（JSON格式）
    │   ├── test_ep0015.json        # 每15个episode保存一次
    │   ├── test_ep0030.json
    │   └── test_final.json         # 最终测试数据
    ├── checkpoints/                # 模型检查点
    │   ├── checkpoint_ep0100.pth   # 每100个episode保存一次
    │   └── checkpoint_final.pth    # 最终检查点
    └── logs/                       # 日志文件（预留）
```

## 保存的数据

### 训练数据（train/）

每个训练数据文件包含：
- `episodes`: 每个 episode 的详细数据
  - episode 编号
  - score（得分）
  - running_score（移动平均得分）
  - source_domain_acc（源域分类准确率，仅 DANN）
  - target_domain_acc（目标域分类准确率，仅 DANN）
  - c1_score, c2_score（目标域测试得分）
  - timestamp（时间戳）
- `training_records`: 所有 episode 的得分列表
- `running_scores`: 所有 episode 的移动平均得分列表
- `losses`: 损失值列表（如果记录）

### 测试数据（test/）

每个测试数据文件包含：
- `test_results`: 测试结果列表
  - `test_name`: 测试名称（如 "episode_15"）
  - `timestamp`: 时间戳
  - `results`: 测试结果
    - `source_env_score`: 源域得分
    - `target_env_c1_score`: 目标域 c1 得分
    - `target_env_c2_score`: 目标域 c2 得分
    - `mean_target_score`: 平均目标域得分
    - `running_score`: 运行平均得分
  - `metadata`: 额外元数据（episode, eta 等）

## 使用方法

### 1. 运行训练

训练代码已自动集成实验记录系统，只需运行 notebook 即可：

```python
# 在训练开始前，系统会自动创建实验记录器
from experiment_logger import ExperimentLogger
from datetime import datetime

experiment_name = f"dann_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
logger = ExperimentLogger(experiment_name=experiment_name, base_dir="./experiments")
```

### 2. 查看实验

使用 `view_experiments.ipynb` 查看所有实验：

```python
from experiment_manager import ExperimentManager

manager = ExperimentManager(base_dir="./experiments")
experiments = manager.list_experiments()

# 查看单个实验
info = manager.get_experiment_info(experiments[0])

# 对比多个实验
comparison = manager.compare_experiments(["dann_exp1", "baseline_exp1"])
```

### 3. 加载检查点

```python
from experiment_logger import ExperimentLogger

logger = ExperimentLogger("dann_20240101_120000", base_dir="./experiments")
episode = logger.load_checkpoint(
    "./experiments/dann_20240101_120000/checkpoints/checkpoint_ep0100.pth",
    model=net,
    optimizer=optimizer
)
```

## 实验命名建议

为了便于管理大量实验，建议使用有意义的命名：

- `dann_lr1e4_eta02` - DANN，学习率 1e-4，eta 0.2
- `baseline_lr1e4` - Baseline，学习率 1e-4
- `dann_exp1` - DANN 实验 1
- `baseline_exp1` - Baseline 实验 1

## 快速验证想法

实验记录系统支持快速迭代：

1. **修改超参数** → 运行训练 → 自动保存
2. **查看结果** → 使用 `view_experiments.ipynb` 对比
3. **分析数据** → 读取 JSON 文件进行深入分析
4. **继续实验** → 基于结果调整参数

## 数据格式示例

### 训练数据示例

```json
{
  "experiment_name": "dann_20240101_120000",
  "start_time": "2024-01-01T12:00:00",
  "episodes": [
    {
      "episode": 15,
      "score": 100.5,
      "running_score": 50.2,
      "c1_score": 80.3,
      "c2_score": 75.2,
      "timestamp": "2024-01-01T12:05:00"
    }
  ],
  "training_records": [100.5, 120.3, ...],
  "running_scores": [50.2, 55.1, ...]
}
```

### 测试数据示例

```json
{
  "experiment_name": "dann_20240101_120000",
  "test_results": [
    {
      "test_name": "episode_15",
      "timestamp": "2024-01-01T12:05:00",
      "results": {
        "source_env_score": 100.5,
        "target_env_c1_score": 80.3,
        "target_env_c2_score": 75.2,
        "mean_target_score": 77.75,
        "running_score": 50.2
      },
      "metadata": {
        "episode": 15,
        "eta": 0.1
      }
    }
  ]
}
```

## 注意事项

- 所有数据以 JSON 格式保存，便于后续分析
- 测试数据每 15 个 episode 保存一次
- 训练数据和检查点每 100 个 episode 保存一次
- 训练结束时自动保存最终数据
- 实验目录会自动创建，无需手动管理
