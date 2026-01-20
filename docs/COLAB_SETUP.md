# Colab 设置指南

## 快速开始

### 方法 1: 使用 Git Clone（推荐）

1. 在 Colab 中打开 `car-racing-dann.ipynb`
2. 运行第一个 cell（Git Clone）：
```python
!git clone https://github.com/bochendong/car-racing-revistied.git 2>/dev/null || echo "Repository already exists or clone failed"
%cd car-racing-revistied

# 导入模块化代码
from environment import Env
from dann_model import DANN
from agent import Agent
from utils import get_random_buffer, eval

print("✅ 模块导入成功！")
```

3. 初始化模型和智能体时，**必须传入 `device` 参数**：
```python
criterion = nn.CrossEntropyLoss().to(device)
net = DANN(num_out = 2).double().to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-4)
agent = Agent(net = net,  criterion = criterion,  optimizer = optimizer, 
              buffer_capacity = 1024, batch_size = 128, device = device)  # 注意：需要 device 参数
```

### 方法 2: 使用 Notebook 内嵌代码

如果不想使用 git clone，可以：
1. 跳过 git clone 的 cell
2. 运行创建模块文件的 cells（Cell 4-8）
3. 或者直接使用 notebook 中的内嵌代码

## 重要提示

- **使用模块化代码（git clone）时**：Agent 初始化必须传入 `device` 参数
- **使用内嵌代码时**：Agent 初始化可能不需要 `device` 参数（取决于代码版本）
- 所有代码错误已修复，可以直接运行

## 文件结构

```
car-racing-revistied/
├── car-racing-dann.ipynb    # 主训练 notebook
├── environment.py            # 环境封装模块
├── dann_model.py             # DANN 模型定义
├── agent.py                  # PPO Agent 实现（已修复所有错误）
├── utils.py                  # 工具函数
├── README.md                 # 项目说明
└── COLAB_SETUP.md            # 本文件
```

## 常见问题

**Q: 为什么需要 device 参数？**  
A: 模块化的 Agent 类需要知道使用哪个设备（CPU/GPU）来处理张量。

**Q: 如果遇到导入错误怎么办？**  
A: 确保已经运行了 git clone 的 cell，并且当前目录在 `car-racing-revistied` 文件夹中。

**Q: 可以同时使用两种方法吗？**  
A: 不建议。选择一种方法并保持一致。
