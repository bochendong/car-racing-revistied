import torch
import torch.nn as nn


class BaselineModel(nn.Module):
    """
    Baseline 模型（不使用域适应，只做 PPO）
    与 DANN 模型结构相同，但移除了域分类器
    """
    def __init__(self):
        super(BaselineModel, self).__init__()

        self.sketch = nn.Sequential(
            nn.Conv3d(4, 4, kernel_size=(1, 1, 3), stride=1),
            nn.Tanh(),
        )

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        self.cnn_base = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU()
        )

        self.alpha_head = nn.Sequential(
            nn.Linear(100, 3),
            nn.Softplus()
        )

        self.beta_head = nn.Sequential(
            nn.Linear(100, 3),
            nn.Softplus()
        )

        self.apply(self._weights_init)

    def forward(self, input):
        """
        前向传播（baseline 版本，不包含域分类）
        
        :param input: torch.Tensor, 输入张量
        :return: tuple, (alpha, beta), value, sketch
        """
        sketch = self.sketch(input)
        sketch = torch.squeeze(sketch)

        feature = self.feature_extractor(sketch)
        out = self.cnn_base(feature)
        out = out.view(-1, 256)
        v = self.value_head(out)
        out = self.fc(out)

        alpha = self.alpha_head(out) + 1
        beta = self.beta_head(out) + 1

        return (alpha, beta), v, sketch

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)
