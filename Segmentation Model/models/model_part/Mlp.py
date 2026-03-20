# 多层感知机
import torch.nn as nn

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features  # 输入特征的维度
        hidden_features = hidden_features or in_features  # 隐藏特征维度
        self.fc1 = nn.Linear(in_features, hidden_features)  # 线性层
        self.act = act_layer()  # 激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)  # 线性层
        self.drop = nn.Dropout(drop)  # 随机丢弃神经元，丢弃率默认为 0

    # 定义前向传播
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x