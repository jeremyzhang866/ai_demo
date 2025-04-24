import torch
import torch.nn as nn


# 定义一个简单的包含Layer Normalization的神经网络
class NetWithLayerNorm(nn.Module):
    def __init__(self):
        super(NetWithLayerNorm, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.ln1 = nn.LayerNorm(50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 模拟输入数据，假设是一批特征向量，形状为(batch_size, feature_dim)
input_data = torch.randn(20, 100)
model = NetWithLayerNorm()
output = model(input_data)
print(output.shape)