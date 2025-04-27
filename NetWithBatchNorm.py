import torch
import torch.nn as nn


# 定义一个简单的包含Batch Normalization的神经网络
class NetWithBatchNorm(nn.Module):
    def __init__(self):
        super(NetWithBatchNorm, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(16 * 64 * 64, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out.view(-1, 16 * 64 * 64)
        out = self.fc1(out)
        return out


# 模拟输入数据，假设是一批图像数据，形状为(batch_size, channels, height, width)
input_data = torch.randn(10, 3, 64, 64)
model = NetWithBatchNorm()
output = model(input_data)
print(output.shape)