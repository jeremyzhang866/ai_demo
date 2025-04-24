import torch
import torch.nn as nn

# 这是你的模型类（常被称作 model_class）
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

# 实例化模型（这一步有时会写成 model = model_class(...)）
model = MyModel()