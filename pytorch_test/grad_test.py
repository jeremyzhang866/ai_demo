import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的线性模型
model = nn.Linear(2, 1)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 打印模型的初始参数
print("初始模型参数：")
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")

# 创建输入和目标输出
inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
targets = torch.tensor([[1.0], [2.0]])

# 前向传播
outputs = model(inputs)
loss = criterion(outputs, targets)

# 反向传播
optimizer.zero_grad()
loss.backward()

# 打印梯度
print("\n梯度：")
for name, param in model.named_parameters():
    print(f"{name}.grad: {param.grad}")

# 更新参数
optimizer.step()

# 打印更新后的参数
print("\n更新后的模型参数：")
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")

# 查看优化器的状态字典
print("\n优化器状态字典：")
print(optimizer.state_dict())