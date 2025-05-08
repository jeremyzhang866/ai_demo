import torch
import time
import os

# 选择目标 GPU（如 GPU 0）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")

# 创建一个小张量，控制运算量
a = torch.randn(1024, 1024, device=device)
b = torch.randn(1024, 1024, device=device)

print("开始维持 GPU 使用率约 10%，按 Ctrl+C 退出...")

try:
    while True:
        # 执行一次小矩阵乘法
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        # 每秒执行一次计算：休眠时间调节占用率（越短占用越高）
        time.sleep(0.9)  # 调节这个值控制利用率
except KeyboardInterrupt:
    print("退出脚本。")