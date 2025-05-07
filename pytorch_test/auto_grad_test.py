import torch, psutil, os, gc, time

def cpu_mem_gb() -> float:
    """当前进程占用的 CPU 内存（GB）"""
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3

def gpu_mem_gb() -> float:
    """当前设备 0 已分配显存（GB），无 GPU 时返回 0"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated(0) / 1024**3
    return 0.0

def show(stage: str):
    print(f"{stage:<25s} | CPU {cpu_mem_gb():6.2f} GB | GPU {gpu_mem_gb():6.2f} GB")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype   = torch.float16 if torch.cuda.is_available() else torch.float32   # GPU 用 fp16 省显存

# --- 构造一个占内存稍大的模型和输入 ------------------
model = torch.nn.Sequential(
    *[torch.nn.Linear(4096, 4096, bias=False) for _ in range(12)]
).to(device, dtype=dtype)

x = torch.randn(32, 4096, device=device, dtype=dtype, requires_grad=False)  # batch=32

criterion = torch.nn.MSELoss()

print(">>> 开始演示（单位 GB）")
show("before forward")

# ---------- 前向传播：激活值 & 计算图被保留 ----------
y = model(x)                 # 激活值生成
target = torch.zeros_like(y) # 随便造个目标
loss = criterion(y, target)

show("after forward")

# ---------- 反向传播：要用到激活值 ----------
loss.backward()              # 这里才真正用到前向保存的激活值

show("after backward")

# ---------- 显式删除占用并清理缓存 ----------
del y, loss, target
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect(); time.sleep(1)   # 给 Python / CUDA 一点时间回收

show("after cleanup")