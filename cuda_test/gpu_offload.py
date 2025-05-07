import torch
import psutil
import os

def print_cpu_memory():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 3)  # 转换为 GB
    print(f"CPU 内存使用: {mem:.2f} GB")

def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"GPU 已分配内存: {allocated:.2f} GB")
        print(f"GPU 已保留内存: {reserved:.2f} GB")
    else:
        print("未检测到可用的 GPU。")

# 设置张量大小以接近 5GB
# float32 类型的元素占用 4 字节
# 5GB / 4B = 1.25 * 10^9 个元素
num_elements = int(5 * 1024 ** 3 / 4)  # 约为 1.34 * 10^9
tensor_shape = (num_elements,)

print("=== 创建张量前 ===")
print_cpu_memory()
print_gpu_memory()

# 在 CPU 上创建张量
cpu_tensor = torch.ones(tensor_shape, dtype=torch.float32)
print("\n=== 创建张量后（在 CPU 上） ===")
print_cpu_memory()
print_gpu_memory()

# 将张量移动到 GPU
if torch.cuda.is_available():
    gpu_tensor = cpu_tensor.to("cuda")
    print("\n=== 张量移动到 GPU 后 ===")
    print_cpu_memory()
    print_gpu_memory()

    # 将张量移回 CPU
    cpu_tensor = gpu_tensor.to("cpu")
    print("\n=== 张量移回 CPU 后 ===")
    print_cpu_memory()
    print_gpu_memory()

    # 清理 GPU 内存
    del gpu_tensor
    torch.cuda.empty_cache()
    print("\n=== 清理 GPU 内存后 ===")
    print_cpu_memory()
    print_gpu_memory()
else:
    print("\n未检测到可用的 GPU，无法执行 GPU 操作。")