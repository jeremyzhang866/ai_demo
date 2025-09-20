#!/usr/bin/env python3
import time
import argparse

import torch

def busy_work(device: torch.device, size: int, work_time_ratio: float):
    """
    在指定 device 上用 size×size 的矩阵乘法模拟工作，
    并根据 work_time_ratio 控制忙/闲比例。
    """
    # 预分配固定大小的矩阵
    A = torch.randn((size, size), device=device)
    B = torch.randn((size, size), device=device)

    # 计算一次乘法所需时间（warm-up）
    # 用于估算实际执行一次 mm 的平均耗时
    torch.cuda.synchronize(device)
    t0 = time.time()
    C = A @ B
    torch.cuda.synchronize(device)
    single_run = time.time() - t0

    # 设定总周期 T，使单次 busy 时间 = work_time_ratio * T
    # 取 T = single_run / work_time_ratio
    T = single_run / work_time_ratio
    busy_time = work_time_ratio * T
    idle_time = T - busy_time

    print(f"Warm-up matrix size={size}, single_run={single_run:.4f}s, "
          f"target busy={busy_time:.4f}s, idle={idle_time:.4f}s")

    # 无限循环：忙碌 + 睡眠
    while True:
        t_start = time.time()
        # 忙碌阶段：重复做矩阵乘直到累积时间 >= busy_time
        while time.time() - t_start < busy_time:
            _ = A @ B
        torch.cuda.synchronize(device)

        # 空闲阶段
        time.sleep(idle_time)

def main():
    parser = argparse.ArgumentParser(
        description="保持 GPU 占用率在指定比例的 demo 脚本")
    parser.add_argument(
        "--gpu", type=int, default=0,
        help="要加载的 GPU 编号 (默认 0)")
    parser.add_argument(
        "--size", type=int, default=1024,
        help="矩阵规模 (size×size), 默认 1024")
    parser.add_argument(
        "--ratio", type=float, default=0.1,
        help="目标忙碌比例 (0~1), 默认 0.1")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("找不到可用的 CUDA GPU")
    device = torch.device(f"cuda:{args.gpu}")
    print(f"Using device {device}, target GPU busy ratio: {args.ratio*100:.1f}%")
    busy_work(device, args.size, args.ratio)

if __name__ == "__main__":
    main()