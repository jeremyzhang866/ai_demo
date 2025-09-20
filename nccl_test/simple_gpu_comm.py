#!/usr/bin/env python3
"""
简化版4卡GPU通信原语示例
专注于最基本的通信操作
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def simple_communication_demo(rank, world_size):
    """简单的通信演示"""
    # 设置分布式环境
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    
    # 设置GPU设备
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    print(f"[GPU {rank}] 开始通信演示")
    
    # 创建测试数据
    data = torch.randn(1000, 1000, device=device) * (rank + 1)
    print(f"[GPU {rank}] 初始数据均值: {data.mean().item():.4f}")
    
    # 1. All-Reduce: 所有GPU数据求和
    all_reduce_data = data.clone()
    dist.all_reduce(all_reduce_data)
    print(f"[GPU {rank}] All-Reduce后均值: {all_reduce_data.mean().item():.4f}")
    
    # 2. All-Gather: 收集所有GPU的数据
    gathered_data = [torch.zeros_like(data) for _ in range(world_size)]
    dist.all_gather(gathered_data, data)
    print(f"[GPU {rank}] All-Gather收集到{len(gathered_data)}个张量")
    
    # 3. Broadcast: GPU 0广播数据到其他GPU
    broadcast_data = torch.zeros_like(data)
    if rank == 0:
        broadcast_data.fill_(999.0)
    dist.broadcast(broadcast_data, src=0)
    print(f"[GPU {rank}] Broadcast后数据均值: {broadcast_data.mean().item():.4f}")
    
    # 4. 点对点通信 (GPU 0 -> GPU 1)
    if rank == 0:
        send_data = torch.full((100, 100), 888.0, device=device)
        dist.send(send_data, dst=1)
        print(f"[GPU {rank}] 发送数据到GPU 1")
    elif rank == 1:
        recv_data = torch.zeros(100, 100, device=device)
        dist.recv(recv_data, src=0)
        print(f"[GPU {rank}] 接收数据均值: {recv_data.mean().item():.4f}")
    
    # 同步所有GPU
    dist.barrier()
    print(f"[GPU {rank}] 通信演示完成")
    
    # 清理
    dist.destroy_process_group()


def main():
    """主函数"""
    world_size = 4  # 4张GPU
    
    if torch.cuda.device_count() < world_size:
        print(f"需要{world_size}张GPU，但只有{torch.cuda.device_count()}张")
        return
    
    print("启动4卡GPU通信演示...")
    mp.spawn(simple_communication_demo, args=(world_size,), nprocs=world_size, join=True)
    print("演示完成!")


if __name__ == '__main__':
    main()