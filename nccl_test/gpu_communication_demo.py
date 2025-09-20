#!/usr/bin/env python3
"""
4张GPU卡通信原语示例代码
展示基本的GPU间数据传输操作，包含设备初始化、数据分配、点对点通信和同步机制
"""

import os
import sys
import time
import logging
import argparse
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed import ReduceOp

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [Rank %(rank)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class GPUCommunicator:
    """GPU通信管理器"""
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')
        self.logger = logging.getLogger(f'Rank-{rank}')
        
        # 为日志添加rank信息
        for handler in self.logger.handlers:
            handler.setFormatter(logging.Formatter(
                f'[%(asctime)s] [Rank {rank}] [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
    
    def log_info(self, message: str):
        """带rank信息的日志输出"""
        print(f"[Rank {self.rank}] {message}")
        sys.stdout.flush()
    
    def initialize_device(self):
        """初始化GPU设备"""
        self.log_info(f"初始化GPU设备: {self.device}")
        torch.cuda.set_device(self.rank)
        
        # 检查GPU可用性
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用")
        
        gpu_name = torch.cuda.get_device_name(self.rank)
        gpu_memory = torch.cuda.get_device_properties(self.rank).total_memory / 1024**3
        self.log_info(f"GPU信息: {gpu_name}, 显存: {gpu_memory:.2f}GB")
        
        # 清空GPU缓存
        torch.cuda.empty_cache()
        self.log_info("GPU设备初始化完成")
    
    def create_test_tensors(self, size: int = 1024) -> dict:
        """创建测试用的张量数据"""
        self.log_info(f"创建测试张量，大小: {size}x{size}")
        
        tensors = {
            'data': torch.randn(size, size, device=self.device, dtype=torch.float32),
            'identity': torch.eye(size, device=self.device, dtype=torch.float32),
            'rank_tensor': torch.full((size, size), self.rank, device=self.device, dtype=torch.float32),
            'sequence': torch.arange(size * size, device=self.device, dtype=torch.float32).reshape(size, size)
        }
        
        for name, tensor in tensors.items():
            self.log_info(f"张量 '{name}': shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
            self.log_info(f"张量 '{name}' 统计: mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}")
        
        return tensors
    
    def point_to_point_communication(self, tensors: dict):
        """点对点通信示例"""
        self.log_info("=== 开始点对点通信测试 ===")
        
        # 创建通信用的张量
        send_tensor = tensors['rank_tensor'].clone()
        recv_tensor = torch.zeros_like(send_tensor)
        
        if self.rank == 0:
            # Rank 0 发送数据到 Rank 1
            target_rank = 1
            self.log_info(f"发送张量到 Rank {target_rank}")
            self.log_info(f"发送数据统计: mean={send_tensor.mean().item():.4f}")
            
            dist.send(send_tensor, dst=target_rank)
            self.log_info(f"数据发送完成到 Rank {target_rank}")
            
        elif self.rank == 1:
            # Rank 1 接收来自 Rank 0 的数据
            source_rank = 0
            self.log_info(f"等待接收来自 Rank {source_rank} 的数据")
            
            dist.recv(recv_tensor, src=source_rank)
            self.log_info(f"数据接收完成，来自 Rank {source_rank}")
            self.log_info(f"接收数据统计: mean={recv_tensor.mean().item():.4f}")
            
            # 验证数据正确性
            expected_value = float(source_rank)
            if torch.allclose(recv_tensor, torch.full_like(recv_tensor, expected_value)):
                self.log_info("✓ 点对点通信数据验证成功")
            else:
                self.log_info("✗ 点对点通信数据验证失败")
        
        # 同步所有进程
        dist.barrier()
        self.log_info("点对点通信测试完成")
    
    def all_reduce_communication(self, tensors: dict):
        """All-Reduce通信示例"""
        self.log_info("=== 开始All-Reduce通信测试 ===")
        
        # 使用rank_tensor进行all-reduce
        tensor = tensors['rank_tensor'].clone()
        original_mean = tensor.mean().item()
        self.log_info(f"All-Reduce前: mean={original_mean:.4f}")
        
        # 执行all-reduce求和
        dist.all_reduce(tensor, op=ReduceOp.SUM)
        
        reduced_mean = tensor.mean().item()
        expected_mean = sum(range(self.world_size))  # 0+1+2+3 = 6
        self.log_info(f"All-Reduce后: mean={reduced_mean:.4f}, 期望值={expected_mean:.4f}")
        
        # 验证结果
        if abs(reduced_mean - expected_mean) < 1e-6:
            self.log_info("✓ All-Reduce通信验证成功")
        else:
            self.log_info("✗ All-Reduce通信验证失败")
        
        dist.barrier()
        self.log_info("All-Reduce通信测试完成")
    
    def all_gather_communication(self, tensors: dict):
        """All-Gather通信示例"""
        self.log_info("=== 开始All-Gather通信测试 ===")
        
        # 每个rank贡献一个小张量
        local_tensor = torch.full((256, 256), self.rank, device=self.device, dtype=torch.float32)
        self.log_info(f"本地张量: shape={local_tensor.shape}, value={self.rank}")
        
        # 准备接收所有rank的数据
        gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(self.world_size)]
        
        # 执行all-gather
        dist.all_gather(gathered_tensors, local_tensor)
        
        # 验证收集到的数据
        self.log_info("All-Gather结果验证:")
        for i, tensor in enumerate(gathered_tensors):
            expected_value = float(i)
            actual_value = tensor.mean().item()
            self.log_info(f"  来自Rank {i}: mean={actual_value:.4f}, 期望={expected_value:.4f}")
            
            if abs(actual_value - expected_value) < 1e-6:
                self.log_info(f"  ✓ Rank {i} 数据验证成功")
            else:
                self.log_info(f"  ✗ Rank {i} 数据验证失败")
        
        dist.barrier()
        self.log_info("All-Gather通信测试完成")
    
    def broadcast_communication(self, tensors: dict):
        """Broadcast通信示例"""
        self.log_info("=== 开始Broadcast通信测试 ===")
        
        # 使用Rank 0作为广播源
        broadcast_tensor = torch.zeros(512, 512, device=self.device, dtype=torch.float32)
        
        if self.rank == 0:
            # Rank 0 初始化广播数据
            broadcast_tensor.fill_(99.99)
            self.log_info(f"广播源数据: mean={broadcast_tensor.mean().item():.4f}")
        else:
            self.log_info(f"广播前数据: mean={broadcast_tensor.mean().item():.4f}")
        
        # 执行广播
        dist.broadcast(broadcast_tensor, src=0)
        
        # 验证广播结果
        received_mean = broadcast_tensor.mean().item()
        expected_value = 99.99
        self.log_info(f"广播后数据: mean={received_mean:.4f}, 期望={expected_value:.4f}")
        
        if abs(received_mean - expected_value) < 1e-6:
            self.log_info("✓ Broadcast通信验证成功")
        else:
            self.log_info("✗ Broadcast通信验证失败")
        
        dist.barrier()
        self.log_info("Broadcast通信测试完成")
    
    def reduce_scatter_communication(self, tensors: dict):
        """Reduce-Scatter通信示例"""
        self.log_info("=== 开始Reduce-Scatter通信测试 ===")
        
        # 创建输入张量列表，每个rank一个
        input_tensors = []
        for i in range(self.world_size):
            tensor = torch.full((256, 256), i + self.rank, device=self.device, dtype=torch.float32)
            input_tensors.append(tensor)
            self.log_info(f"输入张量 {i}: mean={tensor.mean().item():.4f}")
        
        # 准备输出张量
        output_tensor = torch.zeros(256, 256, device=self.device, dtype=torch.float32)
        
        # 执行reduce-scatter
        dist.reduce_scatter(output_tensor, input_tensors, op=ReduceOp.SUM)
        
        # 验证结果
        result_mean = output_tensor.mean().item()
        # 期望值计算：每个位置应该是所有rank对应位置的和
        expected_value = sum(i + self.rank for i in range(self.world_size))
        self.log_info(f"Reduce-Scatter结果: mean={result_mean:.4f}, 期望={expected_value:.4f}")
        
        if abs(result_mean - expected_value) < 1e-6:
            self.log_info("✓ Reduce-Scatter通信验证成功")
        else:
            self.log_info("✗ Reduce-Scatter通信验证失败")
        
        dist.barrier()
        self.log_info("Reduce-Scatter通信测试完成")
    
    def synchronization_test(self):
        """同步机制测试"""
        self.log_info("=== 开始同步机制测试 ===")
        
        # 模拟不同的工作负载
        work_time = (self.rank + 1) * 0.5  # 不同rank有不同的工作时间
        self.log_info(f"模拟工作负载，耗时: {work_time:.1f}秒")
        
        start_time = time.time()
        time.sleep(work_time)
        work_end_time = time.time()
        
        self.log_info(f"工作完成，实际耗时: {work_end_time - start_time:.2f}秒")
        self.log_info("等待其他rank完成...")
        
        # 同步所有进程
        barrier_start = time.time()
        dist.barrier()
        barrier_end = time.time()
        
        self.log_info(f"同步完成，barrier耗时: {barrier_end - barrier_start:.4f}秒")
        self.log_info("所有rank已同步")
    
    def memory_usage_report(self):
        """内存使用情况报告"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            self.log_info(f"GPU内存使用: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """设置分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    print(f"[Rank {rank}] 初始化分布式环境...")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    print(f"[Rank {rank}] 分布式环境初始化完成")


def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()


def worker_process(rank: int, world_size: int):
    """工作进程主函数"""
    try:
        # 设置分布式环境
        setup_distributed(rank, world_size)
        
        # 创建通信管理器
        communicator = GPUCommunicator(rank, world_size)
        
        # 1. 设备初始化
        communicator.log_info("=" * 60)
        communicator.log_info("开始GPU通信原语演示")
        communicator.log_info("=" * 60)
        communicator.initialize_device()
        
        # 2. 数据分配
        tensors = communicator.create_test_tensors(size=512)
        communicator.memory_usage_report()
        
        # 3. 基本通信操作
        communicator.point_to_point_communication(tensors)
        communicator.all_reduce_communication(tensors)
        communicator.all_gather_communication(tensors)
        communicator.broadcast_communication(tensors)
        communicator.reduce_scatter_communication(tensors)
        
        # 4. 同步机制
        communicator.synchronization_test()
        
        # 最终内存报告
        communicator.memory_usage_report()
        
        communicator.log_info("=" * 60)
        communicator.log_info("GPU通信原语演示完成")
        communicator.log_info("=" * 60)
        
    except Exception as e:
        print(f"[Rank {rank}] 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        cleanup_distributed()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='4卡GPU通信原语演示')
    parser.add_argument('--world-size', type=int, default=4, help='GPU数量')
    parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo'], help='通信后端')
    args = parser.parse_args()
    
    # 检查GPU数量
    if not torch.cuda.is_available():
        print("错误: CUDA不可用")
        return
    
    gpu_count = torch.cuda.device_count()
    if gpu_count < args.world_size:
        print(f"错误: 需要{args.world_size}张GPU，但只检测到{gpu_count}张")
        return
    
    print(f"检测到{gpu_count}张GPU，使用{args.world_size}张进行通信演示")
    print(f"通信后端: {args.backend}")
    print("=" * 60)
    
    # 启动多进程
    mp.spawn(
        worker_process,
        args=(args.world_size,),
        nprocs=args.world_size,
        join=True
    )
    
    print("=" * 60)
    print("所有进程已完成")


if __name__ == '__main__':
    main()