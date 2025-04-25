import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def demo_all_reduce(rank: int, world_size: int):
    setup(rank, world_size)

    tensor = torch.ones(1).cuda(rank) * (rank + 1)
    print(f"[{rank}] 初始 tensor: {tensor.item()}")

    # 所有 GPU 的 tensor 相加（in-place）
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"[{rank}] all_reduce 后 tensor: {tensor.item()}")

    cleanup()


def run_demo():
    world_size = torch.cuda.device_count()
    mp.spawn(demo_all_reduce, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    run_demo()