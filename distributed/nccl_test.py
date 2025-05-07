import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, world_size):
    # 每个进程绑定到对应 GPU
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    # 初始化通信组
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29500",
        world_size=world_size,
        rank=rank
    )

    # barrier：所有进程同步
    dist.barrier()
    if rank == 0:
        print("=== barrier 同步完成 ===")

    # broadcast：rank=0 的张量广播到所有进程
    tensor = torch.zeros(1, device=device)
    if rank == 0:
        tensor += 42
    dist.broadcast(tensor, src=0)
    print(f"[rank {rank}] broadcast 后 tensor = {tensor.item()}")

    # all_reduce：所有进程的 tensor 求和，再广播回每个进程
    tensor = torch.tensor([rank + 1], device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"[rank {rank}] all_reduce(sum) 后 tensor = {tensor.item()}")

    # reduce：所有进程的 tensor 求和，结果保存在 dst=0
    tensor = torch.tensor([rank + 1], device=device, dtype=torch.float32)
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f"[rank {rank}] reduce(sum -> 0) 后 tensor = {tensor.item()}")

    # scatter：rank=0 准备一个列表，分别发送给各进程
    if rank == 0:
        scatter_list = [torch.tensor([i*10 + 1], device=device) for i in range(world_size)]
    else:
        scatter_list = None
    out_tensor = torch.zeros(1, device=device, dtype=torch.float32)
    dist.scatter(out_tensor, scatter_list=scatter_list, src=0)
    print(f"[rank {rank}] scatter 后 out_tensor = {out_tensor.item()}")

    # gather：各进程发一个 tensor 到 rank=0
    send_tensor = torch.tensor([rank*100 + 1], device=device, dtype=torch.float32)
    if rank == 0:
        gather_list = [torch.zeros(1, device=device) for _ in range(world_size)]
    else:
        gather_list = None
    dist.gather(send_tensor, gather_list=gather_list, dst=0)
    if rank == 0:
        gathered = [t.item() for t in gather_list]
        print(f"[rank {rank}] gather 收到 = {gathered}")

    # all_gather：各进程发 tensor，所有进程都收集到同一个列表
    send_tensor = torch.tensor([rank*1000 + 1], device=device, dtype=torch.float32)
    allgather_list = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(allgather_list, send_tensor)
    print(f"[rank {rank}] all_gather 收到 = {[t.item() for t in allgather_list]}")

    # reduce_scatter：先所有进程准备列表，然后对列表中对应索引元素做 reduce，再分散给各进程
    # 这里先让每个进程构造同样的列表
    chunk = 1
    send_chunks = [torch.tensor([rank + j], device=device, dtype=torch.float32)
                   for j in range(world_size)]
    out_chunk = torch.zeros(1, device=device)
    dist.reduce_scatter(out_chunk, scatter_list=send_chunks, op=dist.ReduceOp.SUM)
    print(f"[rank {rank}] reduce_scatter(sum) 后 out_chunk = {out_chunk.item()}")

    # 结束
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 3
    # 可先设置 MASTER_ADDR、MASTER_PORT 环境变量
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)