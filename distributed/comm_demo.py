# comm_demo.py
import os, torch, torch.distributed as dist, torch.multiprocessing as mp

# ---------- 1. 工具：打印 helper ----------
def info(rank, msg):
    print(f"[Rank {rank}] {msg}", flush=True)

# ---------- 2. 通信原语封装 ----------
def broadcast_tensor(tensor, src=0):
    """把 src rank 的 tensor 广播到所有进程"""
    dist.broadcast(tensor, src)
    return tensor

def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM):
    """对所有进程的 tensor 做 op 并写回本地 tensor"""
    dist.all_reduce(tensor, op)
    return tensor

def reduce_tensor(tensor, dst=0, op=dist.ReduceOp.SUM):
    """把所有进程的 tensor 做 op，结果保留在 dst 进程"""
    dist.reduce(tensor, dst, op)
    return tensor

def scatter_tensor(output_tensor, scatter_list=None, src=0):
    """src 进程把 scatter_list[i] 发送给 rank i"""
    dist.scatter(output_tensor, scatter_list=scatter_list, src=src)
    return output_tensor

def gather_tensor(send_tensor, gather_list=None, dst=0):
    """各进程发送 send_tensor 给 dst，dst 收集到 gather_list"""
    dist.gather(send_tensor, gather_list=gather_list, dst=dst)
    return gather_list if dist.get_rank() == dst else None

def all_gather_tensor(send_tensor):
    """所有进程收集所有进程的 send_tensor"""
    gather_list = [torch.zeros_like(send_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list, send_tensor)
    return gather_list

def reduce_scatter_tensor(output_tensor, input_list, op=dist.ReduceOp.SUM):
    """
    先把 input_list 的同索引元素做 op，再把结果分散给各进程
    input_list 长度必须等于 world_size
    """
    dist.reduce_scatter(output_tensor, input_list, op)
    return output_tensor

def barrier_wait():
    """强制所有进程同步"""
    dist.barrier()

# ---------- 3. 进程入口 ----------
def run(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # --------- 示例数据 ---------
    x = torch.ones(4, device=f"cuda:{rank}") * (rank + 1)

    # 1) broadcast
    y = x.clone()
    broadcast_tensor(y, src=0)
    info(rank, f"after broadcast → {y}")
    

    # 2) all_reduce
    z = x.clone()
    all_reduce_tensor(z)
    info(rank, f"after all_reduce(sum) → {z}")
    

    # 3) reduce
    r = x.clone()
    reduce_tensor(r, dst=0)
    if rank == 0:
        info(rank, f"after reduce(sum→0) → {r}")
    

    # 4) scatter
    # ------- 修正版本 -------
    if rank == 0:
        scatter_list = [
            torch.tensor([i], device=f"cuda:{rank}", dtype=torch.float32)
            for i in range(world_size)
        ]
    else:
        scatter_list = None
    out_scat = torch.zeros(1, device=f"cuda:{rank}", dtype=torch.float32)
    dist.scatter(out_scat, scatter_list=scatter_list, src=0)
    info(rank, f"after scatter → {out_scat}")
    

    # 5) gather
    send = torch.tensor([rank], device=f"cuda:{rank}")
    if rank == 0:
        gather_list = [torch.zeros_like(send) for _ in range(world_size)]
    else:
        gather_list = None
    gather_tensor(send, gather_list, dst=0)
    if rank == 0:
        info(rank, f"after gather → {gather_list}")
    

    # 6) all_gather
    ag = all_gather_tensor(send)
    info(rank, f"after all_gather → {ag}")
    

    # # 7) reduce_scatter
    # rs_out = torch.zeros(1, device=f"cuda:{rank}")
    # rs_inp = [torch.tensor([rank+i], device=f"cuda:{rank}") for i in range(world_size)]
    # reduce_scatter_tensor(rs_out, rs_inp)
    # info(rank, f"after reduce_scatter(sum) → {rs_out}")
    # 

    # 8) barrier
    barrier_wait()
    if rank == 0:
        info(rank, "barrier reached!")
    

    dist.destroy_process_group()

# ---------- 4. 启动 ----------
if __name__ == "__main__":
    world_size = 3
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)