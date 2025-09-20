#!/bin/bash

echo "=== GPU通信原语演示脚本 ==="

# 检查CUDA是否可用
python3 -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"

echo ""
echo "选择运行模式:"
echo "1. 完整演示 (详细日志)"
echo "2. 简化演示 (基本功能)"
echo "3. 检查环境"

read -p "请输入选择 (1-3): " choice

case $choice in
    1)
        echo "运行完整演示..."
        python3 gpu_communication_demo.py --world-size 4 --backend nccl
        ;;
    2)
        echo "运行简化演示..."
        python3 simple_gpu_comm.py
        ;;
    3)
        echo "检查环境..."
        python3 -c "
import torch
import torch.distributed as dist
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA版本: {torch.version.cuda}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'GPU数量: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
print(f'NCCL可用: {dist.is_nccl_available()}')
"
        ;;
    *)
        echo "无效选择"
        ;;
esac