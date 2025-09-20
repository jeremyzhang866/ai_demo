import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# 模拟MHATokenToKVPool的缓冲区创建
def create_kv_cache_example():
    # 配置参数
    size = 100  # 简化为100个token位置，便于演示
    page_size = 16
    head_num = 4  # 简化为4个注意力头
    head_dim = 8  # 简化为8维，便于显示
    layer_num = 3  # 简化为3层
    
    # 创建K和V缓冲区
    k_buffer = []
    v_buffer = []
    
    for layer_id in range(layer_num):
        k_tensor = torch.zeros((size + page_size, head_num, head_dim), 
                              dtype=torch.float32, device="cpu")  # 使用float32便于显示
        v_tensor = torch.zeros((size + page_size, head_num, head_dim), 
                              dtype=torch.float32, device="cpu")
        k_buffer.append(k_tensor)
        v_buffer.append(v_tensor)
    
    print(f"创建的KV缓存结构:")
    print(f"- 总层数: {layer_num}")
    print(f"- 每层K缓存形状: {k_buffer[0].shape}")
    print(f"- 每层V缓存形状: {v_buffer[0].shape}")
    print(f"- 总token位置数: {size + page_size}")
    print(f"- 注意力头数: {head_num}")
    print(f"- 每头维度: {head_dim}")
    
    return k_buffer, v_buffer

def visualize_tensor_structure():
    """可视化tensor结构"""
    print("\n" + "="*60)
    print("🔍 KV缓存Tensor结构可视化")
    print("="*60)
    
    # 创建一个小规模示例便于可视化
    layer_num = 3
    token_positions = 12  # 减少位置数便于显示
    head_num = 4
    head_dim = 6  # 减少维度便于显示
    
    k_buffers = []
    v_buffers = []
    
    for layer_id in range(layer_num):
        k_tensor = torch.zeros(token_positions, head_num, head_dim)
        v_tensor = torch.zeros(token_positions, head_num, head_dim)
        k_buffers.append(k_tensor)
        v_buffers.append(v_tensor)
    
    print(f"📊 可视化配置:")
    print(f"   - 层数: {layer_num}")
    print(f"   - Token位置: {token_positions}")
    print(f"   - 注意力头数: {head_num}")
    print(f"   - 每头维度: {head_dim}")
    print(f"   - 每层K缓存形状: {k_buffers[0].shape}")
    print(f"   - 每层V缓存形状: {v_buffers[0].shape}")
    
    return k_buffers, v_buffers, layer_num, token_positions, head_num, head_dim

def simulate_kv_operations():
    """模拟KV缓存的SET和GET操作，并填充实际数据"""
    print("\n" + "="*60)
    print("🚀 模拟KV缓存操作")
    print("="*60)
    
    k_buffers, v_buffers, layer_num, token_positions, head_num, head_dim = visualize_tensor_structure()
    
    # 模拟处理一个batch的数据
    batch_size = 2
    seq_len = 3
    layer_id = 1  # 操作第1层
    
    print(f"\n📝 模拟场景:")
    print(f"   - Batch size: {batch_size}")
    print(f"   - 序列长度: {seq_len}")
    print(f"   - 目标层: {layer_id}")
    
    # 1. 生成模拟的K和V值
    num_tokens = batch_size * seq_len
    torch.manual_seed(42)  # 固定随机种子便于复现
    cache_k = torch.randn(num_tokens, head_num, head_dim) * 0.5 + torch.arange(num_tokens).float().unsqueeze(-1).unsqueeze(-1) * 0.1
    cache_v = torch.randn(num_tokens, head_num, head_dim) * 0.3 + torch.arange(num_tokens).float().unsqueeze(-1).unsqueeze(-1) * 0.2
    
    # 2. 分配存储位置
    loc = torch.tensor([2, 3, 4, 5, 6, 7])  # 连续位置2-7
    
    print(f"\n💾 SET操作:")
    print(f"   - 存储位置: {loc.tolist()}")
    print(f"   - K值形状: {cache_k.shape}")
    print(f"   - V值形状: {cache_v.shape}")
    
    # 3. 执行存储操作
    k_buffers[layer_id][loc] = cache_k
    v_buffers[layer_id][loc] = cache_v
    
    # 4. 显示存储后的数据
    print(f"\n📋 存储后的数据预览:")
    for i, pos in enumerate(loc[:3]):  # 只显示前3个位置
        print(f"   位置{pos}:")
        print(f"     K[头0]: {k_buffers[layer_id][pos, 0, :].round(decimals=2).tolist()}")
        print(f"     V[头0]: {v_buffers[layer_id][pos, 0, :].round(decimals=2).tolist()}")
    
    # 5. 模拟GET操作
    print(f"\n🔍 GET操作:")
    query_positions = torch.tensor([2, 3, 4])
    retrieved_k = k_buffers[layer_id][query_positions]
    retrieved_v = v_buffers[layer_id][query_positions]
    
    print(f"   - 查询位置: {query_positions.tolist()}")
    print(f"   - 检索到的K形状: {retrieved_k.shape}")
    print(f"   - 检索到的V形状: {retrieved_v.shape}")
    
    return k_buffers, v_buffers, loc, layer_id

def create_tensor_heatmap():
    """创建tensor的热力图可视化"""
    print("\n" + "="*60)
    print("🎨 创建Tensor热力图可视化")
    print("="*60)
    
    k_buffers, v_buffers, loc, layer_id = simulate_kv_operations()
    
    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('KV缓存Tensor可视化 - 第1层', fontsize=16, fontweight='bold')
    
    # 1. K缓存热力图 - 所有头的平均值
    k_data = k_buffers[layer_id][:10, :, :].mean(dim=1).numpy()  # 只显示前10个位置
    im1 = axes[0, 0].imshow(k_data, cmap='viridis', aspect='auto')
    axes[0, 0].set_title('K缓存热力图 (所有头平均)', fontweight='bold')
    axes[0, 0].set_xlabel('特征维度')
    axes[0, 0].set_ylabel('Token位置')
    axes[0, 0].set_yticks(range(10))
    axes[0, 0].set_yticklabels([f'位置{i}' for i in range(10)])
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 标记已使用的位置
    for pos in loc:
        if pos < 10:
            axes[0, 0].add_patch(Rectangle((0, pos-0.5), k_data.shape[1], 1, 
                                         fill=False, edgecolor='red', linewidth=2))
    
    # 2. V缓存热力图 - 所有头的平均值
    v_data = v_buffers[layer_id][:10, :, :].mean(dim=1).numpy()
    im2 = axes[0, 1].imshow(v_data, cmap='plasma', aspect='auto')
    axes[0, 1].set_title('V缓存热力图 (所有头平均)', fontweight='bold')
    axes[0, 1].set_xlabel('特征维度')
    axes[0, 1].set_ylabel('Token位置')
    axes[0, 1].set_yticks(range(10))
    axes[0, 1].set_yticklabels([f'位置{i}' for i in range(10)])
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 标记已使用的位置
    for pos in loc:
        if pos < 10:
            axes[0, 1].add_patch(Rectangle((0, pos-0.5), v_data.shape[1], 1, 
                                         fill=False, edgecolor='red', linewidth=2))
    
    # 3. 单个头的K缓存详细视图
    k_head0 = k_buffers[layer_id][:10, 0, :].numpy()
    im3 = axes[1, 0].imshow(k_head0, cmap='coolwarm', aspect='auto')
    axes[1, 0].set_title('K缓存详细视图 (头0)', fontweight='bold')
    axes[1, 0].set_xlabel('特征维度')
    axes[1, 0].set_ylabel('Token位置')
    axes[1, 0].set_yticks(range(10))
    axes[1, 0].set_yticklabels([f'位置{i}' for i in range(10)])
    plt.colorbar(im3, ax=axes[1, 0])
    
    # 4. 单个头的V缓存详细视图
    v_head0 = v_buffers[layer_id][:10, 0, :].numpy()
    im4 = axes[1, 1].imshow(v_head0, cmap='coolwarm', aspect='auto')
    axes[1, 1].set_title('V缓存详细视图 (头0)', fontweight='bold')
    axes[1, 1].set_xlabel('特征维度')
    axes[1, 1].set_ylabel('Token位置')
    axes[1, 1].set_yticks(range(10))
    axes[1, 1].set_yticklabels([f'位置{i}' for i in range(10)])
    plt.colorbar(im4, ax=axes[1, 1])
    
    # 添加图例
    red_patch = mpatches.Patch(color='red', label='已使用位置')
    fig.legend(handles=[red_patch], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('/home/zjm.zhang/zjm_workspace/ai_demo/pytorch_test/kv_cache_heatmap.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 热力图已保存到: kv_cache_heatmap.png")

def create_3d_visualization():
    """创建3D可视化"""
    print("\n" + "="*60)
    print("🌟 创建3D Tensor可视化")
    print("="*60)
    
    k_buffers, v_buffers, loc, layer_id = simulate_kv_operations()
    
    # 创建3D图形
    fig = plt.figure(figsize=(15, 10))
    
    # K缓存3D可视化
    ax1 = fig.add_subplot(121, projection='3d')
    
    # 只显示已使用的位置
    used_positions = loc[:6]  # 前6个位置
    k_data = k_buffers[layer_id][used_positions, :, :].numpy()
    
    # 创建网格
    positions, heads, dims = np.meshgrid(
        range(len(used_positions)), 
        range(k_data.shape[1]), 
        range(k_data.shape[2])
    )
    
    # 绘制3D散点图
    colors = k_data.flatten()
    scatter = ax1.scatter(positions.flatten(), heads.flatten(), dims.flatten(), 
                         c=colors, cmap='viridis', alpha=0.6, s=20)
    
    ax1.set_xlabel('Token位置')
    ax1.set_ylabel('注意力头')
    ax1.set_zlabel('特征维度')
    ax1.set_title('K缓存3D可视化', fontweight='bold')
    
    # V缓存3D可视化
    ax2 = fig.add_subplot(122, projection='3d')
    
    v_data = v_buffers[layer_id][used_positions, :, :].numpy()
    colors_v = v_data.flatten()
    scatter_v = ax2.scatter(positions.flatten(), heads.flatten(), dims.flatten(), 
                           c=colors_v, cmap='plasma', alpha=0.6, s=20)
    
    ax2.set_xlabel('Token位置')
    ax2.set_ylabel('注意力头')
    ax2.set_zlabel('特征维度')
    ax2.set_title('V缓存3D可视化', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/zjm.zhang/zjm_workspace/ai_demo/pytorch_test/kv_cache_3d.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 3D可视化已保存到: kv_cache_3d.png")

def print_detailed_tensor_info():
    """打印详细的tensor信息"""
    print("\n" + "="*60)
    print("📊 详细Tensor信息")
    print("="*60)
    
    k_buffers, v_buffers, loc, layer_id = simulate_kv_operations()
    
    print(f"\n🔍 第{layer_id}层详细信息:")
    print(f"   K缓存tensor:")
    print(f"     - 形状: {k_buffers[layer_id].shape}")
    print(f"     - 数据类型: {k_buffers[layer_id].dtype}")
    print(f"     - 设备: {k_buffers[layer_id].device}")
    print(f"     - 内存占用: {k_buffers[layer_id].numel() * k_buffers[layer_id].element_size()} bytes")
    print(f"     - 非零元素数: {torch.count_nonzero(k_buffers[layer_id])}")
    
    print(f"\n   V缓存tensor:")
    print(f"     - 形状: {v_buffers[layer_id].shape}")
    print(f"     - 数据类型: {v_buffers[layer_id].dtype}")
    print(f"     - 设备: {v_buffers[layer_id].device}")
    print(f"     - 内存占用: {v_buffers[layer_id].numel() * v_buffers[layer_id].element_size()} bytes")
    print(f"     - 非零元素数: {torch.count_nonzero(v_buffers[layer_id])}")
    
    # 显示具体数值
    print(f"\n📋 已使用位置的具体数值:")
    for i, pos in enumerate(loc[:3]):
        print(f"\n   位置{pos} (Token {i}):")
        print(f"     所有头的K值:")
        for head in range(k_buffers[layer_id].shape[1]):
            k_vals = k_buffers[layer_id][pos, head, :].round(decimals=2)
            print(f"       头{head}: {k_vals.tolist()}")
        
        print(f"     所有头的V值:")
        for head in range(v_buffers[layer_id].shape[1]):
            v_vals = v_buffers[layer_id][pos, head, :].round(decimals=2)
            print(f"       头{head}: {v_vals.tolist()}")

def demonstrate_attention_computation():
    """演示attention计算过程"""
    print("\n" + "="*60)
    print("🧠 演示Attention计算过程")
    print("="*60)
    
    k_buffers, v_buffers, loc, layer_id = simulate_kv_operations()
    
    # 模拟一个query
    torch.manual_seed(123)
    query = torch.randn(1, 4, 6) * 0.5  # [1, head_num, head_dim]
    
    print(f"🔍 Query tensor:")
    print(f"   - 形状: {query.shape}")
    print(f"   - 值: {query.round(decimals=2)}")
    
    # 获取已存储的K和V
    used_positions = loc[:4]  # 使用前4个位置
    keys = k_buffers[layer_id][used_positions]    # [4, 4, 6]
    values = v_buffers[layer_id][used_positions]  # [4, 4, 6]
    
    print(f"\n📚 从缓存获取的K和V:")
    print(f"   - K形状: {keys.shape}")
    print(f"   - V形状: {values.shape}")
    
    # 计算attention scores
    # query: [1, 4, 6], keys: [4, 4, 6] -> scores: [1, 4, 4]
    scores = torch.matmul(query, keys.transpose(-2, -1))  # [1, 4, 6] @ [4, 6, 4] -> [1, 4, 4]
    
    print(f"\n🎯 Attention Scores:")
    print(f"   - 形状: {scores.shape}")
    print(f"   - 值 (头0): {scores[0, 0, :].round(decimals=2).tolist()}")
    print(f"   - 值 (头1): {scores[0, 1, :].round(decimals=2).tolist()}")
    
    # 应用softmax
    attention_weights = torch.softmax(scores, dim=-1)
    
    print(f"\n⚖️ Attention Weights (softmax后):")
    print(f"   - 形状: {attention_weights.shape}")
    print(f"   - 权重 (头0): {attention_weights[0, 0, :].round(decimals=3).tolist()}")
    print(f"   - 权重 (头1): {attention_weights[0, 1, :].round(decimals=3).tolist()}")
    
    # 计算最终输出
    # attention_weights: [1, 4, 4], values: [4, 4, 6] -> output: [1, 4, 6]
    output = torch.matmul(attention_weights, values)
    
    print(f"\n🎊 最终输出:")
    print(f"   - 形状: {output.shape}")
    print(f"   - 值 (头0): {output[0, 0, :].round(decimals=2).tolist()}")
    print(f"   - 值 (头1): {output[0, 1, :].round(decimals=2).tolist()}")

# 创建缓存
k_buffer, v_buffer = create_kv_cache_example()

# 运行所有可视化函数
if __name__ == "__main__":
    print("🚀 开始KV缓存完整可视化演示...")
    
    # 1. 基础结构可视化
    visualize_tensor_structure()
    
    # 2. 操作演示
    simulate_kv_operations()
    
    # 3. 创建热力图
    create_tensor_heatmap()
    
    # 4. 创建3D可视化
    create_3d_visualization()
    
    # 5. 详细信息
    print_detailed_tensor_info()
    
    # 6. Attention计算演示
    demonstrate_attention_computation()
    
    print("\n" + "="*60)
    print("✅ 所有可视化完成！")
    print("📁 图片已保存到当前目录:")
    print("   - kv_cache_heatmap.png (热力图)")
    print("   - kv_cache_3d.png (3D可视化)")
    print("="*60)