import torch
def show_actual_tensor_structure():
    """展示实际的Tensor结构"""
    
    print("\n=== 实际Tensor结构展示 ===")
    
    # 创建一个小规模的示例
    k_buffer_layer1 = torch.zeros(10, 2, 4)  # [位置, 头, 维度]
    v_buffer_layer1 = torch.zeros(10, 2, 4)
    
    # 模拟存储一些值
    positions = [0, 1, 2]
    for i, pos in enumerate(positions):
        # 为每个位置的每个头设置不同的值
        k_buffer_layer1[pos, 0, :] = torch.tensor([1.0 + i, 2.0 + i, 3.0 + i, 4.0 + i])
        k_buffer_layer1[pos, 1, :] = torch.tensor([5.0 + i, 6.0 + i, 7.0 + i, 8.0 + i])
        
        v_buffer_layer1[pos, 0, :] = torch.tensor([0.1 + i*0.1, 0.2 + i*0.1, 0.3 + i*0.1, 0.4 + i*0.1])
        v_buffer_layer1[pos, 1, :] = torch.tensor([0.5 + i*0.1, 0.6 + i*0.1, 0.7 + i*0.1, 0.8 + i*0.1])
    
    print("K Buffer (第1层) 结构:")
    print(f"形状: {k_buffer_layer1.shape}")
    print("内容 (前3个位置):")
    for pos in range(3):
        print(f"  位置{pos}:")
        for head in range(2):
            print(f"    头{head}: {k_buffer_layer1[pos, head, :].tolist()}")
    
    print("\nV Buffer (第1层) 结构:")
    print(f"形状: {v_buffer_layer1.shape}")
    print("内容 (前3个位置):")
    for pos in range(3):
        print(f"  位置{pos}:")
        for head in range(2):
            print(f"    头{head}: {v_buffer_layer1[pos, head, :].tolist()}")
    
    print("\n=== 访问模式示例 ===")
    # 模拟GET操作
    query_positions = torch.tensor([0, 1, 2])
    retrieved_k = k_buffer_layer1[query_positions]  # [3, 2, 4]
    retrieved_v = v_buffer_layer1[query_positions]  # [3, 2, 4]
    
    print(f"查询位置: {query_positions.tolist()}")
    print(f"检索到的K形状: {retrieved_k.shape}")
    print(f"检索到的V形状: {retrieved_v.shape}")
    
    print("\n检索结果:")
    print("K值:")
    print(retrieved_k)
    print("V值:")
    print(retrieved_v)

show_actual_tensor_structure()