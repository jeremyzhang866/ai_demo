def complete_kv_cache_workflow():
    """展示完整的KV缓存工作流程"""
    
    print("=== 完整KV缓存工作流程 ===")
    
    # 1. 初始化（模拟模型启动）
    print("\n1. 初始化阶段:")
    print("   - 创建k_buffer和v_buffer")
    print("   - 分配GPU内存")
    print("   - 初始化为零值")
    
    # 2. 第一次前向传播（SET操作）
    print("\n2. 第一次前向传播 - SET操作:")
    print("   输入: '你好，我是'")
    print("   Token IDs: [101, 102, 103, 104]")
    print("   分配位置: [0, 1, 2, 3]")
    
    # 模拟存储
    layer_id = 1
    positions = [0, 1, 2, 3]
    for i, pos in enumerate(positions):
        k_val = f"K_token_{i+101}"  # 模拟K值
        v_val = f"V_token_{i+101}"  # 模拟V值
        print(f"   位置{pos}: 存储 {k_val}, {v_val}")
    
    # 3. 生成阶段（GET + SET操作）
    print("\n3. 生成新token - GET + SET操作:")
    print("   生成token: '助手' (ID: 105)")
    
    print("   GET操作:")
    print("     - 获取位置[0,1,2,3]的历史KV值")
    print("     - 用于计算attention with 新query")
    
    print("   SET操作:")
    print("     - 计算新token的KV值")
    print("     - 存储到位置[4]")
    print(f"     位置4: 存储 K_token_105, V_token_105")
    
    # 4. 继续生成
    print("\n4. 继续生成 - 循环GET+SET:")
    for i, (token_id, token) in enumerate([(106, "。"), (107, "有"), (108, "什么")]):
        pos = 5 + i
        print(f"   生成'{token}' (ID: {token_id}):")
        print(f"     GET: 位置[0-{pos-1}]的KV值")
        print(f"     SET: 位置[{pos}]存储新KV值")
    
    print("\n5. 最终状态:")
    print("   k_buffer[1][0:8] 包含8个token的K值")
    print("   v_buffer[1][0:8] 包含8个token的V值")
    print("   每个位置对应一个token的表示")

complete_kv_cache_workflow()