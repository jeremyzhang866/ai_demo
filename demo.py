# 使用列表生成式直接构建 prompts 列表
prompts = [f"AI is {i}" for i in range(10)]

print(prompts)