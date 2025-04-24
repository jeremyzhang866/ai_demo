from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 加载基础模型和 tokenizer
base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16)

# 加载 LoRA 适配器
lora_adapter_id = "JosephusCheung/TinyLlama-1.1B-Chat-v0.6-LoRA"
model = PeftModel.from_pretrained(model, lora_adapter_id)

# 设置为评估模式
model.eval()

# 编码输入
prompt = "请简单介绍一下老子是谁？"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 推理
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=100)

# 解码输出
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("🤖 回答：", output)
