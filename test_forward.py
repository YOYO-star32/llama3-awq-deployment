import torch 
from transformers import AutoModelForCausalLM 
model = AutoModelForCausalLM.from_pretrained( 
'./', 
torch_dtype=torch.float16, 
device_map='auto', 
attn_implementation='eager' 
)
 input_ids = torch.randint(0, 1000, (1, 10)).to('cuda') 
with torch.no_grad(): 
outputs = model(inputids=inputids)
print"="∗60 
print("✅ 模型功能验证成功！（前向传播测试）") 
print"="∗60 
print(f"输入序列形状: {input_ids.shape}") 
print(f"输出logits形状: {outputs.logits.shape}") 
print(f"模型运行设备: {model.device}") 
print"="∗60 
print("说明：模型可正常接收输入并生成输出，证明部署后核心推理功能可用。")