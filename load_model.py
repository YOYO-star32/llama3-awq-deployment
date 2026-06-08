import os 
import gc 
import torch
from transformers import AutoModelForCausalLM
os.environ['TRANSFORMERSNOADVISORYWARNINGS'] = '1' 
print"="∗60
 print("正在加载Llama-3-8B-Instruct AWQ模型到GPU...") 
print"="∗60 
model = AutoModelForCausalLM.from_pretrained( 
'./', 
torch_dtype=torch.float16, 
device_map='auto',
 	low_cpu_mem_usage=True, 
attn_implementation='eager'
 ) 
print"\n"+"="∗60
 print("✅ 模型部署验证成功！") 
print"="∗60 
print(f"模型已成功加载到设备: {model.device}") 
print(f"模型参数数量: {model.num_parameters()/1e9:.2f}B") 
print(f"GPU内存使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
print(f"GPU内存峰值: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
print"="∗60 
del model 
gc.collect() 
torch.cuda.empty_cache() 
print("\n模型已卸载，内存已清理")