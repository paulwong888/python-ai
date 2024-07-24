from transformers import AutoModelForCausalLM, AutoTokenizer
from peft.peft_model import PeftModelForCausalLM

# model_name = "Qwen/Qwen2-0.5B"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
adapter_name = "fine_tuned_model"
# new_model = ""

tokenizer = AutoTokenize.from_pretrained(model_name)
origin_model = AutoModelForCausalLM.from_pretrained(model_name)
peft_model = PeftModelForCausalLM.from_pretrained(model=origin_model, model_id=adapter_name)

# 获取微调后的模型权重
finetuned_weights = peft_model.state_dict()

# 应用微调后的权重到原始模型
for key in finetuned_weights:
    if key in origin_model.state_dict():
        origin_model.state_dict()[key] = finetuned_weights[key]
    else:
        print(f"Key {key} not found in original model")

# 保存合并后的模型
combined_model_path = "/opt/tool/ai/fine-tunning/my-test/saves/llama3-8b/lora/docker-commnad-nlp/export"
origin_model.save_pretrained(combined_model_path)
tokenizer.save_pretrained(combined_model_path)
