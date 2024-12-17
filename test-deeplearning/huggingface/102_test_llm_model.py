import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {"role": "user", "content": "How do chat templates work?"},
    {"role": "assistant", "content": "Chat templates help LLMs like me generate more coherent responses by providing a structured way to organize the conversation."},
    {"role": "user", "content": "How do I use them?"},
]

model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)

generated_ids = model.generate(model_inputs)
generated_ids = model.generate(model_inputs, do_sample=True) #此参数使得每次回答都不一样
generated_ids = model.generate(model_inputs, do_sample=True, temperature=0.5) #temperature=0.1,则控制回答每次都一样,如果是0.5,则会出居中的结果
generated_ids = model.generate(model_inputs, do_sample=True, temperature=1.0, top_p=0.1) #top_p=0.1,只能使用概率排名前10的预测
decoded_response = tokenizer.batch_decode(generated_ids)
print(decoded_response[0])
