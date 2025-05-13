# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import torch
from transformers import pipeline

# pipe = pipeline("text-generation", model="test-deeplearning/train_llm/02_prep_for_train/models/TinyLlama_Storyteller", torch_dtype=torch.bfloat16, device_map="auto")
# pipe = pipeline("text-generation", model="unsloth/Meta-Llama-3.1-8B", torch_dtype=torch.bfloat16, device_map="auto")
pipe = pipeline("text-generation", model="test-deeplearning/train_llm/05_scale/models/Llama3_Storyteller_deepspeed/merged", torch_dtype=torch.bfloat16, device_map="auto")

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating

messages = [
    {"role": "system", "content": "You are an amazing storyteller. From the following synopsis, create an engaging story."},
    {"role": "user", "content": "A bright student was working with The Fuzzy Scientist on a project."},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=512)
print(outputs[0]["generated_text"])
# <|system|>
# You are a friendly chatbot who always responds in the style of a pirate.</s>
# <|user|>
# How many helicopters can a human eat in one sitting?</s>
# <|assistant|>
# ...
