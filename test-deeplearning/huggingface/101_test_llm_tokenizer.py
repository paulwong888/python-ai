from transformers import AutoTokenizer

messages = [
    {"role": "user", "content": "How do chat templates work?"},
    {"role": "assistant", "content": "Chat templates help LLMs like me generate more coherent responses by providing a structured way to organize the conversation."},
    {"role": "user", "content": "How do I use them?"},
]

tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
# tokenizer = AutoTokenizer.from_pretrained("unsloth/mistral-7b-instruct-v0.3")
tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-7b-it")
tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct")

tokenizer.apply_chat_template(conversation=messages, tokenize=False)