from transformers import BertTokenizer, BertModel
from transformers.utils.generic import ModelOutput

model_path = "/Volumes/Lexar-2T/PAUL/DOWNLOAD/TOOL/AI/MODEL/google-bert/bert-base-chinese"

tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_path)
print(f"Max input length: {tokenizer.model_max_length}")

texts = [
    "你好，世界",
    "我喜欢编程",
    "我是一个学生"
]
tokens = tokenizer(
    text=texts,
    add_special_tokens=True,
    padding=True,
    return_tensors="pt",
    max_length=128
)
input_ids = tokens["input_ids"]
attention_mask = tokens["attention_mask"]
print(f"shape of input_ids: {input_ids.shape}")
print(input_ids)
print(f"shape of attention_mask: {attention_mask.shape}")
print(attention_mask)

model: BertModel = BertModel.from_pretrained(model_path)
features = model(input_ids=input_ids, attention_mask=attention_mask)
print(type(features))
pooler_output = features.pooler_output
print(f"shape of pooler_output: {pooler_output.shape}")
print(pooler_output)