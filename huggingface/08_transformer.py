from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch

#获取token
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)

string_arr = [
    "Only those who will risk going too far can possibly find out how far one can go.",
    "Baby shark, doo doo doo doo doo doo, Baby shark!"
]
inputs = tokenizer(string_arr, padding=True, truncation=True, return_tensors="pt")
print(inputs)

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
outputs = model(**inputs)

print(outputs)

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

print(model.config.id2label(predictions))
