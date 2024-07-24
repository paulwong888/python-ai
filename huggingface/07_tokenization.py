from datasets import load_dataset
from transformers import AutoTokenizer

def tokenize(batch, tokenizer):
    return tokenizer(batch["verse_text"], padding=True, truncation=True)

string = "Only those who will risk going too far can possibly find out how far one can go."

model_name = "distilbert-base-uncased-finetuned-sst-2-english" #直接叫model名字
tokenizer = AutoTokenizer.from_pretrained(model_name)

sentiment = load_dataset("poem_sentiment", trust_remote_code=True) #真正获取数据

print(tokenize(sentiment["train"][:3], tokenizer))
