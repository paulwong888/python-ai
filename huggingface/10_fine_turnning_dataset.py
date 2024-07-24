from datasets import load_dataset
from transformers import AutoTokenizer

def tokenize(batch):
    return tokenizer(batch["verse_text"], padding=True, truncation=True)

sentiment = load_dataset("poem_sentiment", trust_remote_code=True)
print(type(sentiment["train"][:1]["verse_text"]))

model_name = "distilbert-base-uncased" # 第三天預設的distilbert-base-uncased-finetuned-sst-2-english用這個
tokenizer = AutoTokenizer.from_pretrained(model_name)

sentiment_encoded = sentiment.map(tokenize, batched=True, batch_size=None)

print(next(iter(sentiment_encoded["train"]))) #忘記這裡為什麼要用 next(iter())才能看到印出來的資料，可以回去看載入極巨大資料篇
