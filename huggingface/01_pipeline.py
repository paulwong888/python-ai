from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

classifier = pipeline("sentiment-analysis") #使用情感分析
result = classifier(
    [
        "寶寶覺得苦，但寶寶不說",
        "我愛寶寶"
    ]
)
print(result)