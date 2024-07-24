from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments

def tokenize(batch):
    return tokenizer(batch["verse_text"], padding=True, truncation=True)

sentiment = load_dataset("poem_sentiment", trust_remote_code=True)

model_name = "distilbert-base-uncased" # 第三天預設的distilbert-base-uncased-finetuned-sst-2-english用這個
tokenizer = AutoTokenizer.from_pretrained(model_name)

sentiment_encoded = sentiment.map(tokenize, batched=True, batch_size=None)

print(next(iter(sentiment_encoded["train"]))) #忘記這裡為什麼要用 next(iter())才能看到印出來的資料，可以回去看載入極巨大資料篇


#############################
#准备model
from transformers import AutoModelForSequenceClassification
import torch

labels = {"negative","positive","no_impact","mixed"}
num_labels = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = (AutoModelForSequenceClassification
         .from_pretrained(model_name, num_labels=num_labels,
                          id2label={
                              "0": "negative",
                              "1": "positive",
                              "2": "no_impact",
                              "3": "mixed"    
                          },
                          label2id={
                              "negative": "0",
                              "positive": "1",
                              "no_impact": "2",
                              "mixed": "3"
                          }
                          )
         .to(device)
)
         
#############################
#准备arguments
batch_size = 64
logging_steps = len(sentiment_encoded["train"]) // batch_size
my_model_name = "poem_model"
training_args = TrainingArguments(output_dir=my_model_name,
                                  num_train_epochs=40,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False,
                                  label_names= labels,
                                  report_to = "mlflow",
                                  logging_steps=logging_steps)

from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

#############################
#开始训练
trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=sentiment_encoded["train"],
                  eval_dataset=sentiment_encoded["validation"],
                  tokenizer=tokenizer)
trainer.train()