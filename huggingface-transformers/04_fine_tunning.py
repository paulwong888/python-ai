import gc
import torch
from torchtext.datasets import IMDB
import random
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer

from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# print(torch.cuda.is_available())

gc.collect()
torch.cuda.empty_cache()

train_iter = IMDB(split="train")
test_iter = IMDB(split="test")

train_lists = list(train_iter)
test_lists = list(test_iter)

train_lists_small = random.sample(train_lists, 1000)
test_lists_small = random.sample(test_lists, 1000)
print(train_lists_small[0])
print(test_lists_small[0])

train_texts = []
train_labels = []

for label, text in train_lists_small:
    train_labels.append(0 if label == "1" else 1)
    # print(label)
    train_texts.append(text)

test_texts = []
test_labels = []

for label, text in test_lists_small:
    test_labels.append(0 if label == "1" else 1)
    test_texts.append(text)

print(train_texts[0])
print(train_labels[0])
print(test_texts[0])
print(test_labels[0])


train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
print(len(train_texts))
print(len(train_labels))
print(len(val_texts))
print(len(val_labels))

base_model = "distilbert-base-uncased"
new_model = "my-adapter"

tokenizer = DistilBertTokenizerFast.from_pretrained(base_model)

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)
# Display IDs up to the 0th 5 tokens
print(train_encodings["input_ids"][0][:5])
# Decode and display tokens up to the 0th 5 tokens
print(tokenizer.decode(train_encodings["input_ids"][0][:5]))

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)
    
train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

for i in train_dataset:
    print(i)
    break

# Load distilbert model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
print(model)

training_args = TrainingArguments(
    output_dir="./results", # Output directory path
    num_train_epochs=8, # Training epoch
    # per_device_train_batch_size=16, # mini batch size for training(per device)
    # per_device_eval_batch_size=64, # mini batch size for evaluation(per device)
    per_device_train_batch_size=8, # mini batch size for training(per device)
    per_device_eval_batch_size=8, # mini batch size for evaluation(per device)
    warmup_steps=500, # Number of warm-up steps for learning rate scheduler
    weight_decay=0.01, # Strength of weight decay
    logging_dir="./logs", # Log directory path
    logging_steps=10,
)

# Transfer the tensor to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

trainer = Trainer(
    model=model, # Instantiated Pre-trained model
    args=training_args, # Hyper parameters defined in transformers.Arguments
    train_dataset=train_dataset, # Training dataset
    eval_dataset=val_dataset # Evaluation Dataset
)
trainer.train()
# trainer.save_model(new_model)
model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)

# Flush memory
del trainer, model
gc.collect()
torch.cuda.empty_cache()


# Reload tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained(base_model)
# Note that we reload the model in fp16 
# fp16_model = AutoModelForCausalLM.from_pretrained(
#     base_model,
#     low_cpu_mem_usage=True,
#     return_dict=True,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )

# Merge adapter with base model
# model = PeftModel.from_pretrained(fp16_model, new_model)
# model = model.merge_and_unload()

# model.save_pretrained("my-model")
# tokenizer.save_pretrained("my-model")

# input_tokens = tokenizer(["I feel fantastic", "My life is going something wrong", "I have not figured out what the chosen title has to do with the movie."], truncation=True, padding=True)
# outputs = model(torch.tensor(input_tokens["input_ids"]).to(device))
# label_dict = {0:"positive", 1:"negative"}
# print([label_dict[i] for i in torch.argmax(outputs["logits"], axis=1).cpu().numpy()])