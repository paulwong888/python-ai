import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from transformers import(
    AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments
)

def preprocess(example):
    example["prompt"] = f"{example["instruction"]} {example["input"]} {example["output"]}"
    return example

def tokenize_dataset(dataset: Dataset):
    tokenize_dataset = dataset.map(
        lambda example : tokenizer(example["prompt"], truncation=True, max_length=128), 
        batched=True, remove_columns=["prompt"]
    )
    return tokenize_dataset

dataset = load_dataset("hakurei/open-instruct-v1", split="train")
dataset.to_pandas().sample(20)

dataset = dataset.map(preprocess, remove_columns=["instruction", "input", "output"])
dataset.to_pandas().sample(20)#["prompt"][:1]

dataset = dataset.shuffle(seed=42).select(range(100000)).train_test_split(test_size=0.1)

train_dataset = dataset["train"]
test_dataset = dataset["test"]

model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

train_dataset = tokenize_dataset(train_dataset)
test_dataset = tokenize_dataset(test_dataset)

model = AutoModelForCausalLM.from_pretrained(model_name)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
output_dir="./diablogpt2-instruct"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator
)

trainer.train()
trainer.save_model()

model = AutoModelForCausalLM.from_pretrained(output_dir)

prompt = ""
device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_text(prompt):
    prompt =prompt + tokenizer.eos_token
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=64, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text[:generated_text.rfind(".") + 1]

print(generate_text("What's the best way to cook chicken breast?"))
print(generate_text("Should I invest in stocks?"))
print(generate_text("I need a place to go for this summer vacation, what locations would you recommend?"))
print(generate_text("What's the fastest route from NY to Boston?"))
