import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling,
    Trainer, TrainingArguments
)

class DatasetBuilder():
    def __init__(self, model_name:str):
        self.prompt_column_name = "prompt"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dataset = load_dataset("hakurei/open-instruct-v1", split="train")
        dataset = dataset.shuffle(seed=42).select(range(10000))
        dataset = dataset.map(self.prerpocess, remove_columns=["instruction","input","output"])
        dataset = dataset.train_test_split(test_size=0.1)
        self.ori_dataset = dataset
        self.train_dataset = self.tokenize_dataset(dataset["train"])
        self.test_dataset = self.tokenize_dataset(dataset["test"])

    def prerpocess(self, example: Dataset):
        example[self.prompt_column_name] = f"{example["instruction"]} {example["input"]} {example["output"]}"
        return example
    
    def tokenize_dataset(self, dataset: Dataset):
        return dataset.map(
            lambda example : self.tokenizer(example[self.prompt_column_name], truncation=True, max_length=128),
            batched=True, remove_columns=[self.prompt_column_name]
        )
    
class ChatGPT2Model():
    def __init__(self, model_name:str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

    def generate_text(self, prompt):
        prompt = prompt + self.tokenizer.eos_token
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs, max_length=64, pad_token_id=self.tokenizer.eos_token_id)
        generate_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generate_text
    
class MyTrainer():
    def __init__(self, model: ChatGPT2Model, dataset_builder: DatasetBuilder):
        self.output_dir="./diablogpt2-instruct"
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16
        )
        self.trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset = dataset_builder.train_dataset,
            eval_dataset = dataset_builder.test_dataset,
            data_collator = model.data_collator
        )
    
    def train(self):
        self.trainer.train()
        self.trainer.save_model()
        
if __name__ == "__main__":
    model_name = "microsoft/DialoGPT-medium"
    dataset_builder = DatasetBuilder(model_name)
    dataset_builder.ori_dataset["train"][dataset_builder.prompt_column_name][0]
    dataset_builder.test_dataset
    model = ChatGPT2Model(model_name)
    model.generate_text("Brainstorm a list of possible New Year's resolutions.")
    model.generate_text("What's the best way to cook chicken breast?")
    model.generate_text("Should I invest in stocks?")
    model.generate_text("I need a place to go for this summer vacation, what locations would you recommend?")
    model.generate_text("What's the fastest route from NY to Boston?")
