from datasets import load_dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from peft.tuners.lora.config import LoraConfig
from peft.utils.peft_types import TaskType
from peft import get_peft_model
import torch


class DatasetBuilder:

    def __init__(self, dataset_name_or_path : str, model_name : str):
        self.dataset_name_or_path = dataset_name_or_path
        self.model_name = model_name

    def process_func(self, example : str):         
        """
        将数据集进行预处理
        """
        MAX_LENGTH = 384
        input_ids, attention_mask, labels = [], [], []
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        instruction = tokenizer(
            f"<|im_start|>system\n你是Docker专家，你会接收到一段文本，请输出相关的Docker 命令<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        response = tokenizer(f"{example['output']}", add_special_tokens=False)
        input_ids = (
            instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
        )
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        labels = (
            [-100] * len(instruction["input_ids"])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
        )
        if len(input_ids) > MAX_LENGTH:  # 做一个截断
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    
    def build_encoded_dataset(self):
        # origin_dataset = load_dataset("/home/paul/paul/work/workspaces/python/huggingface-transformers/data/MattCoddity")
        origin_dataset = load_dataset(self.dataset_name_or_path)
        print(origin_dataset)

        train_dataset = origin_dataset["train"]
        # validation_dataset = origin_dataset["validation"]
        print(train_dataset)

        # model_name = "distilbert-base-uncased" # 第三天預設的distilbert-base-uncased-finetuned-sst-2-english用這個

        # origin_dataset_encoded = origin_dataset.map(process_func, batched=True, batch_size=None)
        origin_dataset_encoded = train_dataset.map(self.process_func, remove_columns=train_dataset.column_names)

        # print(next(iter(origin_dataset_encoded))) #忘記這裡為什麼要用 next(iter())才能看到印出來的資料，可以回去看載入極巨大資料篇
        return origin_dataset_encoded
    
class PeftModelBuilder:
     
    def __init__(self):
         pass
     
    def build_perf_model(self, base_model : str, task_type : str, target_modules):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pretrained_model = AutoModelForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=True).to(device)
        
        for name, parameter in pretrained_model.named_parameters():
            print(name)
        
        config = LoraConfig(
            task_type=task_type,
            r=8,
            target_modules=target_modules,
            lora_dropout=0.1,  # Dropout 比例
         )
        print(config)
        
        peft_model = get_peft_model(pretrained_model, peft_config=config)
        print(peft_model)
        return peft_model
    
    
    