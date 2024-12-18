from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
import torch, gc
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

from func_fine_turnning_dataset import DatasetBuilder, PeftModelBuilder


gc.collect()
torch.cuda.empty_cache()

# ds = load_dataset("/opt/tool/ai/fine-tunning/LLaMA-Factory/data/MattCoddity/")
dataset_file_path = "/home/paul/PAUL/work/workspaces/python-ai/huggingface-transformers/data/MattCoddity/"
# base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
# base_model = "Qwen/Qwen2-0.5B"
base_model = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"

dataset_builder = DatasetBuilder(dataset_file_path, base_model)
encoded_dataset = dataset_builder.build_encoded_dataset()
print(type(encoded_dataset))



target_modules=["q_proj", "v_proj", "o_proj"]
peftmodel_builder = PeftModelBuilder()
peft_model = peftmodel_builder.build_perf_model(base_model, TaskType.CAUSAL_LM, target_modules)
print(peft_model)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=1e-3,
    num_train_epochs=3,
    logging_steps=20,
    # 其他参数...
)

tokenizer = AutoTokenizer.from_pretrained(base_model)
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=encoded_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

fine_tuned_model = "./fine_tuned_model"
peft_model.save_pretrained(fine_tuned_model)
# tokenizer.save_pretrained(fine_tuned_model)

# Flush memory
del trainer, peft_model, 
gc.collect()
torch.cuda.empty_cache()

