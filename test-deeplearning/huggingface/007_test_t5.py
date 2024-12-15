from datasets import load_dataset, Dataset
from transformers import T5Tokenizer

class DatasetBuilder():
    def __init__(self, model_name:str):
        self.tokenizer:T5Tokenizer = T5Tokenizer.from_pretrained(model_name)
        # wanted_features = ["review_body", "review_headline", "product_title", "star_rating", "verified_purchase"]
        wanted_features = ["review_body", "product_title", "star_rating"]
        print("load_dataset")
        dataset = load_dataset(
            "rkf2778/amazon_reviews_mobile_electronics",
            split="train",
        )
        print("shuffle")
        dataset = dataset.shuffle(seed=42).select(range(10000))
        print("remove_columns")
        dataset = dataset.remove_columns([item for item in dataset.features if item not in wanted_features])
        print("add_prompt")
        dataset = dataset.map(self.add_prompt, remove_columns=wanted_features[0:2])
        # print("filter")
        # dataset = dataset.filter(lambda instance: instance[wanted_features[4]] and len(instance[wanted_features[0]]) > 100)
        print("class_encode_column")
        dataset = dataset.class_encode_column(wanted_features[2])
        print("train_test_split")
        self.ori_dataset = dataset
        dataset = dataset.train_test_split(
            test_size=0.1, seed=42, stratify_by_column=wanted_features[2]
        )
        self.train_dataset = self.map_func(dataset["train"])
        self.test_dataset = self.map_func(dataset["test"])

    def add_prompt(self, dataset: Dataset):
        dataset["prompt"] = f"review: {dataset["product_title"]} {dataset["star_rating"]} Stars!"
        dataset["response"] = f"{dataset["review_body"]}"
        return dataset

    def tokenize_dataset(self, dataset: Dataset):
        tokenize_func = \
            lambda data: self.tokenizer(
                data, return_tensors="pt", padding="max_length",
                truncation=True, max_length=128
            )
        inputs = tokenize_func(dataset["prompt"])
        targets = tokenize_func(dataset["response"])
        inputs.update({"labels": targets["input_ids"]})
        return inputs
    
    def map_func(self, dataset: Dataset):
        return dataset.map(
            self.tokenize_dataset,
            batched=True,
            remove_columns=['star_rating', 'prompt', 'response']
        )

if __name__ == "__main__":
    model_name = "t5-base"
    dataset_builder = DatasetBuilder(model_name)
    dataset_builder.ori_dataset
    dataset_builder.train_dataset
    wanted_features = ["review_body", "product_title", "star_rating"]
    wanted_features[0:2]
