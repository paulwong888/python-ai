import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer
from pandas import DataFrame

class TextClassifyDataset(Dataset):
    def __init2__(self, file_path, tokenizer: BertTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.examples = list()
        self.labels = set()
        # file_path = "/Volumes/Lexar-2T/PAUL/DOWNLOAD/TOOL/AI/MODEL/data-set/Chinese Text Multi-classification Dataset/train.txt"
        with open(file=file_path, mode="r", encoding="utf-8") as file:
            for line in file.readlines():
                (text, label) = line.strip().split("\t")
                self.examples.append((text, int(label)))
                self.labels.add(label)

    def __init__(self, data_frame: DataFrame, tokenizer: BertTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_frame = data_frame
        self.labels = set(self.data_frame["label"])
    
    def __len__(self):
        # return len(self.examples)
        return len(self.data_frame)
    
    def __getitem__(self, index):
        return self.data_frame.iloc[index]
    
    def get_label_num(self):
        # return len(np.unique(data['label']))
        return len(self.labels)

    def collate_fn(self, batch):
        sentences = [text.iloc[0] for text in batch]
        # sentences = torch.Tensor(sentences)
        labels = [text.iloc[1] for text in batch]
        labels = torch.LongTensor(labels)
        tokens = self.tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_attention_mask=True,
            return_tensors="pt"
        )
        # return torch.LongTensor(tokens["input_ids"]), torch.Tensor(tokens["attention_mask"]), labels
        return tokens["input_ids"], tokens["attention_mask"], labels
    
    def get_text_list(self):
        return self.data_frame["text"]
    
    def get_label_list(self):
        return self.data_frame["label"]