import torch
import os
from text_classify_dataset import TextClassifyDataset
from bert_classifier import BertClassifier
from file_loader import FileLoader
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
from sklearn.model_selection import train_test_split
from pandas import DataFrame

class Trainer():
    def __init__(self):
        self.model_path = "/Volumes/Lexar-2T/PAUL/DOWNLOAD/TOOL/AI/MODEL/google-bert/bert-base-chinese"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def build_data(self, file_path):
        file_loader = FileLoader(file_path)
        return file_loader.build_dataset()

    def build_data_set(self, data_frame: DataFrame):
        tokenizer = BertTokenizer.from_pretrained(self.model_path)
        dataset = TextClassifyDataset(data_frame, tokenizer)
        return dataset

    def build_data_loader(self, dataset: TextClassifyDataset):
        # text_list = dataset.get_text_list()
        # label_list = dataset.get_label_list()
        # X_train, = train_test_split()
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=50,
            shuffle=True,
            # collate_fn=lambda x : dataset.collate_fn(x, tokenizer=tokennizer)
            collate_fn=dataset.collate_fn
        )
        return data_loader, dataset.get_label_num()
    
    def build_model(self, class_num):
        model = BertClassifier(self.model_path, class_num)
        print(model)
        return model.to(self.device)
    
    def train(self):
        file_path = "/Volumes/Lexar-2T/PAUL/DOWNLOAD/TOOL/AI/MODEL/data-set/Chinese Text Multi-classification Dataset/train.txt"
        
        X_train, y_train = self.build_data(file_path)
        dataset = self.build_data_set(X_train)
        (data_loader, class_num) = self.build_data_loader(dataset)

        model_path = "/Volumes/Lexar-2T/PAUL/DOWNLOAD/TOOL/AI/MODEL/google-bert/bert-base-chinese"
        model = self.build_model(class_num)

        optimizer = Adam(model.parameters(), lr=5e-5)
        loss_func = nn.CrossEntropyLoss()
        os.makedirs("output_models", exist_ok=True)
        epochs = 10

        for epoch in range(epochs):
            for (batch_idx, data) in enumerate(data_loader):
                input_ids = data[0]#.to(self.device)
                attention_mask = data[1]#.to(self.device)
                label = data[2]#.to(self.device)

                optimizer.zero_grad()
                output = model(input_ids, attention_mask)
                # print(output.shape, label.shape)
                loss = loss_func(output, label)
                loss.backward()
                optimizer.step()

                predict = torch.argmax(output, dim=1)
                correct = (predict == label).sum().item()
                acc = correct / output.size(0)

                print(
                    f"Epoch {epoch}/{epochs}"
                    f"| Batch {batch_idx + 1}/{len(data_loader)}"
                    f"| Loss: {loss.item():.4f}"
                    f"| acc: {correct}/{output.size(0)}={acc:.3f}"
                )
            model_name = f"./output_models/chinese_news_classify{epoch}.pth"
            print(f"saved model: {model_name}")
            torch.save(model.state_dict(), model_name)

if __name__ == "__main__":
    Trainer().train()