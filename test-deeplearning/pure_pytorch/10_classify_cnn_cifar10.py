import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import requests
from torch import nn, Tensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageOps

"""
poor performance
"""
class ClassifyCnnCifar10(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.linear1 = nn.Linear(50 * 5 * 5, 500)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(500, 10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        # print(X.shape)
        X = X.view(X.shape[0], -1)
        # print(X.shape)
        X = F.relu(self.linear1.forward(X))
        X = self.dropout1(X)
        X = self.linear2.forward(X)
        return X
    
"""
improve performance
out channel *2
kernal: 5->3
add padding=1
add one more Conv2d layer
"""
class ClassifyCnnCifar10New(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.linear1 = nn.Linear(64 * 4 * 4, 500)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(500, 10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        # print(X.shape)
        X = X.view(X.shape[0], -1)
        # print(X.shape)
        X = F.relu(self.linear1.forward(X))
        X = self.dropout1(X)
        X = self.linear2.forward(X)
        return X
    
class ClassifyCnnCifar10Trainer():
    def __init__(self, model: nn.Module):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.to_device(model)
        self.classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "hose", "ship", "truck"]
        self.transform = self.__transform__()

    def to_device(self, model:nn.Module):
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model, device_ids=[0,1])
                model = model.to("cuda")
        else:
            model = model.to("cpu")
        return model
    
    def __transform__(self):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(np.full(3, 0.5, float), np.full(3, 0.5, float))
        ])
        return transform
    
    def __transform_new__(self, train: bool):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=10, saturation=(0.2)),
            transforms.ToTensor(),
            transforms.Normalize(np.full(3, 0.5, float), np.full(3, 0.5, float))
        ])
        if train:
            return transform
        else:
            return self.__transform__()

    def build_data_set(self, train: bool, shuffle: bool, transform):
        transform = self.transform
        dataset = datasets.CIFAR10("./data/cifar10", train=train, download=True, transform=transform)
        data_loader = DataLoader(dataset, 100, shuffle=shuffle)
        return data_loader
    
    def img_convert(self, img: Tensor):
        img = img.clone().float().detach().numpy()
        img = img.transpose(1, 2, 0)
        img = img * np.full(3, 0.5, float) + np.full(3, 0.5, float)
        img = img.clip(0, 1)
        return img
    
    def plot_data_set(self, data_loader: DataLoader):
        dataiter = iter(data_loader)
        (images, targets) = dataiter.__next__()
        figure = plt.figure(figsize=(25, 4))

        for i in range(20):
            ax = figure.add_subplot(2, 10, i+1)
            plt.imshow(self.img_convert(images[i]))
            ax.set_title(self.classes[targets[i].item()])
        plt.show()

    def train(self, train_data_loader: DataLoader, val_data_loader: DataLoader, lr):
        ephochs = 15
        loss_func = nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001) increase too small
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        train_running_loss_history = []
        train_running_corrects_history = []
        val_running_loss_history = []
        val_running_corrects_history = []

        for i in range(ephochs):
            train_running_loss = 0.0
            train_running_corrects = 0.0
            val_running_loss = 0.0
            val_running_corrects = 0.0
            for (images, targets) in train_data_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                # print(images.shape)
                outputs = self.model.forward(images)
                # print(outputs)
                loss = loss_func(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print(images.size(0) == batch_size)
                train_running_loss += loss.item()
                # train_running_loss += loss.item() * images.size(0)
                (_, preds) =torch.max(outputs, 1)
                train_running_corrects += torch.sum(preds == targets.data)
            else: #while for finished
                with torch.no_grad():
                    for (val_inputs, val_targets) in val_data_loader:
                        val_inputs = val_inputs.to(self.device)
                        val_targets = val_targets.to(self.device)
                        val_outputs = self.model.forward(val_inputs)
                        val_loss = loss_func(val_outputs, val_targets)

                        val_running_loss += val_loss.item()
                        (_, val_preds) = torch.max(val_outputs, 1)
                        val_running_corrects += torch.sum(val_preds == val_targets.data)

                print(len(train_data_loader))
                print(len(train_data_loader.sampler))
                epoch_train_running_loss = train_running_loss / len(train_data_loader)
                # epoch_train_running_loss = train_running_loss / len(train_data_loader.sampler)
                epoch_train_running_croorcts = train_running_corrects / len(train_data_loader)
                train_running_loss_history.append(epoch_train_running_loss)
                train_running_corrects_history.append(epoch_train_running_croorcts.cpu().numpy())

                epoch_val_running_loss = val_running_loss / len(val_data_loader)
                epoch_val_running_croorcts = val_running_corrects / len(val_data_loader)
                val_running_loss_history.append(epoch_val_running_loss)
                val_running_corrects_history.append(epoch_val_running_croorcts.cpu().numpy())
                print(f"epoch: {i}, loss {epoch_train_running_loss:.4f}, acc {epoch_train_running_croorcts:.4f}", 
                    f"epoch: {i}, val_loss {epoch_val_running_loss:.4f}, val_acc {epoch_val_running_croorcts:.4f}")
        else:
            plt.plot(train_running_loss_history, label="train loss")
            plt.plot(val_running_loss_history, label="validation loss")
            plt.show()
            plt.plot(train_running_corrects_history, label="train acc")
            plt.plot(val_running_corrects_history, label="validation acc")
            plt.show()

    def test_web_img(self):
        url = "https://assets.petco.com/petco/image/upload/f_auto,q_auto/green-tree-frog-care-sheet-hero"
        response = requests.get(url, stream=True)
        img = Image.open(response.raw)
        img = self.transform(img)

        plt.imshow(self.img_convert(img))
        plt.show()

        img = img.to(self.device).unsqueeze(0)
        output = self.model.forward(img)
        (_, pred) = torch.max(output, 1)
        print(self.classes[pred.item()])
    
def train_test():
    trainer = ClassifyCnnCifar10Trainer(ClassifyCnnCifar10())
    train_data_loader = trainer.build_data_set(True, True, trainer.transform)
    trainer.plot_data_set(train_data_loader)

    val_data_loader = trainer.build_data_set(False, False, trainer.transform)

    lr=0.001
    trainer.train(train_data_loader, val_data_loader, lr=lr)
    return trainer
    
def train_test_new():
    trainer = ClassifyCnnCifar10Trainer(ClassifyCnnCifar10New())
    train_data_loader = trainer.build_data_set(True, True, trainer.__transform_new__(True))
    trainer.plot_data_set(train_data_loader)

    val_data_loader = trainer.build_data_set(False, False, trainer.transform)

    lr=0.0001
    trainer.train(train_data_loader, val_data_loader, lr=lr)
    return trainer


if __name__ == "__main__":
    # trainer = ClassifyCnnCifar10Trainer(ClassifyCnnCifar10())
    # trainer = ClassifyCnnCifar10Trainer(ClassifyCnnCifar10New())
    # train_test()

    trainer = train_test_new()

    # trainer = ClassifyCnnCifar10Trainer(ClassifyCnnCifar10New())
    trainer.test_web_img()