import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import requests
from torch import nn, Tensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageOps

class ClassifyCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.linear1 = nn.Linear(50 * 4 * 4, 500)
        self.linear2 = nn.Linear(500, 10)

    def forward(self, X):
        X = F.relu(self.conv1.forward(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2.forward(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 50 * 4 * 4)
        X = F.relu(self.linear1.forward(X))
        X = self.linear2.forward(X)
        return X

class ClassifyTrainer():
    def __init__(self, model: nn.Module):
        super().__init__()
        self.device = torch.device("cuda:0,1 " if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def build_data(self, train: bool, shuffle: bool):
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        dataset = datasets.MNIST("./mnist", download=False, train=train, transform=transform)
        data_loader = DataLoader(dataset, batch_size=100, shuffle=shuffle)
        return data_loader
    
    def img_convert(self, img: Tensor):
        img = img.clone().float().detach().numpy()
        img = img.transpose(1, 2, 0)
        img = img * np.full(3, 0.5, float) + np.full(3, 0.5, float)
        img = img.clip(0, 1)
        return img
    
    def plot_dataset(self, data_loader: DataLoader):
        dataiter = iter(data_loader)
        (images, targets) = dataiter.__next__()
        figure = plt.figure(figsize=(25, 4))

        for i in range(20):
            ax = figure.add_subplot(2, 10, i+1)
            plt.imshow(self.img_convert(images[i]))
            ax.set_title(targets[i].item())
        plt.show()

    def train(self, train_data_loader: DataLoader, val_data_loader: DataLoader):
        ephochs = 5
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

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

                train_running_loss += loss.item()
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

                epoch_train_running_loss = train_running_loss / len(train_data_loader)
                epoch_train_running_croorcts = train_running_corrects / len(train_data_loader)
                train_running_loss_history.append(epoch_train_running_loss)
                train_running_corrects_history.append(epoch_train_running_croorcts)

                epoch_val_running_loss = val_running_loss / len(val_data_loader)
                epoch_val_running_croorcts = val_running_corrects / len(val_data_loader)
                val_running_loss_history.append(epoch_val_running_loss)
                val_running_corrects_history.append(epoch_val_running_croorcts)
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
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        url = "https://d00.paixin.com/thumbs/2899123/37531951/staff_1024.jpg"
        response = requests.get(url, stream=True)
        img = Image.open(response.raw)
        img = ImageOps.invert(img)
        img = img.convert("1")
        img = transform(img)

        plt.imshow(self.img_convert(img))
        plt.show()

        output = self.model.forward(img)
        print(type(output))
        print(output)
        (_, pred) = torch.max(output, 1)
        print(pred)


def train_test():
    trainer = ClassifyTrainer(ClassifyCnn())

    train_data_loader = trainer.build_data(True, True)
    trainer.plot_dataset(train_data_loader)

    val_data_loader = trainer.build_data(train=False, shuffle=False)
    trainer.train(train_data_loader=train_data_loader, val_data_loader=val_data_loader)

    trainer.test_web_img()

if __name__ == "__main__":

    train_test()
    # ClassifyTrainer(ClassifyCnn()).test_web_img()