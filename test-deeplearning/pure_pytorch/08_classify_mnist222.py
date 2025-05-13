import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import Tensor
from torch.optim import Adam


class ClassifyMnist(nn.Module):
    def __init__(self, input_size, H1, H2, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, output_size)

    def forward(self, X):
        X = F.dropout(F.relu(self.linear1(X)))
        X = F.dropout(F.relu(self.linear2(X)))
        X = self.linear3(X)
        return X

class ClassifyTrainer():

    def build_data(self, train: bool, shuffle: bool):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5))
        ])
        dataset = datasets.MNIST("./data/mnist", train=train, download=False, transform=transform)
        data_loader = DataLoader(dataset=dataset, batch_size=100, shuffle=shuffle)
        return data_loader
    
    def img_convert(self, img: Tensor):
        img = img.clone().detach().numpy()
        img = img.transpose(1, 2, 0)
        img = img * np.full(3, 0.5, float) + np.full(3, 0.5, float)
        img = img.clip(0, 1)
        return img

    def pylot_train_data(self, data_loader: DataLoader):
        dataiter = iter(data_loader)
        (images, targets) = dataiter.__next__()
        figure = plt.figure(figsize=(25, 4))

        for i in range(20):
            ax = figure.add_subplot(2, 10, i+1)
            plt.imshow(self.img_convert(images[i]))
            ax.set_title(targets[i].item())
        plt.show()

    def train(self, model: nn.Module, data_loader: DataLoader, val_data_loader: DataLoader):
        ephochs = 20
        loss_func = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=0.0001)

        running_loss_history = []
        running_corrects_history = []
        val_running_loss_history = []
        val_running_corrects_history = []

        for i in range(ephochs):
            train_running_loss = 0.0
            train_running_corrects = 0.0
            val_running_loss = 0.0
            val_running_corrects = 0.0
            for (images, targets) in data_loader:
                images = images.view(images.shape[0], -1)
                outputs = model.forward(images)
                loss = loss_func(outputs, targets)

                train_running_loss += loss.item()
                (_, preds) = torch.max(outputs, 1)
                train_running_corrects += torch.sum(preds == targets.data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    for (val_images, val_targets) in val_data_loader:
                        val_images = val_images.view(val_images.shape[0], -1)
                        val_outputs = model.forward(val_images)
                        val_loss = loss_func(val_outputs, val_targets)

                        val_running_loss += val_loss.item()
                        (_, val_preds) = torch.max(val_outputs, 1)
                        val_running_corrects += torch.sum(val_preds == val_targets)
                
                epoch_loss = train_running_loss / len(data_loader)
                epoch_corrects = train_running_corrects / len(data_loader)
                running_loss_history.append(epoch_loss)
                running_corrects_history.append(epoch_corrects)

                val_epoch_loss = val_running_loss / len(val_data_loader)
                val_epoch_corrects = val_running_corrects / len(val_data_loader)
                val_running_loss_history.append(val_epoch_loss)
                val_running_corrects_history.append(val_epoch_corrects)

                print(f"epoch: {i}, loss {epoch_loss:.4f}, acc {epoch_corrects:.4f}", f"epoch: {i}, val_loss {val_epoch_loss:.4f}, val_acc {val_epoch_corrects:.4f}")
                # print(f"epoch: {i}, val_loss {val_epoch_loss:.4f}, val_acc {val_epoch_corrects:.4f}")
                # print()
        else:
            plt.plot(running_loss_history, label="trainning loss")
            plt.plot(val_running_loss_history, label="validation loss")
            plt.show()
            plt.plot(running_corrects_history, label="trainning acc")
            plt.plot(val_running_corrects_history, label="validation acc")
            plt.show()



def train_test():
    trainer = ClassifyTrainer()

    data_loader = trainer.build_data(train=True, shuffle=True)
    val_data_loader = trainer.build_data(train=False, shuffle=False)
    trainer.pylot_train_data(data_loader)

    model = ClassifyMnist(28 * 28, 125, 64, 10)
    trainer.train(model, data_loader, val_data_loader)

if __name__ == "__main__":
    train_test()