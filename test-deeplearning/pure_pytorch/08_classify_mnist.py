import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import requests
from PIL import Image, ImageOps

class ClassifyMnist(nn.Module):
    def __init__(self, input_features, H1, H2, output_features):
        super().__init__()
        # self.data_loader = data_loader
        # self.val_data_loader = val_data_loader
        self.linear1 = nn.Linear(in_features=input_features, out_features=H1)
        self.linear2 = nn.Linear(in_features=H1, out_features=H2)
        self.linear3 = nn.Linear(in_features=H2, out_features=output_features)

    def forward(self, X_part: Tensor):
        X_part = F.relu(self.linear1(X_part))
        X_part = F.relu(self.linear2(X_part))
        X_part = self.linear3(X_part)
        return X_part
    
    def train(self, data_loader: DataLoader, val_data_loader: DataLoader):
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        epochs = 20
        running_history = []
        running_corrects_history = []
        val_running_history = []
        val_running_corrects_history = []

        for i in range(epochs):
            running_loss = 0.0 
            running_corrects = 0.0 
            val_running_loss = 0.0 
            val_running_corrects = 0.0 
            for (inputs, targets) in data_loader:
                print(f"inputs.shape : {inputs.shape}")
                print(f"before view : {inputs}")
                inputs = inputs.view(inputs.shape[0], -1)
                print(f"after view : {inputs.shape}")
                outputs = self.forward(inputs)
                loss = loss_func(outputs, targets)
                running_loss += loss.item()

                # print(f"outputs.shape : {outputs.shape}")
                # print(f"outputs value : {outputs}")
                # print(f"torch.max(outputs, 1) : {torch.max(outputs, 1)}")
                _, preds = torch.max(outputs, 1)
                # print(f"_, preds = torch.max(outputs, 1) -> _ : {_}, preds : {preds}")
                # print(f"targets.data : {targets.data}")
                # print(f"torch.sum(preds == targets.data) : {torch.sum(preds == targets.data)}")
                running_corrects += torch.sum(preds == targets.data)
                # print(f"len(self.data_loader) : {len(self.data_loader)}")

                # if i == 0:
                #     break
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    for (val_inputs, val_targets) in val_data_loader:
                        val_outputs = self.forward(val_inputs.view(val_inputs.shape[0], -1))
                        val_loss = loss_func(val_outputs, val_targets)
                        (_, val_preds) = torch.max(val_outputs, 1)
                        
                        val_running_loss += val_loss.item()
                        val_running_corrects += torch.sum(val_preds==val_targets)

                epoch_loss = running_loss/len(data_loader)
                epoch_acc = running_corrects/len(data_loader)
                running_history.append(epoch_loss)
                running_corrects_history.append(epoch_acc)

                val_epoch_loss = val_running_loss/len(val_data_loader)
                val_epoch_acc = val_running_corrects/len(val_data_loader)
                val_running_history.append(val_epoch_loss)
                val_running_corrects_history.append(val_epoch_acc)
                print(f"epoch: {i}, loss {epoch_loss:.4f}, acc {epoch_acc:.4f}")
                print(f"epoch: {i}, val_loss {val_epoch_loss:.4f}, val_acc {val_epoch_acc:.4f}")
                print()
                break
        else:
            plt.plot(running_history, label="trainning loss")
            plt.plot(val_running_history, label="validation loss")
            plt.legend()
            plt.show()
            plt.plot(running_corrects_history, label="trainning corrects")
            plt.plot(val_running_corrects_history, label="validation corrects")
            plt.show()


def build_data(train: bool, shuffle: bool):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ])
    trainning_data_set = datasets.MNIST("./mnist", train=train, transform=transform)
    print(trainning_data_set)
    trainning_data_loader = DataLoader(dataset=trainning_data_set, batch_size=100, shuffle=shuffle)
    return trainning_data_loader

def img_convert(tensor: Tensor):
    img = tensor.clone().detach().numpy()
    img = img.transpose(1, 2, 0)
    # print(f"img_convert: img.shape : {img.shape}")
    # print(f"np.ndarray((0.5, 0.5, 0.5)) : {np.ndarray((0.5, 0.5, 0.5))}")
    # img = img * np.full(1, 0.5, float) + np.full(1, 0.5, float)
    # img = img * np.ndarray((0.5, 0.5, 0.5)) + np.ndarray((0.5, 0.5, 0.5))
    img = img * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    # print(f"img_convert: img.shape : {img.shape}")
    img = img.clip(0, 1)
    return img

def plot_data_set(data_loader: DataLoader):
    data_iterator = iter(data_loader)
    (inputs, labels) = data_iterator.__next__()
    figure = plt.figure(figsize=(25, 4))

    for idx in range(20):
        ax = figure.add_subplot(2, 10, idx+1)
        plt.imshow(img_convert(inputs[idx]))
        ax.set_title(labels[idx].item())
    plt.show()
    pass

def plot_validation(val_data_loader: DataLoader, model: ClassifyMnist):
    dataiter = iter(val_data_loader)
    (images, labels) = dataiter.__next__()
    images_ = images.view(images.shape[0], -1)
    outputs = model.forward(images_)
    (_, preds) = torch.max(outputs, 1)

    figure = plt.figure(figsize=(25, 4))
    for i in range(20):
        ax = figure.add_subplot(2, 10, i+1, xticks=[], yticks=[])
        plt.imshow(img_convert(images[i]))
        ax.set_title(f"{str(preds[i].item())}({str(labels[i].item())})", color=("green" if preds[i] == labels[i] else "red"))
    plt.show()


def train_test():
    data_loader = build_data(train=True, shuffle=True)
    val_data_loader = build_data(train=False, shuffle=False)
    plot_data_set(data_loader)

    model = ClassifyMnist(28 * 28, 125, 64, 10)
    model.train(data_loader, val_data_loader)

    test_web_img()

    plot_validation(val_data_loader=val_data_loader, model=model)



# @staticmethod
def test_web_img():
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
    ])
    # url = "https://www.dhlabels.com/123-large_default/placas-y-numeros-number-5-100mm.jpg"
    # url = "https://pic.izihun.com/pic/art_font/2019/01/16/10/png_temp_1570538605460_0534.jpg"
    url = "https://d00.paixin.com/thumbs/2899123/37531951/staff_1024.jpg"
    response = requests.get(url=url, stream=True)
    img = Image.open(response.raw)
    img = ImageOps.invert(img)
    img = img.convert("1")
    img = transform(img)
    print(img.shape)

    temp_img = img_convert(img)
    plt.imshow(temp_img)
    plt.show()

    print(type(img))
    # print(img.shape)
    # img = torch.from_numpy(img).float()


    print(img.shape)
    print(img.shape[0])
    img = img.view(img.shape[0], -1)
    print(img.shape)

    # img = img.view(784, -1)
    print(f"5-----> {img.shape}")

    model = ClassifyMnist(28 * 28, 125, 64, 10)
    output = model.forward(img)

    _, pred = torch.max(output, 1)
    print(pred)

if __name__ == "__main__":
    train_test()
    # test_web_img()

    a = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    print(a.shape, a)
    a = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    print(a.shape, a)