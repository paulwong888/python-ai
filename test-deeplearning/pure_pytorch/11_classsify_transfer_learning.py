import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import requests
from torch import nn, Tensor
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image, ImageOps

class ClassifyAlex():
    def __init__(self, output_size: int):
        super().__init__()
        # self.model = models.alexnet(pretrained=True)
        self.model = models.vgg16(pretrained=True)
        for param in self.model.features.parameters():
            param.requires_grad = False
        n_inputs = self.model.classifier[6].in_features
        last_layer = nn.Linear(n_inputs, output_size)
        self.model.classifier[6] = last_layer
        print(self.model)

    def get_model(self):
        return self.model
    
    def get_classes(self):
        return self.classes

class ClassifyAlexTrainer():
    def __init__(self, model: nn.Module, classes):
        self.classes = classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.to_device(model)
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
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(np.full(3, 0.5, float), np.full(3, 0.5, float))
        ])
        return transform
    
    def __transform_new__(self):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
            transforms.ToTensor(),
            transforms.Normalize(np.full(3, 0.5, float), np.full(3, 0.5, float))
        ])
        return transform

    def build_data_set(self, train: bool, shuffle: bool, transform):
        root = "./data/ants_and_bees/train" if train else "./data/ants_and_bees/val"
        dataset = datasets.ImageFolder(root, transform=transform)
        data_loader = DataLoader(dataset, 20, shuffle=shuffle)
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

                epoch_train_running_loss = train_running_loss / len(train_data_loader.dataset)
                epoch_train_running_croorcts = train_running_corrects / len(train_data_loader.dataset)
                train_running_loss_history.append(epoch_train_running_loss)
                train_running_corrects_history.append(epoch_train_running_croorcts.cpu().numpy())

                epoch_val_running_loss = val_running_loss / len(val_data_loader.dataset)
                epoch_val_running_croorcts = val_running_corrects / len(val_data_loader.dataset)
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
        url = "https://pic.rmb.bdstatic.com/bjh/events/07152484884308d17d6a79f42c4966806813.png@h_1280"
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
    classes = ["ant", "bee"]
    trainer = ClassifyAlexTrainer(ClassifyAlex(len(classes)).get_model(), classes)
    train_data_loader = trainer.build_data_set(True, True, trainer.__transform_new__())
    val_data_loader = trainer.build_data_set(False, False, trainer.transform)

    trainer.plot_data_set(train_data_loader)
    # trainer.plot_data_set(val_data_loader)


    lr=0.0001
    trainer.train(train_data_loader, val_data_loader, lr=lr)
    return trainer
    


if __name__ == "__main__":
    trainer = train_test()

    # trainer = ClassifyCnnCifar10Trainer(ClassifyCnnCifar10New())
    trainer.test_web_img()


