import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

class ClassifiDnn(nn.Module):
    def __init__(self, input_size, H1, output_size, X: Tensor, y: Tensor):
        super().__init__()
        self.X = X
        self.y = y
        self.linear1 = nn.Linear(in_features=input_size, out_features=H1)
        self.linear2 = nn.Linear(in_features=H1, out_features=output_size)

    def forward(self, X_part):
        # print(f"X_part shape: {X_part.shape}")
        pred_y = F.sigmoid(self.linear1(X_part))
        pred_y = F.sigmoid(self.linear2(pred_y))
        return pred_y
    
    def show_fit(self, title):
        plt.title(title)
        X_numpy = self.X.numpy()
        y_numpy = self.y.view(-1).numpy()
        plt.scatter(X_numpy[y_numpy==0, 0], X_numpy[y_numpy==0, 1])
        plt.scatter(X_numpy[y_numpy==1, 0], X_numpy[y_numpy==1, 1])
        plt.show()

    def train(self, batch_size):
        epochs = 500
        loss_func = nn.BCELoss()
        optimizer = Adam(self.parameters(), lr=0.1)
        data_set = TensorDataset(self.X, self.y)
        # print(data_set)
        data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)

        losses = []
        
        for i in range(epochs):
            super().train()
            train_loss = 0.0
            train_corrects = 0.0
            for (inputs, targets) in data_loader:
                # print(inputs, targets)
                outputs = self.forward(inputs)
                loss = loss_func(outputs, targets)
                
                train_loss += loss.item()
                # _, preds = torch.max(outputs, 1)
                # train_corrects += torch.sum(preds == targets.data)
                # print()
                # print(f"targets.shape : {targets.shape}")
                # print(f"outputs : {outputs}")
                # print(f"preds : {preds.shape}")
                # print(f"train_corrects : {train_corrects}")
                # print()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                train_loss = train_loss/len(data_loader)
                # epoch_corrects = train_corrects/len(data_loader)
                losses.append(train_loss)
                # print(f"epoch: {i} loss {train_loss} acc {epoch_corrects}")
                print(f"epoch: {i} loss {train_loss}")
                break
            break
        else:
            plt.plot(range(epochs), losses)
            plt.show()

    def test(self, X: Tensor, y: Tensor, batch_size):
        data_set = TensorDataset(X, y)
        data_loader = DataLoader(data_set, batch_size, shuffle=True)
        
        corrects = 0
        with torch.no_grad():
            for (inputs, targets) in data_loader:
                outputs = self.forward(inputs)
                pred_y = outputs.argmax(dim=1, keepdim=True)
                corrects += pred_y.eq(targets.view_as(pred_y)).sum().item()

            print(corrects)
            print("Acuracy: %.3f" % (corrects / len(data_set)))

    
def build_data(p_nts):
    X_numpy, y_numpy = datasets.make_circles(n_samples=p_nts, random_state=123, noise=0.1,factor=0.2)
    print(X_numpy.shape, y_numpy.shape)
    X = torch.tensor(X_numpy).float()
    y = torch.tensor(y_numpy).float().view(p_nts, 1)
    # print(X, y)
    return (X, y)

def train_test():
    n_pts = 500
    (X, y) = build_data(n_pts)
    input_features = X.shape[1]
    output_features = y.shape[1]

    model = ClassifiDnn(input_features, 4, output_features, X, y)
    model.show_fit("Initial Model")

    model.train(100)

    (X_test, y_test) = build_data(200)
    model.test(X_test, y_test, 20)

if __name__ == "__main__":
    train_test()
