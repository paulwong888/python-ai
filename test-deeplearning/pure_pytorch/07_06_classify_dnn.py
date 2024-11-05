import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class ClassifyDnn(nn.Module):
    def __init__(self, input_size, H1, output_size, X: Tensor, y: Tensor):
        super().__init__()
        self.X = X
        self.y = y
        self.linear1 = nn.Linear(in_features=input_size, out_features=H1)
        self.linear2 = nn.Linear(in_features=H1, out_features=output_size)

    def forward(self, X_part: Tensor):
        pred_y = torch.sigmoid(self.linear1.forward(X_part))
        pred_y = torch.sigmoid(self.linear2.forward(pred_y))
        return pred_y
    
    def show_fit(self, title):
        plt.title(title)
        temp_X = self.X.numpy()
        temp_y = self.y.view(self.y.size()[0]).numpy()
        plt.scatter(temp_X[temp_y==0, 0], temp_X[temp_y==0, 1])
        plt.scatter(temp_X[temp_y==1, 0], temp_X[temp_y==1, 1])

        plt.show()
        pass

    def train(self, epochs, batch_size):
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        dats_set = TensorDataset(self.X, self.y)
        print(dats_set)
        data_loader = DataLoader(dataset=dats_set, batch_size=batch_size, shuffle=True)
        losses = []
        for i in range(epochs):
            for (inputs, outputs) in data_loader:
                pred_y = self.forward(inputs)
                loss = loss_fn(pred_y, outputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            print("epochs:", i, "loss", loss.item())

        plt.plot(range(epochs), losses)
        plt.show()

        # result_y = self.forward(self.X).detach().numpy()
        # input_x = self.X.numpy()
        # print(input_x[0].shape, result_y[0].shape)
        # plt.scatter(input_x[0].reshape(500), result_y[0].reshape(500))
        # plt.show()


    def predit(self, X) -> int:
        pre_y = self.forward(X)
        if pre_y >= 0.5:
            return 1
        else:
            return 0



def train_test():
    n_pts = 500
    (X_numpy, y_numpy) = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2)
    # X_numpy -> [[1.1, 2.2], [3.3, 4.4]] 二维数组
    # y_numpy -> [1, 0] 一维数组
    X = torch.tensor(X_numpy).float()
    print(X_numpy.shape[1], y_numpy.shape)
    y = torch.tensor(y_numpy).float().view(n_pts, -1)
    # print(X, y)

    input_features = X_numpy.shape[1]
    output_features = 1
    model = ClassifyDnn(input_features, 4, output_features, X, y)
    model.show_fit("Initial Model")

    model.train(500, n_pts)

    point1 = torch.tensor([0.1, 0.1])
    point2 = torch.tensor([0.5, 0.7])

    print(model.predit(point1))
    print(model.predit(point2))


if __name__ == "__main__":
    train_test()