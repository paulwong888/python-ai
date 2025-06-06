import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor

class Classify(nn.Module):
    def __init__(self, input_size, output_size, X: Tensor, y: Tensor):
        super().__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=output_size)
        self.X = X
        self.y = y
        print(y.size()[0])
    
    def forward(self, X_part):
        pred_y = torch.sigmoid(self.linear.forward(X_part))
        return pred_y
    
    def get_params(self):
        (w, b) = self.parameters()
        (w1, w2) = w.view(2)
        b1 = b.item()
        return (w1.item(), w2.item(), b1)
    
    def show_fit(self, title):
        plt.title(title)
        X_temp = self.X.numpy()
        y_temp = self.y.view(100).numpy()
        # X[y==0, 0] 返回的是一个一维数组，包含了所有属于第一个簇（标签为0）的数据点的x坐标
        plt.scatter(X_temp[y_temp==0, 0], X_temp[y_temp==0, 1])
        # X[y==1, 0] 返回的是一个一维数组，包含了所有属于第一个簇（标签为1）的数据点的x坐标
        plt.scatter(X_temp[y_temp==1, 0], X_temp[y_temp==1, 1])

        (w1, w2, b1) = self.get_params()
        x1 = np.array([-2.0, 2.0])
        y1 = (w1*x1 + b1)/-w2
        plt.plot(x1, y1, "r")
        plt.show()

    def train(self, epochs, batch_size):
        data_set = TensorDataset(self.X, self.y)
        data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)

        loss_fn = nn.BCELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        losses = []
        for i in range(epochs):
            for (inputs, outputs) in data_loader:
                pre_y = self.forward(inputs)
                loss = loss_fn(pre_y, outputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
            print("epochs:", i, "loss", loss.item())
        
        plt.plot(range(epochs), losses)
        plt.show()

    def predit(self, X) -> int:
        pre_y = self.forward(X)
        if pre_y >= 0.5:
            return 1
        else:
            return 0

def train_test():
    n_pts = 100

    """
    使用make_blobs函数生成数据集，X 是二维数组，y是一维数组
    设置参数：n_pts 表示生成的数据点数量为100，
    centers 表示数据点的中心位置，这里有两个中心点，分别在(-0.5, 0.5)和(0.5, -0.5)。
    random_state参数设置为123以确保结果的可重复性，
    cluster_std参数设置为0.4，表示每个簇的标准差。
    """
    centers = [[-0.5, 0.5], [0.5, -0.5]]
    (X_numpy, y_numpy) = datasets.make_blobs(n_samples=n_pts, random_state=123, centers=centers, cluster_std=0.4)
    print(X_numpy, y_numpy)

    X = torch.Tensor(X_numpy)
    y = torch.Tensor(y_numpy.reshape(100, 1))
    model = Classify(2,1, X, y)
    model.show_fit("Initial Model")

    model.train(1000, 100)
    model.show_fit("Trained Model")

    point1 = torch.tensor([1.0, -1.0])
    point2 = torch.tensor([-1.0, 1.0])
    print(f"Red point positive probalility = {model.forward(point1).item()}")
    print(f"Black point positive probalility = {model.forward(point2).item()}")
    print(f"Red point in class {model.predit(point1)}")
    print(f"Black point in class {model.predit(point2)}")

    plt.plot(point1.numpy()[0], point1.numpy()[1], "ro")
    plt.plot(point2.numpy()[0], point2.numpy()[1], "ko")
    model.show_fit("Trained Model")

if __name__ == "__main__":
    train_test()