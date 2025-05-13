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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)

        dats_set = TensorDataset(self.X, self.y)
        print(dats_set)
        data_loader = DataLoader(dataset=dats_set, batch_size=batch_size, shuffle=True)
        losses = []
        for i in range(epochs):
            super().train()
            train_loss = 0.0
            train_correct = 0
            train_batches = 0
            for (inputs, targets) in data_loader:
                pred_y = self.forward(inputs)
                loss = loss_fn(pred_y, targets)

                # Accumulate metrics.
                # _, indices = torch.max(pred_y.data, 1)
                # train_correct += (indices == targets).sum().item()
                # train_batches += 1
                train_loss += loss.item() * inputs.size(0)
                # train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss += train_loss/len(data_loader.sampler)
            # train_loss += train_loss/train_batches
            # train_acc = train_correct/(train_batches * batch_size)
            losses.append(train_loss)
            # losses.append(loss.item())
            # x = x @ x
            # print(f"epochs: {i} loss {loss.item():.4f}")
            print(f"epochs: {i} loss {train_loss:.4f}")

        plt.plot(range(epochs), losses)
        plt.show()

        # result_y = self.forward(self.X).detach().numpy()
        # input_x = self.X.numpy()
        # print(input_x[0].shape, result_y[0].shape)
        # plt.scatter(input_x[0].reshape(500), result_y[0].reshape(500))
        # plt.show()

    def test(self, X_test: Tensor, y_test: Tensor):
        batch_size=100
        data_set = TensorDataset(X_test, y_test)
        data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True)

        # self.eval()
        test_lost = 0
        correct = 0
        with torch.no_grad():
            for input, label in data_loader:
                output = self.forward(input)
                pred_y = output.argmax(dim=1, keepdim=True)
                correct += pred_y.eq(label.view_as(pred_y)).sum().item()
            
            print(correct)
            print("Acuracy: %.3f" % (correct / len(data_set)))


    def predit(self, X) -> int:
        pre_y = self.forward(X)
        if pre_y >= 0.5:
            return 1
        else:
            return 0
        
    def plot_decision_boundary(self):
        x_span = np.linspace(min(self.X[:, 0]), max(self.X[:, 0] + 0.25))
        y_span = np.linspace(min(self.X[:, 1]), max(self.X[:, 1]) + 0.25)
        xx, yy = np.meshgrid(x_span, y_span)
        grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float()
        pred_func = self.forward(grid)
        z = pred_func.view(xx.shape).detach().numpy()
        plt.contour(xx, yy, z)
        self.show_fit("Trained Model")
        # plt.show()

def build_data(n_pts):
    (X_numpy, y_numpy) = datasets.make_circles(n_samples=n_pts, random_state=123, noise=0.1, factor=0.2)
    # X_numpy -> [[1.1, 2.2], [3.3, 4.4]] 二维数组
    # y_numpy -> [1, 0] 一维数组
    X = torch.tensor(X_numpy).float()
    print(X_numpy.shape[1], y_numpy.shape)
    y = torch.tensor(y_numpy).float().view(n_pts, -1)
    # print(X, y)
    return X, y

def train_test():
    n_pts = 500
    X, y = build_data(n_pts)
    input_features = X.shape[1]
    output_features = 1
    model = ClassifyDnn(input_features, 4, output_features, X, y)
    model.show_fit("Initial Model")

    model.train(500, n_pts)

    point1 = torch.tensor([0.1, 0.1])
    point2 = torch.tensor([0.5, 0.7])

    print(model.predit(point1))
    print(model.predit(point2))

    X_test, y = build_data(200)
    model.test(X_test, y)

    model.plot_decision_boundary()


if __name__ == "__main__":
    train_test()