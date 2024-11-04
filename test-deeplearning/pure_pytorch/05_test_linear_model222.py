import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class LR(nn.Module):
    def __init__(self, in_features, out_features, X, y):
        super().__init__()
        self.X = X
        self.y = y
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self):
        pred = self.linear(self.X)
        return pred
    
    def get_params(self):
        (w, b) = self.parameters()
        return (w.item(), b.item())

    def show_fit(self, title):
        plt.title = title
        plt.scatter(X, y)

        (w, b) = self.get_params()
        print("w=", w, "b=", b)
        temp_x = np.array([-30, 30])
        temp_y = w * temp_x + b
        print("x=", temp_x, "y=", temp_y)

        plt.plot(temp_x, temp_y, "r")
        plt.show()

    def train(self, epchs, batch_size):
        
        tensor_dataset = TensorDataset(X, y)
        data_loader = DataLoader(tensor_dataset, batch_size, shuffle=True)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(self.linear.parameters(), lr=0.01)
        losses = []

        for i in range(epchs):
            for (inputs, outputs) in data_loader:
                pre_y = self.forward()
                loss = loss_fn(pre_y, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print("epch", i, "loss: ", loss.item())

            losses.append(loss.item())
        plt.plot(range(epchs), losses)
        plt.show()
    
X = torch.randn(100, 1) * 10
y = X + 3 * torch.randn(100, 1)

model = LR(1, 1, X, y)
model.show_fit("Initial Model")

model.train(200, 100)
model.show_fit("Trainer Model")