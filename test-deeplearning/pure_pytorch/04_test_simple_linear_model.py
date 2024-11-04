import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)
model = nn.Linear(in_features=1, out_features=1)
print(model.bias, model.weight)

x = torch.tensor([[2.0], [3.3]])
print(model(x))