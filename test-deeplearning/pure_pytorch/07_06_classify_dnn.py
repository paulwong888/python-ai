import torch
import torch.nn as nn

class ClassifyDnn(nn.Module):
    def __init__(self, input_size, H1, output_size):
        super().__init__()
        self.linear1 = nn.Linear(
            in_features=input_size,
            out_features=H1
        )