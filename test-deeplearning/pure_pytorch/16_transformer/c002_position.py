import torch
import config
from c001_dataset import build_test_seq
from torch import nn, Tensor

class PositionEmbedding(nn.Module):
    def __init__(self, max_position_len, model_dim):
        super(PositionEmbedding, self).__init__()
        self.max_position_len = max_position_len
        self.model_dim = model_dim
        self.embedding = nn.Embedding(config.max_position_len, config.model_dim)
        self.embedding.weight = nn.Parameter(self.build_embedding_table(), requires_grad=False)
        self.weight = self.embedding.weight

    def forward(self, x):
        return self.embedding(x)
    
    def build_embedding_table(self):
        pos_mat = torch.arange(self.max_position_len).reshape(-1, 1)
        i_mat = torch.pow(10000, torch.arange(0, 8, 2).reshape(1, -1) / self.model_dim)
        pe_embedding_table = torch.zeros(self.max_position_len, self.model_dim)
        pe_embedding_table[:, 0::2] = torch.sin(pos_mat / i_mat)
        pe_embedding_table[:, 1::2] = torch.cos(pos_mat / i_mat)

        print(pe_embedding_table)
        return pe_embedding_table

def build_position_tensor(len:Tensor):
    return torch.cat([torch.unsqueeze(torch.arange(max(len)), 0)for x in len])

def test_postion_embedding():
    pe_embedding = PositionEmbedding(config.max_position_len, config.model_dim)
    # pe_embedding.weight = nn.Parameter(pe_embedding_table)
    # print(pe_embedding.weight)
    src_seq,tgt_seq = build_test_seq()

    
    src_pos = build_position_tensor(config.src_len)
    tgt_pos = build_position_tensor(config.tgt_len)

    print(pe_embedding.forward(src_pos))
    print(pe_embedding.forward(tgt_pos))

if __name__ == "__main__":
    test_postion_embedding()
