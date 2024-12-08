import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, voca_len, word_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=voca_len, embedding_dim=word_dim)

    def forward(self, x):
        return self.embedding(x)
    
if __name__ == "__main__":
    embedding = Embedding(10, 4)
    input = torch.LongTensor([1, 2, 3, 4])
    print(embedding(input))
    print(embedding(input))