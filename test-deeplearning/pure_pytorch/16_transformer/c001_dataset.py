import torch
import torch.nn.functional as F
import config
from torch import Tensor, nn

class Dataset():

    def build_seq(self, max_num_words, sentence_size:Tensor, max_seq_len):
        seq = [torch.randint(1, max_num_words, (x,)) for x in sentence_size]
        seq = [F.pad(x, (0, max_seq_len-len(x)) ) for x in seq]
        seq = [torch.unsqueeze(x, 0) for x in seq]
        seq = torch.cat(seq)
        return seq

def build_test_seq():
            
    dataset = Dataset()
    src_seq = dataset.build_seq(config.max_num_src_wrods, config.src_len, config.max_src_seq_len)
    tgt_seq = dataset.build_seq(config.max_num_tgt_wrods, config.tgt_len, config.max_tgt_seq_len)

    print(src_seq)
    print(tgt_seq)

    src_embedding = nn.Embedding(config.max_num_src_wrods, config.model_dim)

    print(src_embedding.weight)
    print(src_embedding(tgt_seq))
    return src_seq, tgt_seq

if __name__ == "__main__":
    build_test_seq()