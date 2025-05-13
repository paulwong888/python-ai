import torch

# 单词表大小
max_num_src_wrods = 8
max_num_tgt_wrods = 8
model_dim = 8

# 序列最大长度
max_src_seq_len = 5
max_tgt_seq_len = 5

max_position_len = 5

src_len = torch.IntTensor([2, 4])
tgt_len = torch.IntTensor([4, 3])