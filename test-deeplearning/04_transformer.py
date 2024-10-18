
import torch
import torch.nn as nn
import torch.optim as optim

# 定义多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性变换
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力权重
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)

        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.out_linear(attn_output)

# 定义前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))

# 定义编码器层
class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, hidden_dim)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.layernorm1(x + attn_output)
        ff_output = self.feed_forward(x)
        return self.layernorm2(x + ff_output)

# 定义编码器
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, hidden_dim):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

# 定义解码器层
class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.enc_dec_attn = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, hidden_dim)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.layernorm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, enc_output, self_mask=None, enc_mask=None):
        attn_output = self.self_attn(x, x, x, self_mask)
        x = self.layernorm1(x + attn_output)
        attn_output = self.enc_dec_attn(x, enc_output, enc_output, enc_mask)
        x = self.layernorm2(x + attn_output)
        ff_output = self.feed_forward(x)
        return self.layernorm3(x + ff_output)

# 定义解码器
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, hidden_dim):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads, hidden_dim) for _ in range(num_layers)])
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, enc_output, self_mask=None, enc_mask=None):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, enc_output, self_mask, enc_mask)
        return self.linear(x)

# 定义 Transformer 模型
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim, num_layers, num_heads, hidden_dim):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_dim, num_layers, num_heads, hidden_dim)
        self.decoder = Decoder(tgt_vocab_size, embed_dim, num_layers, num_heads, hidden_dim)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        return dec_output

# 训练模型
def train(model, optimizer, criterion, src_data, tgt_data, batch_size, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = len(src_data) // batch_size
        for i in range(0, len(src_data), batch_size):
            src = src_data[i:i + batch_size]
            tgt = tgt_data[i:i + batch_size]
            optimizer.zero_grad()
            src_mask = (src!= 0).       (1).       (2)
            tgt_mask = (tgt!= 0).       (1).       (2)
            output = model(src, tgt, src_mask, tgt_mask)
            loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / num_batches}")

# 示例用法
src_vocab_size = 1000
tgt_vocab_size = 1000
embed_dim = 512
num_layers = 6
num_heads = 8
hidden_dim = 2048
batch_size = 64
num_epochs = 10

model = Transformer(src_vocab_size, tgt_vocab_size, embed_dim, num_layers, num_heads, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# 假设已经有了源语言和目标语言的数据集
src_data = torch.randint(1, src_vocab_size, (1000, 20))
tgt_data = torch.randint(1, tgt_vocab_size, (1000, 20))

train(model, optimizer, criterion, src_data, tgt_data, batch_size, num_epochs)