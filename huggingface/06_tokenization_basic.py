#把字符转成数字

string = "Only those who will risk going too far can possibly find out how far one can go."
tokenized_str = string.split()
print(tokenized_str)

token_word2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_str)))}
print(token_word2idx)

input_ids = [token_word2idx[token] for token in tokenized_str]
print(input_ids)