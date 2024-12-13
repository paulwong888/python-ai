import pandas as pd
from transformers import BertModel, AutoTokenizer

model_name = "bert-base-cased"

model = BertModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

sentence = "When life gives you lemons, don't make lemonade."

#查看分解后的单字数组
sentence_tokens = tokenizer.tokenize(sentence)
#查看分解后的单字所对应的索引数组
sentence_token_ids = tokenizer.encode(sentence)
print(len(sentence_tokens))
print(len(sentence_token_ids))

vocab = tokenizer.vocab
print(len(vocab))
# 以DataFrame的方式查看tokenizer中的语汇表
vocab_df = pd.DataFrame({"token":vocab.keys(), "token_id": vocab.values()})
vocab_df = vocab_df.sort_values(by="token_id").set_index("token_id")
print(vocab_df.iloc[101])
print(vocab_df.iloc[102])

list(zip(sentence_tokens, sentence_token_ids))
tokenizer.decode(token_ids=sentence_token_ids)
tokenizer.decode(token_ids=sentence_token_ids[1:-1])

# 组装成model所需的输入参数
sentence_tokennizer_out = tokenizer(sentence)

sentence2 = sentence.replace("don't", "")
sentence_tokennizer_out2 = tokenizer([sentence, sentence2], padding=True)
tokenizer.decode(sentence_tokennizer_out2["input_ids"][0])
tokenizer.decode(sentence_tokennizer_out2["input_ids"][1])