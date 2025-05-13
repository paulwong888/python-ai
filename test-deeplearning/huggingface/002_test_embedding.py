from transformers import AutoTokenizer,BertModel
from scipy.spatial.distance import cosine

model_name = "bert-base-cased"

model = BertModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

sentence = "Tokenize me this please"

def predict(sentence:str):
    tokenizer.tokenize(sentence)
    tokenizer.encode(sentence)
    encode_sentence = tokenizer(sentence, return_tensors="pt")
    type(encode_sentence["input_ids"])
    tokenizer.decode(encode_sentence["input_ids"][0])

    output = model(**encode_sentence)

    last_hidden_state = output.last_hidden_state
    pooler_output = output.pooler_output

    last_hidden_state.shape
    pooler_output.shape

    return output[0]

# def predict(text):


sentence1 = "There was a fly drinking from my soup"
sentence2 = "To become a comercial pilot, he had to fly for 1500 hours."

sentence1_tokens = tokenizer.tokenize(sentence1)
sentence2_tokens = tokenizer.tokenize(sentence2)
index_in_sentence1 = sentence1_tokens.index("fly")
index_in_sentence2 = sentence2_tokens.index("fly")

out1 = predict(sentence1)
out2 = predict(sentence2)
out1.shape
out2.shape

embedding1 = out1[0:, index_in_sentence1, :][0].detach()
embedding2 = out2[0:, index_in_sentence2, :][0].detach()

embedding1.shape
embedding2.shape
cosine(embedding1, embedding2)