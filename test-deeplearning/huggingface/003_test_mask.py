from transformers import AutoTokenizer,AutoModelForMaskedLM
from scipy.special import softmax
import numpy as np

model_anme = "bert-base-cased"
model = AutoModelForMaskedLM.from_pretrained(model_anme)
tokenizer = AutoTokenizer.from_pretrained(model_anme)

mask = tokenizer.mask_token

sentence = f"I want to {mask} pizza for tonight."

sentence_tokens = tokenizer.tokenize(sentence)
sentence_mask_index = sentence_tokens.index(mask)

sentence_encoded = tokenizer(sentence, return_tensors="pt")

outputs = model(**sentence_encoded)
"""
Logits 是模型预测的原始输出，即在应用任何激活函数（如 softmax)之前的分数。
在语言模型中，这些 logits 表示每个可能的词汇在给定上下文中出现的概率分数。
"""
logits = outputs.logits.detach().numpy()[0]
logits.shape

mask_logits = logits[sentence_mask_index + 1]
confidence_scores = softmax(mask_logits)

"""
1. confidence_scores:这是一个数组, 包含了模型对每个可能的词汇填充掩码位置的概率分数。
2. np.argsort(confidence_scores):argsort 函数返回的是数组元素从小到大的索引值。
   例如, 如果 confidence_scores 是 [0.1, 0.3, 0.2], 那么 np.argsort(confidence_scores) 
   将会返回 [0, 2, 1], 因为第一个元素 (索引0) 是最小的, 第二个元素 (索引2) 是次小的, 依此类推。
3. [::-1]:这是一个切片操作, 用于反转数组。在上面的例子中, [0, 2, 1][::-1] 将会得到 [1, 0, 2], 
   即从大到小的索引顺序。
4. [:5]:这又是一个切片操作, 用于取出数组的前5个元素。
   如果 confidence_scores 有超过5个元素, 那么这个操作将会限制结果只包含概率最高的前5个词汇的索引。
"""
for i in np.argsort(confidence_scores)[::-1][:5]:
    pred_token = tokenizer.decode(i)
    score = confidence_scores[i]
    # print(pred_token, score)
    print(sentence.replace(mask, pred_token), score)
"""
have 0.25729114
get 0.17849593
eat 0.15555528
make 0.1142242
order 0.09823022
"""