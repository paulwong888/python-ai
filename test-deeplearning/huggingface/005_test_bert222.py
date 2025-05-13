import torch
import numpy as np
import pandas as pd
from transformers import BertForQuestionAnswering, BertTokenizerFast
from scipy.special import softmax

class BertQAPredict():
    def __init__(self):
        self.model_name = "deepset/bert-base-cased-squad2"
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(self.model_name)
        self.model = BertForQuestionAnswering.from_pretrained(self.model_name)
        pass

    """
    [id,id,id,id...]512个->
    [概率,概率,概率,概率...]512个开始字符的概率/[概率,概率,概率,概率...]512个结束字符的概率->
    [开始字符最高概率的在512中位置,结束字符最高概率的在512中位置]->
    [开始字符最高概率的id值:结束字符最高概率的id值]->
    通过tokenizer映射回字符
    """
    def predict_context512(self, question, context):
        inputs = self.tokenizer(question, context, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        start_score_list, end_score_list = softmax(outputs.start_logits)[0], softmax(outputs.end_logits)[0]
        highest_start_score_index = np.argmax(start_score_list)
        highest_end_score_index = np.argmax(end_score_list)
        answer_index_list = inputs.input_ids[0][highest_start_score_index : highest_end_score_index + 1]
        answer_token_list = self.tokenizer.convert_ids_to_tokens(answer_index_list)
        answer_str = self.tokenizer.convert_tokens_to_string(answer_token_list)

        confidence_score = (start_score_list[highest_start_score_index] + end_score_list[highest_end_score_index]) / 2
        if answer_str == self.tokenizer.cls_token:
            answer_str = None
        return answer_str, confidence_score
    
    """
    "长字符"->
    ["段落","段落","段落","段落"..]->
    ["段落"+"段落","段落"+"段落"..]->
    """
    def chunk_sentences(self, sentences:str, step:int, overlap:int):
        sentences_list = sentences.split("\n") #以换行符为标切割, 不是以句号切割, 这样其实就是以段落做切割
        print(len(sentences_list))
        results_list = []
        for i in range(0, len(sentences_list), step):
            chunk_list = sentences_list[i : i + step + overlap]
            results_list.append("\n".join(chunk_list))
        return results_list

    def predict_context_over512(self, question, context):
        context_list = self.chunk_sentences(context, 2, 1)
        temp_answer = None
        for sentence in context_list:
            answer = self.predict_context512(question, sentence)
            if answer[0]:
                if temp_answer == None:
                    temp_answer = answer
                else:
                    if answer[1] > temp_answer[1]:
                        temp_answer = answer
        return temp_answer

def test_chunk_sentences(sentences):
    bert_qa_predict = BertQAPredict()
    test_sentences:str = """Setence 1.
    Setence 2.
    Setence 3.
    Setence 4.
    Setence 5.
    Setence 6.
    Setence 7.
    Setence 8.
    Setence 9.
    Setence 10.
    """
    # test_sentences = "asdfasdfasdfad"
    print(type(test_sentences))
    # test_sentences.split("\n")
    sentences_list = test_sentences.split("\n") #以换行符为标切割, 不是以句号切割, 这样其实就是以段落做切割
    print(bert_qa_predict.chunk_sentences(test_sentences, 2, 1))
    result = bert_qa_predict.chunk_sentences(sentences, 2, 1)
    print(len(result))

def test_predict():
    bert_qa_predict = BertQAPredict()
    question = "what is Coffee?"
    # answer = "a beverage prepared from roasted coffee beans"
    print(bert_qa_predict.predict_context512(question, context))
    bert_qa_predict = BertQAPredict()

    question = "What are the most common coffee beans??" 
    print(bert_qa_predict.predict_context_over512(question, context3))

    question = "How many people are dependent on coffee for their income?" 
    print(bert_qa_predict.predict_context_over512(question, context3))

if __name__ == "__main__":
    context = """"Coffee is a beverage brewed from roasted, ground coffee beans. Darkly colored, bitter, and slightly acidic, coffee has a stimulating effect on humans, primarily due to its caffeine content. It has the highest sales in the world market for hot drinks.[2]

    Coffee production begins when the seeds of the Coffea plant's fruits (coffee cherries) are separated to produce unroasted green coffee beans. The beans are roasted and then ground into fine particles. Coffee is brewed from the ground roasted beans, which are typically steeped in hot water before being filtered out. It is usually served hot, although chilled or iced coffee is common. Coffee can be prepared and presented in a variety of ways (e.g., espresso, French press, caffè latte, or already-brewed canned coffee). Sugar, sugar substitutes, milk, and cream are often added to mask the bitter taste or enhance the flavor. There are also various coffee substitutes.

    Though coffee is now a global commodity, it has a long history tied closely to food traditions around the Red Sea. The earliest credible evidence of coffee drinking as the modern beverage appears in modern-day Yemen in southern Arabia in the middle of the 15th century in Sufi shrines, where coffee seeds were first roasted and brewed in a manner similar to how it is now prepared for drinking.[3] The coffee beans were procured by the Yemenis from the Ethiopian Highlands via coastal Somali intermediaries, and cultivated in Yemen. By the 16th century, the drink had reached the rest of the Middle East and North Africa, later spreading to Europe.

    The two most commonly grown coffee bean types are C. arabica and C. robusta.[4] Coffee plants are cultivated in over 70 countries, primarily in the equatorial regions of the Americas, Southeast Asia, the Indian subcontinent, and Africa. Green, unroasted coffee is traded as an agricultural commodity. The global coffee industry is massive and worth $495.50 billion as of 2023.[5] In the same year, Brazil was the leading grower of coffee beans, producing 35% of the world's total, followed by Vietnam and Colombia. While coffee sales reach billions of dollars annually worldwide, coffee farmers disproportionately live in poverty. Critics of the coffee industry have also pointed to its negative impact on the environment and the clearing of land for coffee-growing and water use.
    """


    context2 = """"Meanwhile, coffee had been introduced to Brazil in 1727, although its cultivation did not gather momentum until independence in 1822.[35] After this time, massive tracts of rainforest were cleared for coffee plantations, first in the vicinity of Rio de Janeiro and later São Paulo.[36] Brazil went from having essentially no coffee exports in 1800 to being a significant regional producer in 1830, to being the largest producer in the world by 1852. In 1910–1920, Brazil exported around 70% of the world's coffee, Colombia, Guatemala, and Venezuela exported 15%, and Old World production accounted for less than 5% of world exports.[37]

    Many countries in Central America took up cultivation in the latter half of the 19th century, and almost all were involved in the large-scale displacement and exploitation of the indigenous people. Harsh conditions led to many uprisings, coups, and bloody suppression of peasants.[38] The notable exception was Costa Rica, where lack of ready labor prevented the formation of large farms. Smaller farms and more egalitarian conditions ameliorated unrest over the 19th and 20th centuries.[39]

    Rapid growth in coffee production in South America during the second half of the 19th century was matched by an increase in consumption in developed countries, though nowhere has this growth been as pronounced as in the United States, where a high rate of population growth was compounded by doubling of per capita consumption between 1860 and 1920. Though the United States was not the heaviest coffee-drinking nation at the time (Belgium, the Netherlands and Nordic countries all had comparable or higher levels of per capita consumption), due to its sheer size, it was already the largest consumer of coffee in the world by 1860, and, by 1920, around half of all coffee produced worldwide was consumed in the US.[37]

    Coffee has become a vital cash crop for many developing countries. Over one hundred million people in developing countries have become dependent on coffee as their primary source of income. It has become the primary export and economic backbone for African countries like Uganda, Burundi, Rwanda, and Ethiopia,[40] as well as many Central American countries.
    """
    context3 = context + context2

    # test_chunk_sentences(context3)
    test_predict()