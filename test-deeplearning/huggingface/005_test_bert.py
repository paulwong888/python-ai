import torch
import pandas as pd
import numpy as np
import plotly.express as px
from transformers import BertForQuestionAnswering, BertTokenizerFast
from scipy.special import softmax

context = "The giraffe is a large African hoofed mammal belonging to the genus Giraffa. It is the tallest living terrestrial animal and the largest ruminant on Earth. Traditionally, giraffes have been thought of as one species, Giraffa camelopardalis, with nine subspecies. Most recently, researchers proposed dividing them into four extant species due to new research into their mitochondrial and nuclear DNA, and individual species can be distinguished by their fur coat patterns. Seven other extinct species of Giraffa are known from the fossil record."
# question = "How many giraffe species are there?"
question = "What is dog"

model_name = "deepset/bert-base-cased-squad2"

tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

inputs = tokenizer(question, context, truncation=True, return_tensors="pt")

tokenizer.tokenize(context)

with torch.no_grad():
    outputs = model(**inputs)

start_scores, end_scores = softmax(outputs.start_logits)[0], softmax(outputs.end_logits)[0]
start_scores.shape
end_scores.shape

scores_df = pd.DataFrame({
    "Token Position": list(range(len(start_scores))) * 2,
    "Score": list(start_scores) + list(end_scores),
    "Score Type": ["Start"] * len(start_scores) + ["End"] * len(end_scores),
})

px.bar(scores_df, x="Token Position", y="Score", color="Score Type", barmode="group")

start_idx = np.argmax(start_scores)
end_idx = np.argmax(end_scores)

answer_ids = inputs.input_ids[0][start_idx: end_idx+1]
answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids)
answer = tokenizer.convert_tokens_to_string(answer_tokens)
answer

def predict_answer(context, question):
    inputs = tokenizer(question, context, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    start_scores, end_scores = softmax(outputs.start_logits)[0], softmax(outputs.end_logits)[0]
    start_idx = np.argmax(start_scores)
    end_idx = np.argmax(end_scores)

    confidence_score = (start_scores[start_idx] + end_scores[end_idx]) / 2

    answer_ids = inputs.input_ids[0][start_idx: end_idx+1]
    answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids)
    answer = tokenizer.convert_tokens_to_string(answer_tokens)
    if answer == tokenizer.cls_token:
        answer = None
    return answer, confidence_score

question = "What is dog"
predict_answer(context, question)

context = """Coffee is a beverage brewed from roasted, ground coffee beans. Darkly colored, bitter, and slightly acidic, coffee has a stimulating effect on humans, primarily due to its caffeine content. It has the highest sales in the world market for hot drinks.[2]

Coffee production begins when the seeds of the Coffea plant's fruits (coffee cherries) are separated to produce unroasted green coffee beans. The beans are roasted and then ground into fine particles. Coffee is brewed from the ground roasted beans, which are typically steeped in hot water before being filtered out. It is usually served hot, although chilled or iced coffee is common. Coffee can be prepared and presented in a variety of ways (e.g., espresso, French press, caffè latte, or already-brewed canned coffee). Sugar, sugar substitutes, milk, and cream are often added to mask the bitter taste or enhance the flavor. There are also various coffee substitutes.

Though coffee is now a global commodity, it has a long history tied closely to food traditions around the Red Sea. The earliest credible evidence of coffee drinking as the modern beverage appears in modern-day Yemen in southern Arabia in the middle of the 15th century in Sufi shrines, where coffee seeds were first roasted and brewed in a manner similar to how it is now prepared for drinking.[3] The coffee beans were procured by the Yemenis from the Ethiopian Highlands via coastal Somali intermediaries, and cultivated in Yemen. By the 16th century, the drink had reached the rest of the Middle East and North Africa, later spreading to Europe.

The two most commonly grown coffee bean types are C. arabica and C. robusta.[4] Coffee plants are cultivated in over 70 countries, primarily in the equatorial regions of the Americas, Southeast Asia, the Indian subcontinent, and Africa. Green, unroasted coffee is traded as an agricultural commodity. The global coffee industry is massive and worth $495.50 billion as of 2023.[5] In the same year, Brazil was the leading grower of coffee beans, producing 35% of the world's total, followed by Vietnam and Colombia. While coffee sales reach billions of dollars annually worldwide, coffee farmers disproportionately live in poverty. Critics of the coffee industry have also pointed to its negative impact on the environment and the clearing of land for coffee-growing and water use.

Meanwhile, coffee had been introduced to Brazil in 1727, although its cultivation did not gather momentum until independence in 1822.[35] After this time, massive tracts of rainforest were cleared for coffee plantations, first in the vicinity of Rio de Janeiro and later São Paulo.[36] Brazil went from having essentially no coffee exports in 1800 to being a significant regional producer in 1830, to being the largest producer in the world by 1852. In 1910–1920, Brazil exported around 70% of the world's coffee, Colombia, Guatemala, and Venezuela exported 15%, and Old World production accounted for less than 5% of world exports.[37]

Many countries in Central America took up cultivation in the latter half of the 19th century, and almost all were involved in the large-scale displacement and exploitation of the indigenous people. Harsh conditions led to many uprisings, coups, and bloody suppression of peasants.[38] The notable exception was Costa Rica, where lack of ready labor prevented the formation of large farms. Smaller farms and more egalitarian conditions ameliorated unrest over the 19th and 20th centuries.[39]

Rapid growth in coffee production in South America during the second half of the 19th century was matched by an increase in consumption in developed countries, though nowhere has this growth been as pronounced as in the United States, where a high rate of population growth was compounded by doubling of per capita consumption between 1860 and 1920. Though the United States was not the heaviest coffee-drinking nation at the time (Belgium, the Netherlands and Nordic countries all had comparable or higher levels of per capita consumption), due to its sheer size, it was already the largest consumer of coffee in the world by 1860, and, by 1920, around half of all coffee produced worldwide was consumed in the US.[37]

Coffee has become a vital cash crop for many developing countries. Over one hundred million people in developing countries have become dependent on coffee as their primary source of income. It has become the primary export and economic backbone for African countries like Uganda, Burundi, Rwanda, and Ethiopia,[40] as well as many Central American countries.
"""

len(context)
predict_answer(context, "What is Coffee")
predict_answer(context, "What are the most common coffee beans?")
predict_answer(context, "How can I make ice coffee?")

predict_answer(context, "How many people are dependent on coffee for their income?")
predict_answer(context[4000:], "How many people are dependent on coffee for their income?")

def chunk_sentences(sentences, step, overlap):
    chunks = []
    for i in range(0, len(sentences), step):
        chunk = sentences[i : i + step + overlap]
        chunks.append(chunk)
    else:
        return chunks

test_sentences = [
    "Setence 1.",
    "Setence 2.",
    "Setence 3.",
    "Setence 4.",
    "Setence 5.",
    "Setence 6.",
    "Setence 7.",
    "Setence 8.",
    "Setence 9.",
    "Setence 10.",
]

test_result = chunk_sentences(test_sentences, 2, 1)

questions = [
    "What is Coffee",
    "What are the most common coffee beans?",
    "How many people are dependent on coffee for their income?"
]
sentences = context.split("\n")
chunk_sentences_result = chunk_sentences(sentences, 2, 1)

answers = {}

for chunk in chunk_sentences_result:
    context = "\n".join(chunk)
    for question in questions:
        # print(question)
        output, score = predict_answer(context, question)

        if output:
            if question not in answers:
                answers[question] = (output, score)
            else:
                if score > answers[question][1]:
                    answers[question] = (output, score)
