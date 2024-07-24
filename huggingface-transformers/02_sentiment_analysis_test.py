from transformers import pipeline

sentiment = pipeline('sentiment-analysis')

print(sentiment.model._version)

print(sentiment(["I like Olympic games as it’s very exciting."]))
print(sentiment(["I'm against to hold Olympic games in Tokyo in terms of preventing the covid19 to be spre"]))