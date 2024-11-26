import sys
sys.path.append(".")
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from commons.my_dataset import get_beans

# one shot example
text = "I am learning NLP"
print(pd.get_dummies(text.split()))
print()

# frequency example
text = ["I love NLP and I will learn NLP in 2month"]
vectorizer = CountVectorizer()
vectorizer.fit(text)
vector = vectorizer.transform(text)
print(vectorizer.vocabulary_)
print(vector.toarray())
print()

#include word context
text = "I am learning NLP"
print(TextBlob(text).ngrams(1))
print(TextBlob(text).ngrams(2))
print()

#frequency + n-gram example
text = ["I love NLP and I will learn NLP in 2month"]
vectorizer = CountVectorizer(ngram_range=(2, 2))
vectorizer.fit(text)
vector = vectorizer.transform(text)
print(vectorizer.vocabulary_)
print(vector.toarray())
print()
