import nltk
import spacy
import en_core_web_sm
import torch
import time
import numpy as np
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from torch import nn
from sklearn.model_selection import train_test_split

# import torchtext.nn

# nltk.download('punkt_tab')
# data.

# tokenizer = lambda words : words.split()
# print(tokenizer("this is a test for tokenizer"))
# print(word_tokenize("this is a test for tokenizer")) 
# print()

class SentiClassifyRnn(nn.Module):
    def __init__(self, embedding_weights, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_weights)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, text_length):
        embedded = self.embedding(X)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(X, text_length)
        (packed_output, hidden) = self.rnn(packed_embedded)
        (output, output_lengths) = nn.utils.rnn.pad_packed_sequence(packed_output)
        return self.linear(hidden.squeeze(0))

class DatasetBuilder():

    def __init__(self, review_file_path, label_file_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 100
        self.review_file_path = review_file_path
        self.label_file_path = label_file_path

    def load_data(self):
        # with open("data/nlp/sentiment-classification/reviews.txt", "r") as review_file:
        with open(self.review_file_path, "r") as review_file:
            # lambda arguments: expression 没有函数名的函数体
            # map(function, iterable, ...) 对iterable中的元素调用function进行转换
            reviews = list(map(lambda x:x[:-1], review_file.readlines()))
        
        # with open("data/nlp/sentiment-classification/labels.txt", "r") as label_file:
        with open(self.label_file_path, "r") as label_file:
            # print(label_file.readlines()[0:2])
            labels = list(map(lambda y:y[:-1], label_file.readlines()))
        
        return reviews, labels

    def make_token(self, reviews):
        tokenizer = RegexpTokenizer(r"\w+\'?\w+|\w+")
        return tokenizer.tokenize(str(reviews))

    def remove_stopwords(self, reviews):
        exception_stop_words = {
            'again',
            'against',
            'ain',
            'almost',
            'among',
            'amongst',
            'amount',
            'anyhow',
            'anyway',
            'aren',
            "aren't",
            'below',
            'bottom',
            'but',
            'cannot',
            'couldn',
            "couldn't",
            'didn',
            "didn't",
            'doesn',
            "doesn't",
            'don',
            "don't",
            'done',
            'down',
            'except',
            'few',
            'hadn',
            "hadn't",
            'hasn',
            "hasn't",
            'haven',
            "haven't",
            'however',
            'isn',
            "isn't",
            'least',
            'mightn',
            "mightn't",
            'move',
            'much',
            'must',
            'mustn',
            "mustn't",
            'needn',
            "needn't",
            'neither',
            'never',
            'nevertheless',
            'no',
            'nobody',
            'none',
            'noone',
            'nor',
            'not',
            'nothing',
            'should',
            "should've",
            'shouldn',
            "shouldn't",
            'too',
            'top',
            'up',
            'wasn',
            "wasn't",
            'well',
            'weren',
            "weren't",
            'won',
            "won't",
            'wouldn',
            "wouldn't",
        }
        # nltk.download("stopwords")
        stop_words = stopwords.words("english")
        stop_words = set(stop_words).union(STOP_WORDS)
        stop_words = stop_words - exception_stop_words
        return [word for word in reviews if word not in stop_words ]

    # aren't -> are not, trained -> train
    def lammatization(self, reviews):
        # python -m spacy download en_core_web_sm -> 需下载并安装模型
        # nlp = spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])
        nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        # nlp = en_core_web_sm.load()
        # nlp = spacy.load("en_core_web_sm")
        lamma_result = []
        for words in reviews:
            # print(f"before nlp -> {words}")
            doc = nlp(words)
            # print(f"after nlp -> {doc}")
            for token in doc:
                # print(f"token in doc -> {token}")
                # print(f"token.lemma_ -> {token.lemma_}")
                lamma_result.append(token.lemma_)
        return lamma_result

    def pipeline(self, reviews):
        with Timer():
            reviews = self.make_token(reviews)
            reviews = self.remove_stopwords(reviews)
            reviews = self.lammatization(reviews)
        return reviews
    
    def pipeline_all(self, reviews):
        return list(map(lambda x : self.pipeline(x), reviews))

    def word2vec(self, reviews):
        model = Word2Vec(reviews, vector_size=100, min_count=3, window=5, workers=6)
        word_vectors = model.wv
        del model
        return word_vectors

    def word2idx(self, embedding_model: KeyedVectors, review):
        index_review = []
        for word in review:
            try:
                index_review.append(embedding_model.get_index(word))
            except:
                pass
        return torch.tensor(index_review)

    def word2idx_all(self, embedding_model: KeyedVectors, reviews):
        return list(map(lambda review : self.word2idx(embedding_model, review), reviews))

    def iterator_func(self, X, y):
        size = len(X)
        permutation = np.random.permutation(size)
        iterator = []
        for i in range(0,size, self.batch_size):
            indices = permutation[i:i + self.batch_size]
            batch = {}
            batch["text"] = [X[i] for i in indices]
            batch["label"] = [y[i] for i in indices]
            
            batch["text"],batch["label"] = zip(*sorted(zip(batch["text"],batch["label"]),key=lambda x: len(x[0]),reverse=True))
            batch["length"] = [len(review) for review in batch["text"]]
            batch["length"] = torch.IntTensor(batch["length"])
            batch["text"] = torch.nn.utils.rnn.pad_sequence(batch["text"],batch_first=True).t()
            batch["label"] = torch.Tensor(batch["label"])
            
            batch["label"]  = batch["label"].to(self.device)
            batch["length"] = batch["length"].to(self.device) 
            batch["text"]   = batch["text"].to(self.device) 
            
            iterator.append(batch)
            
        return iterator    
    
    def build_data_set(self, item_size=None):
        (reviews, labels) = self.load_data()

        if item_size != None:
            reviews = reviews[0:item_size]
            labels  = labels[0:item_size]

        reviews = self.pipeline_all(reviews)
        labels = [0 if label == 'negative' else 1 for label in labels ]
        embedding_model = self.word2vec(reviews)

        reviews = self.word2idx_all(embedding_model, reviews)

        (X_train, X_test, y_train, y_test) = train_test_split(reviews, labels, test_size=0.2)
        (X_train, X_val, y_train, y_val) = train_test_split(X_train, y_train, test_size=0.2)

        print(f"X_train length: {len(X_train)}, X_val length: {len(X_val)}, X_test length: {len(X_test)}")
        print(f"y_train length: {len(y_train)}, y_val length: {len(y_val)}, y_test length: {len(y_test)}")
        train_iterator = self.iterator_func(X_train,y_train)
        valid_iterator = self.iterator_func(X_val,y_val)
        test_iterator  = self.iterator_func(X_test,y_test)

        return (train_iterator, valid_iterator, train_iterator)

class Timer():
    def __enter__(self):
        self.start = time.time()
        return self.start
    
    def __exit__(self, *args):
        self.end = time.time()
        self.innterval = self.end - self.start
        print(f"执行时间： {self.innterval:.2f} 秒")

def test1():
    dataset_builder = get_dataset_builder()
    (reviews, labels) = dataset_builder.load_data()
    print(len(reviews), "\n", reviews[0])
    print()

    # print(make_token(reviews)[0:5])
    # print()

    reviews = reviews[:100]
    with Timer() as timer:
        reviews = list(map(lambda x:dataset_builder.pipeline(x), reviews))
    print(reviews[:2])
    # nltk.download("stopwords")

    with Timer() as timer:
        word_2_vec = dataset_builder.word2vec(reviews)
        print(word_2_vec.similar_by_word("good", 5))
    
    test_word2vec(word_2_vec)

def test_lammatization():
    dataset_builder = get_dataset_builder()
    print(dataset_builder.lammatization(["this", "isn't", "good"])) # -> ['this', 'be', 'not', 'good']
    print(dataset_builder.lammatization(["are", "is", "be"])) # -> ['be', 'be', 'be']
    print(dataset_builder.lammatization(["walk","walking","walked"])) # -> ['walk', 'walk', 'walk']

def test_remove_stopwords():
    dataset_builder = get_dataset_builder()
    print(dataset_builder.remove_stopwords(["this", "is", "a", "good", "movie"])) # -> ['good', 'movie']
    print(dataset_builder.remove_stopwords(["this", "is","not", "a", "good", "movie"])) # -> ['not', 'good', 'movie']

def test_word2vec():
    dataset_builder = get_dataset_builder()
    (reviews, labels) = dataset_builder.load_data()

    reviews = reviews[0:100]
    labels  = labels[0:100]
    # print(reviews)

    with Timer():
        reviews = dataset_builder.pipeline_all(reviews)
    word_vectors = dataset_builder.word2vec(reviews)

    print(len(word_vectors.key_to_index))
    print(word_vectors.key_to_index)
    print(len(word_vectors.index_to_key))
    print(word_vectors.index_to_key)
    # print(word_vectors.vocab)
    if "good" in word_vectors.index_to_key:
        print(word_vectors.similar_by_word(word="good", topn=5)) # -> ['good', 'movie']
        if "bad" in word_vectors.index_to_key:
            print(word_vectors.similarity("good","bad")) # -> ['not', 'good', 'movie']
    if "bad" in word_vectors.index_to_key:
        print(word_vectors.similar_by_word(word="bad", topn=5)) # -> ['not', 'good', 'movie']

def test_data_set():
    dataset_builder = get_dataset_builder()
    dataset_builder.build_data_set(10)

def get_dataset_builder():
    review_file_path = "data/nlp/sentiment-classification/reviews.txt"
    label_file_path  = "data/nlp/sentiment-classification/labels.txt"
    return DatasetBuilder(review_file_path=review_file_path, label_file_path=label_file_path)

if __name__ == "__main__":
    # test_remove_stopwords()
    # test_lammatization()
    # test_word2vec()
    
    test_data_set()

    # test1()