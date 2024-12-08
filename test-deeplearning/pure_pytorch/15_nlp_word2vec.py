import sys
sys.path.append(".")
import matplotlib.pyplot as plt
# from gensim.models import Word2Vec
import gensim.downloader as api
import tqdm
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.decomposition import PCA
from commons.timer import Timer

def train_my_word2vec_model():
    sentences = [
        ["I", "love", "NLP"],
        ["I", "will", "learn", "NLP", "in", "2month"],
        ["nlp", "is", "furture"],
        ["nlp", "saves", "time", "and ", "solves","lot", "of","industry", "problems"],
        ["nlp", "uses", "machine", "learning"],
    ]

    skipgram = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=1)
    # skipgram = Word2Vec(vector_size=50, window=3, min_count=1, sg=1)
    # skipgram.build_vocab(sentences)
    # skipgram.train(sentences, total_examples=skipgram.corpus_count, epochs=skipgram.epochs)
    print(skipgram)
    print(skipgram.wv.get_vector("nlp"))
    # print(skipgram.wv.get_vector("deep"))

    # 获取所有词向量
    X = skipgram.wv.vectors

    # 使用PCA将词向量降维到2维
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)

    # 绘制降维后的点
    plt.scatter(result[:, 0], result[:, 1])

    # 获取词汇表中的所有单词
    words = list(skipgram.wv.index_to_key)

    # 在每个点旁边添加注释
    for (i, word) in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.show()
    pass

def test_pretrain_word2vec_model():
    # model url: https://drive.usercontent.google.com/download?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM&export=download&authuser=0&confirm=t&uuid=dc707f2e-2a5a-4041-9bca-c642dc5ebc98&at=AENtkXaCJ5dzt6gFzUg8q4R1iMbC%3A1732371889201
    
    """
    Word2Vec 是一个模型训练类，它负责从文本数据中学习词向量。
    KeyedVectors 是一个词向量管理类，它负责存储、检索和操作词向量，可以独立于 Word2Vec 使用。
    在实际应用中，你可能会先使用 Word2Vec 训练一个模型，然后将生成的词向量保存到 KeyedVectors 中，以便在不同的应用中快速检索和使用这些向量。
    """
    model: KeyedVectors = KeyedVectors.load_word2vec_format("data/nlp/word2vec-model/GoogleNews-vectors-negative300.bin", binary=True)
    print(model.similarity("this", "is"))
    print(model.similarity("post", "book"))
    print(model.most_similar(positive=["woman", "king"], negative="man"))


def test_pretrain_word2vec_model2():
    with Timer():
        wv: KeyedVectors = api.load("word2vec-google-news-300")
    print(type(wv))
    print(wv["king"].shape)
    print(wv.most_similar("happy"))

if __name__ == "__main__":
    test_pretrain_word2vec_model2()