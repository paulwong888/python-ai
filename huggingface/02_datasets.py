from datasets import load_dataset_builder
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt

dataset_builder = load_dataset_builder("poem_sentiment", trust_remote_code=True) #获取数据的meta资料

print("dataset_builder.info.description =",dataset_builder.info.description)
print("dataset_builder.info.features =", dataset_builder.info.features)

poem_sentiment = load_dataset("poem_sentiment", trust_remote_code=True) #真正获取数据
print(poem_sentiment) #打印数据

train_ds = poem_sentiment["train"]
valid_ds = poem_sentiment["validation"]
test_ds = poem_sentiment["test"]

print(type(train_ds))

poem_sentiment.set_format(type="pandas") #设置下载数据的格式
df = poem_sentiment["train"][:] #取出key为train的数据

print(df.head(10))

def label_int2str(row):
	return poem_sentiment["train"].features["label"].int2str(row)

df["label_name"] = df["label"].apply(label_int2str) #增加一列
print(df.head(10))

df["label_name"].value_counts().plot.barh()
plt.title("Poem Classes")
plt.show()