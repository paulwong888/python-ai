import pandas as pd
import numpy as np

l = [0,1,7,9, np.NaN, None, 1024, 512]

s1 = pd.Series(data=1)
s2 = pd.Series(data=1, index=list("abcdefhi"), dtype="int32")
s3 = pd.Series(data={"a":99, "b":1})

print(l)
print(s1)
print(s2)
print(s3)

df1 = pd.DataFrame(
    data=np.random.randint(0,151,size=(150,3)),
    index= None,
    columns=["Python", "Math", "En"]
)

print(df1.head())

print(df1.shape)
print(df1.dtypes)
print(df1.index)
print(df1.columns)
print(df1.describe)
print(df1.info())