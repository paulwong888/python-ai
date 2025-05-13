from common_util import buildDataFrameData

df = buildDataFrameData()

print(df)
print(df.loc[["A", "B", "C", "D", "F"]])
print(df.loc["A":"E", ["Python", "Tensorflow"]])
print(df.loc[:, ["Keras", "Tensorflow"]])
print(df.loc["E"::2, "Python": "Tensorflow"])
print(df.loc["A", "Python"])

print(df.iloc[4])
print(df.iloc[2:8, 0:2])
print(df.iloc[[1,3,5], [0,2,1]])
print(df.iloc[1:3, :])
print(df.iloc[:, 0:2])
print(df.iloc[0,2])
