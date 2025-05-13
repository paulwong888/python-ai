from common_util import buildDataFrameData

df = buildDataFrameData()

cond1 = df.Python > 100
print(df[cond1])

cond2 = (df.Python > 50) & (df["Keras"] > 50)
print(df[cond2])

print(df[df.index.isin(["A", "C", "F"])])