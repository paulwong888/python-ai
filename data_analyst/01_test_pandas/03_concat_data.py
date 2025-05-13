import pandas as pd
import numpy as np

from common_util import buildDataFrameData

df1 = buildDataFrameData()
df1.index = list("ABCDEFGHIJ")

df2 = buildDataFrameData()
df2.index = list("BCDEFGHIJM")

df3 = pd.DataFrame(
    data = np.random.randint(0,150, size=[10,2]),
    index = list("ABCDEFGHIJ"),
    columns = ["PyTorch", "Paddle"]
)

print(df1)
print(df2)
df = pd.concat([df1, df2], axis=0)
print(df)
df = pd.concat([df1, df2], axis=1)
print(df)

dff1 = pd.DataFrame(data={"name":["A", "B", "C", "C"], "weight":[70,55,75,65]})
dff2 = pd.DataFrame(data={"name":["A", "B", "C", "D"], "weight":[170,155,175,165]})
dff3 = pd.DataFrame(data={"name":["A", "B", "C", "D"], "weight":[270,255,275,265]})

dff = pd.merge(dff1, dff2, how="inner", on="name")
print(dff)