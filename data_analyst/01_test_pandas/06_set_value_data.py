import pandas as pd
import numpy as np
from common_util import buildDataFrameData

df = buildDataFrameData()

s = pd.Series(
    data = np.random.randint(0,150, size=9),
    index = list("BCDEFG"),
    name = "PyTorch"
)

df["PyTorch"] = s
print(df)

df.loc["A", "Python"] = 256
print(df)

df.iloc[3, 2] = 512
print(df)

df.loc[:, "Python"] = np.array([128] * 10)
print(df)

df[df >= 128] = -df