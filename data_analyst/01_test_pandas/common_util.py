import numpy as np
import pandas as pd

def buildDataFrameData() -> pd.DataFrame:
    return pd.DataFrame(
    data = np.random.randint(0,150, size=[10,3]),
    index = list("ABCDEFGHIJ"),
    columns = ["Python", "Tensorflow", "Keras"]
) 