import pandas as pd
import numpy as np

df = pd.DataFrame(
    data=np.random.randint(0,50, size=[50,5]),
    columns=["IT", "化工", "生物", "教师", "士兵"]
)

print(df)

df.to_csv("./salary.csv", header=True, index=True)

pd.read_table("./salary.csv", header=[0], index_col=0)

df2 = pd.DataFrame(
    data=np.random.randint(0,50, size=[150,3]),
    columns=["Python", "Tensorflow", "Keras"]
)

df.to_excel(
    "./salary.xlsx", 
    sheet_name="salary", 
    header=True, 
    index=False
)