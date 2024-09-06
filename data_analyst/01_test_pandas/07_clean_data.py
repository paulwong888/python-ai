import pandas as pd

df = pd.DataFrame(
    data = {
        "color": ["red", "blue", "red", "green", "blue", None, "red"],
        "price": [10,20,30,40,50,60,70]
    }
)

df.duplicated()
df.drop_duplicates()
print(df)

df.isnull()
df.dropna(how="any")
df.fillna(value=111)