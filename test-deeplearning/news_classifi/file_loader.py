import pandas as pd
from sklearn.model_selection import train_test_split

class FileLoader():

    def __init__(self, file_path):
        super().__init__()
        self.data_frame = pd.read_csv(file_path, sep="\t", header=None, names=["text", "label"])
        self.labels = set(label for label in self.data_frame["label"])
        texts = self.data_frame["text"]
        print(self.data_frame.size)
        row = self.data_frame.iloc[0]
        print(row[0], row[1])
        print(len(self.data_frame))
        print(len(self.labels))
        print()

    def build_dataset(self):
        X_train, y_train = train_test_split(self.data_frame, test_size=0.2, shuffle=True)
        print(type(X_train), len(X_train), type(y_train), len(y_train))
        return X_train, y_train