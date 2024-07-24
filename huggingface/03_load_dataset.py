from datasets import load_dataset

txt_dataset = load_dataset("text", data_files="my_dataset.txt")
json_dataset = load_dataset("json", data_files="my_dataset.jsonl") #注意這裡用的是 JSON Lines 的格式
pandas_dataset = load_dataset("pandas", data_files="my_dataset.pkl")

csv_dataset = load_dataset("csv", data_files="my_dataset.csv",sep=",",
                               names=["text", "label"]) #只载入需要的列的数据