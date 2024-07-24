from datasets import load_dataset

url = "https://mystic.the-eye.eu/public/AI/pile_preliminary_components/NIH_ExPORTER_awarded_grant_text.jsonl.zst"
nih_dataset_streamed = load_dataset(
    "json", data_files=url, split="train", streaming=True
)

shuffled_dataset = nih_dataset_streamed.shuffle(buffer_size=10_000, seed=5566)
print(next(iter(shuffled_dataset)))

nih_dataset_streamed = nih_dataset_streamed.filter(lambda x: x["meta"]["APPLICATION_ID"] >= 1000)
list(nih_dataset_streamed.take(3))

skipped_dataset = nih_dataset_streamed.skip(1000)