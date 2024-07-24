from datasets import load_dataset
url = "https://mystic.the-eye.eu/public/AI/pile_preliminary_components/NIH_ExPORTER_awarded_grant_text.jsonl.zst"
nih_dataset = load_dataset("csv", data_files=url, split= "train")

nih_dataset