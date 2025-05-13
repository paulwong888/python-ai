from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import tqdm
import torch

# dataset = load_dataset("tau/multi_news", split="test")
dataset = load_dataset("alexfabbri/multi_news", split="test", trust_remote_code=True)
# dataset = load_dataset(path="text", data_dir="../../data/multi-news/", split="test")
df = dataset.to_pandas().sample(2000, random_state=42)

model = SentenceTransformer("all-MiniLM-L6-V2")

passage_embeddings = list(model.encode(df["summary"].to_list()))

passage_embeddings[0].shape

query = "Find me some articles about technology and artificial intelligenbce"

def find_relevant_news(query:str):
    query_embedding = model.encode(query)

    similarities = util.cos_sim(query_embedding, passage_embeddings)

    top_indices = torch.topk(similarities.flatten(), k=3).indices

    top_relevant_passages = [df.iloc[x.item()]["summary"][:200] + "..." for x in top_indices]

    return top_relevant_passages

find_relevant_news("Natural disasters")
find_relevant_news("Law enforcement and police")
find_relevant_news("Politics, diplomacy and nationalism")