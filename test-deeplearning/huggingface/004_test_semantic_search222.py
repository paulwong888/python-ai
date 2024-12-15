import torch
import tqdm
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset

class SemanticSearchModel():
    def __init__(self):
        dataset_name = "alexfabbri/multi_news"
        model_name = "all-MiniLM-L6-V2"
        dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
        self.dataset_pandas = dataset.to_pandas().sample(2000, random_state=42)
        self.model = SentenceTransformer(model_name)
        self.passage_embiddings = list(self.model.encode(self.dataset_pandas["summary"].to_list()))

    """
    (query emb, summary list 2000个[emb,emb,emb,emb..])->
    2000个分数[score,score,score..]->
    3个最高分数的索引[index,index,index]->
    从summary list中查找对应的字符[sumarry,sumarry,sumarry]
    """
    def search_relevant_news(self, query:str):
        # 将一个句子转成一个embedding,384维
        query_embedding = self.model.encode(query)
        similarities = util.cos_sim(query_embedding, self.passage_embiddings)
        top_indices = torch.topk(similarities.flatten(), k=3).indices
        top_relevant_passages = [self.dataset_pandas.iloc[x.item()]["summary"][:200] + "..." for x in top_indices]
        return top_relevant_passages
    
if __name__ == "__main__":
    semantic_search_model = SemanticSearchModel()
    query = "Natural disasters"
    query_embedding = semantic_search_model.model.encode(query)
    similarities = util.cos_sim(query_embedding, semantic_search_model.passage_embiddings)
    top_indices = torch.topk(similarities.flatten(), k=3).indices
    top_relevant_passages = [semantic_search_model.dataset_pandas.iloc[x.item()]["summary"][:200] + "..." for x in top_indices]
    print_summary = lambda query : [print(x) for x in semantic_search_model.search_relevant_news(query)]
    print_summary("Natural disasters")
    print_summary("Law enforcement and police")
    print_summary("Politics, diplomacy and nationalism")
    
        