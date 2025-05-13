from torch import nn
from transformers import BertModel

class BertClassifier(nn.Module):
    def __init__(self, model_path, class_num):
        super(BertClassifier, self).__init__()
        self.bert: BertModel = BertModel.from_pretrained(model_path)
        self.linear = nn.Linear(in_features=self.bert.config.hidden_size, out_features=class_num)

    def forward(self, input_ids, attension_mask):
        features = self.bert(input_ids, attension_mask)
        logits = self.linear(features.pooler_output)
        return logits