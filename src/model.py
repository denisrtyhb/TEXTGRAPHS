import torch
from torch import nn
from transformers import AutoModel, PreTrainedModel


class BertSimpleClassifier(nn.Module):
    """Classifier head on top of a BERT-style encoder (see baselines/bert_baselines.ipynb)."""

    def __init__(self, bert_text_encoder: PreTrainedModel, dropout: float = 0.1):
        super().__init__()
        self.bert_text_encoder = bert_text_encoder
        self.dropout = nn.Dropout(p=dropout)
        bert_hidden_dim = bert_text_encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(bert_hidden_dim, bert_hidden_dim),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(bert_hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden_states = self.bert_text_encoder(
            inputs, attention_mask=attention_mask, return_dict=True
        )["last_hidden_state"]
        text_cls_embeddings = torch.stack([elem[0, :] for elem in last_hidden_states])
        return self.classifier(text_cls_embeddings)


def build_classifier(model_name: str, dropout: float = 0.2) -> BertSimpleClassifier:
    bert_model = AutoModel.from_pretrained(model_name)
    return BertSimpleClassifier(bert_model, dropout=dropout)
