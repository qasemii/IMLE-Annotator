import torch
import math

from torch import nn, Tensor
from typing import Optional, Tuple, Callable, Any

from transformers import (
    AutoModelForTokenClassification)


# Weight initializer
def init(layer: nn.Module):
    if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.zeros_(layer.bias)
    else:
        assert f'Do not know how to deal with {type(layer)}'


# Differentiable select-k model
class DifferentiableSelectKModel(torch.nn.Module):
    def __init__(self,
                 diff_fun: Callable[[Tensor], Tensor],
                 fun: Callable[[Tensor], Tensor]):
        super().__init__()
        self.diff_fun = diff_fun
        self.fun = fun

    def forward(self, logits: Tensor) -> Tensor:
        return self.diff_fun(logits) if self.training else self.fun(logits)


# prediction model
class Predictor(torch.nn.Module):
    def __init__(self,
                 embedding_matrix,
                 embedding_dims,
                 hidden_dims: int,
                 select_k: int):
        super().__init__()
        if embedding_matrix is None:
            self.embedding_dims = embedding_dims
        else:
            self.embedding_dims = embedding_matrix.shape[1]

        self.hidden_dims = hidden_dims
        self.select_k = float(select_k)

        self.layer_1 = nn.Linear(in_features=self.embedding_dims, out_features=self.hidden_dims, bias=True)
        init(self.layer_1)

        self.layer_2 = nn.Linear(in_features=self.hidden_dims, out_features=1, bias=True)
        init(self.layer_2)

        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()

    def forward(self,
            token_masks,
            token_embeddings,
            embedding_weights: Tensor) -> Tensor:
        x_emb = self.embeddings(x)  # [B, T] -> [B, T, E]
        res = x_emb * mask  # [B, S, E]

        res = torch.sum(embedding_weights, dim=1) / self.select_k  # [B, E]
        res = self.layer_1(res)  # [B, H]
        res = self.activation(res)  # [B, 1]
        res = self.layer_2(res)
        res = self.output_activation(res)
        return res


# Main model (BERT + Select_K + Predictor)
class Model(torch.nn.Module):
    def __init__(self,
                 model_name_or_path,
                 config,
                 hidden_dims: int,
                 select_k: int,
                 embedding_matrix=None,
                 differentiable_select_k: Optional[Callable[[Tensor], Tensor]] = None):
        super().__init__()

        # BERT model
        self.bert_model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config)

        # Differentiable select_k
        self.differentiable_select_k = differentiable_select_k

        # FC Predictor
        self.predictor = Predictor(
            embedding_matrix=embedding_matrix,
            embedding_dims=768,
            hidden_dims=hidden_dims,
            select_k=select_k)

    def z(self, **kwargs) -> tuple:
        # prediction includes score for labels in ['O', 'B-Xxx', 'I-Xxx']
        bert_output = self.bert_model(**kwargs)
        predictions = bert_output.logits
        # [B, T]
        token_scores = torch.nn.Softmax(dim=2)(predictions)[:, :, 1]
        token_masks = self.differentiable_select_k(token_scores)
        # [B, T, 1]
        token_masks = token_masks.unsqueeze(dim=-1)

        # (for now) we consider BERT last layer embeddings as word embeddings
        seq_embeddings = bert_output.hidden_states[0]
        token_embeddings = seq_embeddings * token_masks
        return token_masks, token_embeddings

    def forward(self, embedding_matrix=None, **kwargs) -> Tensor:
        # [B, T]
        token_masks, token_embeddings = self.z(**kwargs)

        p = self.predictor(
            token_masks,
            token_embeddings,
            embedding_matrix
        )

        return p
