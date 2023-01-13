import torch
import math

from torch import nn, Tensor
from typing import Optional, Tuple, Callable


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
    def __init__(self, embedding_weights: Tensor,
                 hidden_dims: int,
                 select_k: int):
        super().__init__()
        self.nb_words, self.embedding_dim = embedding_weights.shape
        self.hidden_dims = hidden_dims
        self.select_k = float(select_k)

        self.embeddings = nn.Embedding.from_pretrained(embedding_weights, freeze=True)

        self.layer_1 = nn.Linear(in_features=self.embedding_dim, out_features=self.hidden_dims, bias=True)
        init(self.layer_1)

        self.layer_2 = nn.Linear(in_features=self.hidden_dims, out_features=1, bias=True)
        init(self.layer_2)

        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()

    def forward(self, x: Tensor,
                mask: Tensor) -> Tensor:
        x_emb = self.embeddings(x)  # [B, T] -> [B, T, E]
        res = x_emb * mask  # [B, S, E]
        res = torch.sum(res, dim=1) / self.select_k  # [B, E]
        res = self.layer_1(res)  # [B, H]
        res = self.activation(res)  # [B, 1]
        res = self.layer_2(res)
        res = self.output_activation(res)
        return res
