import torch
from torch import Tensor


def select_k(logits: Tensor, select_k: int) -> Tensor:
    scores, indices = torch.topk(logits, select_k, sorted=True)
    mask = torch.zeros_like(logits, device=logits.device).scatter_(-1, indices, 1.0)
    return mask


def mathias_select_k(logits: Tensor, select_k: int) -> Tensor:
    scores, indices = torch.topk(logits, select_k, sorted=True)
    thr_2d = scores[:, -1].view(-1, 1)
    return (logits >= thr_2d).float()