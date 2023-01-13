import torch

from torch import Tensor
from abc import ABC, abstractmethod

from typing import Optional


class BaseTargetDistribution(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def params(self,
               theta: Tensor,
               dy: Optional[Tensor],
               _is_minimization: bool = False) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def process(self,
                theta: Tensor,
                dy: Tensor,
                gradient: Tensor) -> Tensor:
        return gradient


class TargetDistribution(BaseTargetDistribution):
    r"""
    Creates a generator of target distributions parameterized by :attr:`alpha` and :attr:`beta`.

    Example::

        >>> import torch
        >>> target_distribution = TargetDistribution(alpha=1.0, beta=1.0)
        >>> target_distribution.params(theta=torch.tensor([1.0]), dy=torch.tensor([1.0]))
        tensor([2.])

    Args:
        alpha (float): weight of the initial distribution parameters theta
        beta (float): weight of the downstream gradient dy
        do_gradient_scaling (bool): whether to scale the gradient by 1/λ or not
    """
    def __init__(self,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 do_gradient_scaling: bool = False,
                 eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.do_gradient_scaling = do_gradient_scaling
        self.eps = eps

    def params(self,
               theta: Tensor,
               dy: Optional[Tensor],
               alpha: Optional[float] = None,
               beta: Optional[float] = None,
               _is_minimization: bool = False) -> Tensor:
        alpha_ = self.alpha if alpha is None else alpha
        beta_ = self.beta if beta is None else beta

        if _is_minimization is True:
            theta_prime = alpha_ * theta + beta_ * (dy if dy is not None else 0.0)
        else:
            theta_prime = alpha_ * theta - beta_ * (dy if dy is not None else 0.0)
        return theta_prime

    def process(self,
                theta: Tensor,
                dy: Tensor,
                gradient_3d: Tensor) -> Tensor:
        scaling_factor = max(self.beta, self.eps)
        res = (gradient_3d / scaling_factor) if self.do_gradient_scaling is True else gradient_3d
        return res


