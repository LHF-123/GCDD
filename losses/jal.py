"""JAL-CE loss components.

JAL-CE corresponds to the NCEandAMSE objective used by the official JAL
implementation. Targets are the observed noisy labels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NCELoss(nn.Module):
    def __init__(self, scale: float = 1.0, eps: float = 1.0e-8):
        super().__init__()
        self.scale = float(scale)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        target = target.long()

        log_prob = F.log_softmax(logits, dim=1)
        nll = -log_prob.gather(dim=1, index=target.view(-1, 1)).squeeze(1)
        normalizer = (-log_prob).sum(dim=1).clamp_min(self.eps)
        loss = nll / normalizer
        return self.scale * loss.mean()


class AMSELoss(nn.Module):
    def __init__(self, a: float = 30.0, scale: float = 1.0):
        super().__init__()
        self.a = float(a)
        self.scale = float(scale)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        target = target.long()

        prob = F.softmax(logits, dim=1)
        one_hot = F.one_hot(target, num_classes=logits.size(1)).to(device=logits.device, dtype=prob.dtype)
        target_vec = self.a * one_hot
        loss = (prob - target_vec).pow(2).mean()
        return self.scale * loss


class JALCELoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        a: float = 30.0,
        eps: float = 1.0e-8,
    ):
        super().__init__()
        self.nce = NCELoss(scale=alpha, eps=eps)
        self.amse = AMSELoss(a=a, scale=beta)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.nce(logits, target) + self.amse(logits, target)
