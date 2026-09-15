from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

TopK = Tuple[torch.Tensor, torch.Tensor]


def sigmoid_topk_route(
    logits: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    top_k: int,
    renormalize: bool,
    scaling_factor: float,
) -> TopK:
    """DeepSeek-style routing, shared by K2-Horizon's MoE block and its MoVA attention.

    The bias steers *selection only*: the weights returned are gathered from the unbiased
    sigmoid scores, then renormalized and scaled. HF computes the logits in fp32 and both
    routers here are tiny, so we match it exactly rather than route in bf16.
    """
    scores = logits.float().sigmoid()
    scores_for_choice = scores if bias is None else scores + bias.float()
    topk_ids = torch.topk(scores_for_choice, top_k, dim=-1).indices
    topk_weights = scores.gather(-1, topk_ids)
    if renormalize and top_k > 1:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * scaling_factor
    return topk_weights.contiguous(), topk_ids.to(torch.int32).contiguous()


def router_logits(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Router scores without the bias term: both K2-Horizon routers store a bias that is
    a selection correction, not part of the projection."""
    return F.linear(x.float(), weight.float())


__all__ = ["TopK", "router_logits", "sigmoid_topk_route"]
