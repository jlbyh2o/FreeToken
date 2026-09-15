from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP
from freetoken.layers.quantization import QuantKind
from freetoken.moe.fused_nvfp4 import fused_single_proj_nvfp4

from .routing import TopK

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# Packed FP4 codes, two per byte, with one fp8 scale per 16 values along K.
_PACK = 2
_GROUP = 16


class K2HorizonValueExperts(BaseOP):
    """The routed value projections of one MoVA layer: ``sum_k w_k * silu(W_[e_k] x)``.

    Unlike the routed MLP experts these stay GPU-resident. In bf16 they would be 14 GiB
    across the model, which is what forces the reference release to keep them dense; as
    NVFP4 they are under 4 GiB, so there is nothing to stream and the expert id indexes
    the bank directly instead of an offload slot.

    The bank layout is the one the inline-dequant kernels already read -- ``[E, N, K//2]``
    codes, ``[E, N, K//16]`` fp8 block scales, ``[E, N]`` fp16 row scales -- so the
    checkpoint's per-tensor ``weight_global_scale`` is broadcast across rows and stored
    reciprocated by the loader, matching every other NVFP4 bank in the engine.
    """

    def __init__(self, config: ModelConfig, out_features: int, *, prefix: str = ""):
        args = config.k2_args
        hidden = config.hidden_size
        self.num_experts = args.num_value_experts
        self.top_k = args.num_value_experts_per_tok
        self.out_features = out_features
        self.prefix = prefix
        # Asked per module, not off the checkpoint-wide expert_quant: an export can
        # quantize the routed MLP experts and leave these dense (which is exactly what
        # every release before this one did).
        scheme = config.quant.scheme_for(prefix) if config.quant is not None else None
        self.quantized = scheme is not None and scheme.kind is QuantKind.NVFP4

        if self.quantized:
            self.weight_packed = torch.empty(
                (self.num_experts, out_features, hidden // _PACK), dtype=torch.uint8
            )
            self.weight_scale = torch.empty(
                (self.num_experts, out_features, hidden // _GROUP), dtype=torch.float8_e4m3fn
            )
            self.weight_global_scale = torch.empty(
                (self.num_experts, out_features), dtype=torch.float16
            )
        else:
            # Reference path: correctness checks and small configs, not the served model.
            self.weight = torch.empty((self.num_experts, out_features, hidden))

    def _dense_forward(self, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor):
        routes = topk_ids.reshape(-1).long()
        w = self.weight.index_select(0, routes)
        y = torch.bmm(w, x.repeat_interleave(self.top_k, dim=0).unsqueeze(-1)).squeeze(-1)
        y = F.silu(y) * topk_weights.reshape(-1, 1).to(y.dtype)
        return y.view(x.shape[0], self.top_k, self.out_features).sum(dim=1)

    def forward(self, x: torch.Tensor, routing: TopK, *, is_prefill: bool) -> torch.Tensor:
        topk_weights, topk_ids = routing
        if not self.quantized:
            return self._dense_forward(x, topk_weights, topk_ids)
        return fused_single_proj_nvfp4(
            x,
            self.weight_packed,
            self.weight_scale,
            self.weight_global_scale,
            topk_weights,
            topk_ids,
            self.num_experts,
            is_prefill=is_prefill,
        )


__all__ = ["K2HorizonValueExperts"]
