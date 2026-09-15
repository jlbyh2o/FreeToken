from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as F
from freetoken.layers import BaseOP, LinearReplicated, make_moe_layer
from freetoken.utils import nvtx_annotate

from .routing import router_logits, sigmoid_topk_route

if TYPE_CHECKING:
    import torch

    from freetoken.layers.quantization import QuantConfig
    from freetoken.models.config import ModelConfig


class K2HorizonMLP(BaseOP):
    """SwiGLU MLP: the leading dense layers (``intermediate_size``) and every MoE layer's
    always-on shared expert (``moe_intermediate_size``). Unmerged gate/up, so the
    projections load straight off the checkpoint names."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        self.gate_proj = LinearReplicated(hidden_size, intermediate_size, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.gate_proj")
        self.up_proj = LinearReplicated(hidden_size, intermediate_size, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.up_proj")
        self.down_proj = LinearReplicated(intermediate_size, hidden_size, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.down_proj")

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj.forward(x)
        up = self.up_proj.forward(x)
        del x
        return self.down_proj.forward(F.silu(gate) * up)


class K2HorizonSparseBlock(BaseOP):
    """Top-k routed experts plus one always-on shared expert.

    Same router shape as the MoVA one in attention (sigmoid, selection-only gate bias,
    renormalize, scale), so both go through :func:`sigmoid_topk_route`. The routed experts
    are the usual offload-capable layer; only MoVA's stay resident.
    """

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

        self.gate = LinearReplicated(
            config.hidden_size, config.num_experts, has_bias=config.has_router_bias
        )
        # The offload cache indexes experts by MoE-layer id, matching how the loader packs
        # the banks; the leading dense layers own none.
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id - config.first_k_dense_replace,
            renormalize=config.norm_topk_prob,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
        )
        self.shared_experts = K2HorizonMLP(
            config.hidden_size,
            config.moe_intermediate_size * max(1, config.n_shared_experts),
            quant_config=config.quant,
            prefix=f"{prefix}.shared_experts",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        topk_weights, topk_ids = sigmoid_topk_route(
            router_logits(hidden_states, self.gate.weight),
            self.gate.bias,
            top_k=self.top_k,
            renormalize=self.norm_topk_prob,
            scaling_factor=self.routed_scaling_factor,
        )
        # The routed kernels may overwrite hidden_states, so the shared expert reads first.
        shared = self.shared_experts.forward(hidden_states)
        out = self.experts.routed_forward(hidden_states, topk_weights, topk_ids)
        return (out + shared).view(num_tokens, hidden_dim)


__all__ = ["K2HorizonMLP", "K2HorizonSparseBlock"]
