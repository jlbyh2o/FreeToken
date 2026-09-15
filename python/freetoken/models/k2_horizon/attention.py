from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.layers import BaseOP, LinearColParallelMerged, LinearOProj
from freetoken.layers.rotary import get_rope
from freetoken.utils import nvtx_annotate

from .mova import K2HorizonValueExperts
from .routing import router_logits, sigmoid_topk_route

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# softplus with this beta is 1.0 at a zero pre-activation, so an untrained gate is neutral.
_SOFTPLUS_BETA = math.log(2.0)


def _rope(config: ModelConfig):
    return get_rope(
        head_dim=config.head_dim,
        rotary_dim=config.rotary_config.rotary_dim,
        max_position=config.rotary_config.max_position,
        base=config.rotary_config.base,
        rope_scaling=(
            tuple(config.rotary_config.scaling.items()) if config.rotary_config.scaling else None
        ),
    )


class _K2HorizonAttentionBase(BaseOP):
    """Q/K/O plus K2-Horizon's post-attention output gate.

    The projections stay unmerged: the sparse layers have no ``v_proj`` to fuse with and
    the dense ones are three layers out of forty-eight, so a merged QKV would buy almost
    nothing and cost the loader a per-layer split rule.
    """

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        head_dim = config.head_dim
        self.layer_id = layer_id
        tp_size = get_tp_info().size
        assert tp_size == 1, "k2_horizon does not support tensor parallelism yet"
        self.num_qo_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = head_dim
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim

        self.q_proj = LinearColParallelMerged(
            config.hidden_size, [self.qo_attn_dim], has_bias=config.has_attn_bias,
            quant_config=config.quant, prefix=f"{prefix}.q_proj",
        )
        self.k_proj = LinearColParallelMerged(
            config.hidden_size, [self.kv_attn_dim], has_bias=config.has_attn_bias,
            quant_config=config.quant, prefix=f"{prefix}.k_proj",
        )
        self.gate_func = config.k2_args.attention_gate_func
        if self.gate_func is not None:
            assert self.gate_func in ("silu", "softplus"), (
                f"unsupported attention_gate_func {self.gate_func!r}"
            )
            self.gate_proj = LinearColParallelMerged(
                config.hidden_size, [self.qo_attn_dim], has_bias=False,
                quant_config=config.quant, prefix=f"{prefix}.gate_proj",
            )
        self.rotary = _rope(config)
        self.o_proj = LinearOProj(
            self.qo_attn_dim, config.hidden_size, has_bias=config.has_attn_bias,
            quant_config=config.quant, prefix=f"{prefix}.o_proj",
        )

    def _value_states(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _attend(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        q = self.q_proj.forward(x)
        k = self.k_proj.forward(x)
        q, k = self.rotary.forward(ctx.batch.get_attn_positions(), q, k)
        o = ctx.attn_backend.forward(
            q.view(-1, self.num_qo_heads, self.head_dim), k, v, self.layer_id, ctx.batch
        )
        o = o.view(-1, self.qo_attn_dim)
        if self.gate_func is not None:
            gate = self.gate_proj.forward(x)
            o = o * (
                F.silu(gate) if self.gate_func == "silu" else F.softplus(gate, beta=_SOFTPLUS_BETA)
            )
        return self.o_proj.forward(o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._attend(x, self._value_states(x))


class K2HorizonAttention(_K2HorizonAttentionBase):
    """The leading dense layers: an ordinary value projection."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        super().__init__(config, layer_id, prefix=prefix)
        self.v_proj = LinearColParallelMerged(
            config.hidden_size, [self.kv_attn_dim], has_bias=config.has_attn_bias,
            quant_config=config.quant, prefix=f"{prefix}.v_proj",
        )

    def _value_states(self, x: torch.Tensor) -> torch.Tensor:
        return self.v_proj.forward(x)


class K2HorizonMoVAAttention(_K2HorizonAttentionBase):
    """MoVA: the value projection is a router over value-experts.

    Only the values of the *incoming* tokens are routed. What lands in the KV cache is an
    ordinary ``[num_kv_heads * head_dim]`` row, so the cache layout, its budget and every
    attention backend are untouched by MoVA.
    """

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        super().__init__(config, layer_id, prefix=prefix)
        args = config.k2_args
        self.top_k = args.num_value_experts_per_tok
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = True  # calc_router_weights always renormalizes above top-1
        self.v_router = LinearColParallelMerged(
            config.hidden_size, [args.num_value_experts],
            has_bias=config.has_router_bias, quant_config=None, prefix=f"{prefix}.v_router",
        )
        self.v_experts = K2HorizonValueExperts(
            config, self.kv_attn_dim, prefix=f"{prefix}.v_experts"
        )

    @nvtx_annotate("MoVA")
    def _value_states(self, x: torch.Tensor) -> torch.Tensor:
        routing = sigmoid_topk_route(
            router_logits(x, self.v_router.weight),
            self.v_router.bias,
            top_k=self.top_k,
            renormalize=self.norm_topk_prob,
            scaling_factor=self.routed_scaling_factor,
        )
        return self.v_experts.forward(
            x, routing, is_prefill=get_global_ctx().batch.is_prefill
        )


__all__ = ["K2HorizonAttention", "K2HorizonMoVAAttention"]
