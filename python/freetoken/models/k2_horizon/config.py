from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from freetoken.models.config import ModelConfig, RotaryConfig, detect_expert_quant


@dataclass(frozen=True)
class K2HorizonArgs:
    """MoVA geometry and the two scalars the generic ModelConfig has no home for."""

    num_value_experts: int
    num_value_experts_per_tok: int
    # Post-attention output gate activation ("softplus"); None leaves attention ungated.
    attention_gate_func: str | None
    # Pre-sublayer norms normalize this many independent slices of the hidden state.
    layernorm_num_groups: int


def _rope_params(hf_config: Any) -> dict:
    params = getattr(hf_config, "rope_parameters", None)
    if isinstance(params, dict) and params:
        return dict(params)
    params = {"rope_theta": getattr(hf_config, "rope_theta", 10000.0)}
    scaling = getattr(hf_config, "rope_scaling", None)
    if isinstance(scaling, dict):
        params.update(scaling)
    return params


def _first_k_dense(hf_config: Any) -> int:
    """``mlp_only_layers`` as a dense prefix length.

    FreeToken indexes the expert banks (and the offload cache) by MoE-layer id, which
    only models a leading run of dense layers. K2-Horizon ships ``[0, 1, 2]``; anything
    that is not a prefix would silently misalign every expert, so refuse it here.
    """
    dense = sorted(int(i) for i in (getattr(hf_config, "mlp_only_layers", None) or ()))
    if dense and dense != list(range(len(dense))):
        raise NotImplementedError(
            f"k2_horizon: mlp_only_layers={dense} is not a leading prefix; "
            "FreeToken only models dense layers at the front of the stack"
        )
    step = int(getattr(hf_config, "decoder_sparse_step", 1))
    if step != 1:
        raise NotImplementedError(f"k2_horizon: decoder_sparse_step={step} is not supported")
    return len(dense)


def parse_config(hf_config: Any) -> ModelConfig:
    """Parse a HuggingFace ``K2HorizonConfig`` into FreeToken's :class:`ModelConfig`.

    K2-Horizon specifics handled here:
    - MoVA attention: the value projection of every sparse layer is a router over
      ``mova_num_experts`` value-experts (top-k, SiLU) instead of a plain ``v_proj``.
      The leading dense layers keep the plain projection.
    - A DeepSeek-style router (sigmoid scores, selection-only gate bias, renormalize,
      scale by ``router_scaling_factor``) plus one always-on shared expert.
    - Grouped pre-sublayer norms (``layernorm_num_groups``) and a softplus output gate
      on attention.
    """
    head_dim = (
        getattr(hf_config, "head_dim", None)
        or hf_config.hidden_size // hf_config.num_attention_heads
    )
    num_kv_heads = getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)

    rope = _rope_params(hf_config)
    rope_type = rope.get("rope_type", rope.get("type", "default"))
    rope_scaling = None if rope_type in (None, "default") else rope
    rotary_dim = int(getattr(hf_config, "rope_head_dim", None) or head_dim)

    score_func = str(getattr(hf_config, "router_score_func", "sigmoid"))
    if score_func != "sigmoid":
        raise NotImplementedError(f"k2_horizon: router_score_func={score_func!r} is not supported")

    return ModelConfig(
        num_layers=hf_config.num_hidden_layers,
        num_qo_heads=hf_config.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hf_config.hidden_size,
        vocab_size=hf_config.vocab_size,
        intermediate_size=hf_config.intermediate_size,
        hidden_act=hf_config.hidden_act,
        rms_norm_eps=hf_config.rms_norm_eps,
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        rotary_config=RotaryConfig(
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            max_position=hf_config.max_position_embeddings,
            base=rope.get("rope_theta", 10000.0),
            scaling=rope_scaling,
        ),
        num_experts=int(getattr(hf_config, "num_experts", 0)),
        num_experts_per_tok=int(getattr(hf_config, "num_experts_per_tok", 0)),
        moe_intermediate_size=int(getattr(hf_config, "moe_intermediate_size", 0)),
        norm_topk_prob=bool(getattr(hf_config, "norm_topk_prob", True)),
        model_type=getattr(hf_config, "model_type", "k2_horizon"),
        architectures=getattr(hf_config, "architectures", ["K2HorizonForCausalLM"]),
        moe_enabled=True,
        use_qk_norm=bool(getattr(hf_config, "query_key_norm", False)),
        expert_quant=detect_expert_quant(hf_config),
        first_k_dense_replace=_first_k_dense(hf_config),
        n_shared_experts=int(getattr(hf_config, "num_shared_experts", 0)),
        routed_scaling_factor=float(getattr(hf_config, "router_scaling_factor", 1.0) or 1.0),
        has_attn_bias=bool(getattr(hf_config, "attention_bias", False)),
        has_router_bias=bool(getattr(hf_config, "moe_gate_bias", False)),
        k2_args=K2HorizonArgs(
            num_value_experts=int(getattr(hf_config, "mova_num_experts", 0)),
            num_value_experts_per_tok=int(getattr(hf_config, "mova_num_experts_per_tok", 0)),
            attention_gate_func=getattr(hf_config, "attention_gate_func", None),
            layernorm_num_groups=int(getattr(hf_config, "layernorm_num_groups", 1)),
        ),
    )


__all__ = ["K2HorizonArgs", "parse_config"]
