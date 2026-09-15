"""k2_horizon (K2-Horizon-MoVA) config parsing, grouped norms and the MoVA
value-expert layer.

The MoVA reference is HF ``K2HorizonMoVAAttention``'s value path transcribed from
modeling_k2_horizon.py: sigmoid router scores, the gate bias applied to selection only,
the unbiased scores gathered / renormalized / scaled, then a weighted sum of
``silu(W_e x)`` over the selected experts.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.layers import GroupedRMSNormFused
from freetoken.models.k2_horizon.config import parse_config
from freetoken.models.k2_horizon.mova import K2HorizonValueExperts
from freetoken.models.k2_horizon.routing import router_logits, sigmoid_topk_route
from freetoken.models.register import get_model_spec
from freetoken.utils.hf import RawConfigShim

_HIDDEN, _KV_DIM = 256, 64
_NUM_V_EXPERTS, _TOP_K, _SCALING = 8, 2, 2.5


@pytest.fixture(autouse=True)
def _tp_info():
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _hf_config(**overrides) -> RawConfigShim:
    # Trimmed from IFM/K2-Horizon-MoVA-36B-A4B config.json.
    data = {
        "architectures": ["K2HorizonForCausalLM"],
        "model_type": "k2_horizon",
        "hidden_size": _HIDDEN,
        "intermediate_size": 512,
        "moe_intermediate_size": 128,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "rope_head_dim": 64,
        "vocab_size": 1024,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "max_position_embeddings": 524288,
        "rope_parameters": {"rope_theta": 1e7, "rope_type": "default"},
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "num_shared_experts": 1,
        "norm_topk_prob": True,
        "router_score_func": "sigmoid",
        "router_scaling_factor": _SCALING,
        "moe_gate_bias": True,
        "mlp_only_layers": [0, 1, 2],
        "decoder_sparse_step": 1,
        "mova_num_experts": _NUM_V_EXPERTS,
        "mova_num_experts_per_tok": _TOP_K,
        "attention_gate_func": "softplus",
        "layernorm_num_groups": 2,
        "query_key_norm": False,
        "attention_bias": False,
    }
    data.update(overrides)
    return RawConfigShim(data)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_registered():
    spec = get_model_spec("K2HorizonForCausalLM")
    assert spec.module == "freetoken.models.k2_horizon"
    assert ("v_experts", ("v_experts.0",)) in spec.packed_modules_mapping


def test_parse_config_mova_and_router():
    config = parse_config(_hf_config())
    assert config.k2_args.num_value_experts == _NUM_V_EXPERTS
    assert config.k2_args.num_value_experts_per_tok == _TOP_K
    assert config.k2_args.attention_gate_func == "softplus"
    assert config.k2_args.layernorm_num_groups == 2
    assert config.routed_scaling_factor == _SCALING
    assert config.has_router_bias is True
    assert config.n_shared_experts == 1
    assert config.moe_enabled is True


def test_mlp_only_layers_become_a_dense_prefix():
    config = parse_config(_hf_config())
    assert config.first_k_dense_replace == 3
    assert config.num_moe_layers == 3


def test_non_prefix_mlp_only_layers_rejected():
    # Expert banks are indexed by MoE-layer id, which only models a leading dense run;
    # a gap would silently misalign every expert, so parse_config must refuse it.
    with pytest.raises(NotImplementedError, match="leading prefix"):
        parse_config(_hf_config(mlp_only_layers=[0, 2, 5]))


def test_unsupported_router_rejected():
    with pytest.raises(NotImplementedError, match="router_score_func"):
        parse_config(_hf_config(router_score_func="softmax"))


# ---------------------------------------------------------------------------
# Grouped RMSNorm
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_groups", [1, 2, 4])
def test_grouped_rmsnorm_matches_reference(n_groups):
    torch.manual_seed(0)
    x = torch.randn(8, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(_HIDDEN, device="cuda", dtype=torch.bfloat16)

    norm = GroupedRMSNormFused(_HIDDEN, 1e-6, n_groups)
    norm.weight = weight
    got, residual = norm.forward(x.clone())

    ref = x.float().reshape(8, n_groups, -1)
    ref = ref * torch.rsqrt(ref.pow(2).mean(-1, keepdim=True) + 1e-6)
    ref = ref.reshape(8, _HIDDEN) * weight.float()
    torch.testing.assert_close(got.float(), ref, rtol=2e-2, atol=2e-2)
    # the un-normalized input becomes the residual, as RMSNormFused does
    torch.testing.assert_close(residual.float(), x.float())


def test_grouped_rmsnorm_absorbs_residual():
    torch.manual_seed(0)
    x = torch.randn(4, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(4, _HIDDEN, device="cuda", dtype=torch.bfloat16)
    norm = GroupedRMSNormFused(_HIDDEN, 1e-6, 2)
    norm.weight = torch.ones(_HIDDEN, device="cuda", dtype=torch.bfloat16)

    _, new_residual = norm.forward(x.clone(), residual.clone())
    torch.testing.assert_close(new_residual.float(), (x + residual).float())


def test_grouped_rmsnorm_rejects_indivisible_width():
    with pytest.raises(AssertionError, match="divisible"):
        GroupedRMSNormFused(10, 1e-6, 3)


# ---------------------------------------------------------------------------
# MoVA value experts
# ---------------------------------------------------------------------------
def _mova_reference(x, router_w, router_b, weights):
    scores = F.linear(x.float(), router_w.float()).sigmoid()
    selected = torch.topk(scores + router_b.float(), _TOP_K, dim=-1).indices
    routed = scores.gather(-1, selected)
    routed = routed / routed.sum(dim=-1, keepdim=True) * _SCALING
    out = torch.zeros(x.shape[0], _KV_DIM, device=x.device, dtype=torch.float32)
    for token in range(x.shape[0]):
        for k in range(_TOP_K):
            expert = weights[int(selected[token, k])].float()
            out[token] += F.silu(expert @ x[token].float()) * routed[token, k]
    return out


def _fixtures(seed: int = 0):
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    x = (torch.randn(8, _HIDDEN, device=dev) * 0.05).to(torch.bfloat16)
    router_w = (torch.randn(_NUM_V_EXPERTS, _HIDDEN, device=dev) * 0.05).to(torch.bfloat16)
    router_b = (torch.randn(_NUM_V_EXPERTS, device=dev) * 0.1).to(torch.bfloat16)
    weights = (torch.randn(_NUM_V_EXPERTS, _KV_DIM, _HIDDEN, device=dev) * 0.02).to(torch.bfloat16)
    return x, router_w, router_b, weights


def _layer(quantized: bool, weights):
    layer = K2HorizonValueExperts.__new__(K2HorizonValueExperts)
    layer.num_experts, layer.top_k = _NUM_V_EXPERTS, _TOP_K
    layer.out_features, layer.prefix, layer.quantized = _KV_DIM, "v_experts", quantized
    if not quantized:
        layer.weight = weights.clone()
    return layer


def test_routing_matches_reference_selection():
    x, router_w, router_b, _ = _fixtures()
    weights, ids = sigmoid_topk_route(
        router_logits(x, router_w), router_b,
        top_k=_TOP_K, renormalize=True, scaling_factor=_SCALING,
    )
    scores = F.linear(x.float(), router_w.float()).sigmoid()
    expected_ids = torch.topk(scores + router_b.float(), _TOP_K, dim=-1).indices
    # the bias steers selection; the weights come from the UNBIASED scores
    torch.testing.assert_close(ids.long(), expected_ids)
    expected = scores.gather(-1, expected_ids)
    expected = expected / expected.sum(dim=-1, keepdim=True) * _SCALING
    torch.testing.assert_close(weights, expected)


def test_mova_dense_matches_reference():
    x, router_w, router_b, weights = _fixtures()
    routing = sigmoid_topk_route(
        router_logits(x, router_w), router_b,
        top_k=_TOP_K, renormalize=True, scaling_factor=_SCALING,
    )
    got = _layer(False, weights).forward(x, routing, is_prefill=True).float()
    ref = _mova_reference(x, router_w, router_b, weights)
    assert ((got - ref).norm() / ref.norm()).item() < 0.02
