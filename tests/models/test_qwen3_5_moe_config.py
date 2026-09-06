"""qwen3_5_moe quantization detection and the routed-expert bank source specs.

Runs off a trimmed copy of a real checkpoint config.json via RawConfigShim (the exact
object cached_load_hf_config falls back to), so the parse path under test is the one
production hits. The compressed-tensors cases come from an llm-compressor export of a
Qwen3.5-MoE checkpoint (Ornith-1.5-35B-A3B-NVFP4): ``quant_method`` is the bare string
"compressed-tensors" and the NVFP4 geometry lives in ``config_groups``, so the modelopt
``quant_algo`` branches never fire.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.models.qwen3_5_moe.config import parse_config
from freetoken.utils.hf import RawConfigShim

_NUM_LAYERS = 8


def _text_config(num_experts: int = 256) -> dict:
    # Trimmed from Ornith-1.5-35B-A3B-NVFP4 (Qwen3_5MoeForConditionalGeneration).
    return {
        "attention_bias": False,
        "attn_output_gate": True,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_5_moe_text",
        "moe_intermediate_size": 512,
        "num_attention_heads": 16,
        "num_experts": num_experts,
        "num_experts_per_tok": 8,
        "num_hidden_layers": _NUM_LAYERS,
        "num_key_value_heads": 2,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "shared_expert_intermediate_size": 512,
        "tie_word_embeddings": False,
        "vocab_size": 248320,
    }


def _hf_config(quantization_config: dict | None = None, num_experts: int = 256) -> RawConfigShim:
    data: dict = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "hidden_size": 2048,
        "text_config": _text_config(num_experts),
        "vision_config": {"model_type": "qwen3_5_moe_vision", "depth": 27},
        "image_token_id": 248056,
    }
    if quantization_config is not None:
        data["quantization_config"] = quantization_config
    return RawConfigShim(data)


_CT_NVFP4_QUANT = {
    # llm-compressor NVFP4: no quant_algo at all, and quant_method is the tooling name.
    "quant_method": "compressed-tensors",
    "format": "nvfp4-pack-quantized",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "format": "nvfp4-pack-quantized",
            "weights": {
                "num_bits": 4,
                "type": "float",
                "group_size": 16,
                "strategy": "tensor_group",
            },
        }
    },
}

_MODELOPT_NVFP4_QUANT = {"quant_method": "modelopt", "quant_algo": "NVFP4"}


def test_compressed_tensors_moe_resolves_nvfp4_experts():
    """The regression this file exists for: a compressed-tensors NVFP4 MoE export used to
    resolve expert_quant="none", which sent the routed experts to the bf16 bank provider and
    died with "Missing MoE expert source layers" after writing the whole dense pass."""
    cfg = parse_config(_hf_config(_CT_NVFP4_QUANT))
    assert cfg.expert_quant == "nvfp4"
    assert cfg.moe_enabled


def test_compressed_tensors_dense_keeps_no_expert_quant():
    """A dense compressed-tensors checkpoint (Qwen3.6-27B) has no routed experts to
    describe; its packed FP4 is carried by dense_quant instead."""
    cfg = parse_config(_hf_config(_CT_NVFP4_QUANT, num_experts=0))
    assert cfg.expert_quant == "none"
    assert not cfg.moe_enabled
    assert cfg.dense_quant == "nvfp4"


def test_compressed_tensors_moe_keeps_the_dense_quant_verdicts():
    """Flipping expert_quant must not disturb the dense-side wiring: attention and the
    shared-expert MLP stay native W4A16 and lm_head stays bf16."""
    cfg = parse_config(_hf_config(_CT_NVFP4_QUANT))
    assert (cfg.attn_quant, cfg.dense_quant, cfg.lm_head_quant) == ("nvfp4", "nvfp4", "none")


def test_modelopt_nvfp4_still_detected():
    cfg = parse_config(_hf_config(_MODELOPT_NVFP4_QUANT))
    assert cfg.expert_quant == "nvfp4"


def test_unquantized_checkpoint_has_no_expert_quant():
    assert parse_config(_hf_config()).expert_quant == "none"


def test_compressed_tensors_expert_source_spec():
    """The compressed-tensors spec maps weight_packed/weight_global_scale onto the canonical
    modelopt kinds and inverts the quant-side global; the modelopt spec stays identity."""
    from freetoken.models.qwen3_5_moe.weight import (
        _NVFP4_CT_SOURCE_SPEC,
        _NVFP4_SOURCE_SPEC,
    )

    ct = _NVFP4_CT_SOURCE_SPEC
    m = ct.key_pattern.match(
        "model.language_model.layers.5.mlp.experts.7.gate_proj.weight_packed"
    )
    assert m is not None
    assert (m.group("layer"), m.group("expert"), m.group("proj")) == ("5", "7", "gate_proj")
    assert ct.kind_map == {"weight_packed": "weight", "weight_global_scale": "weight_scale_2"}
    assert ct.global_reciprocal
    # W4A16 serving never consumes the calibrated activation scale.
    assert ct.key_pattern.match(
        "model.language_model.layers.5.mlp.experts.7.gate_proj.input_global_scale"
    ) is None
    # The MTP head's experts live outside the model.language_model anchor.
    assert ct.key_pattern.match(
        "mtp.layers.40.mlp.experts.7.gate_proj.weight_packed"
    ) is None
    assert _NVFP4_SOURCE_SPEC.kind_map is None
    assert not _NVFP4_SOURCE_SPEC.global_reciprocal
    assert ct.layer_to_bank(5, None) == 5  # every qwen3_5_moe layer is MoE


def test_ingest_global_inverts_only_the_quant_side_scale():
    from freetoken.models.nvfp4_banks import _ingest_global
    from freetoken.models.qwen3_5_moe.weight import (
        _NVFP4_CT_SOURCE_SPEC,
        _NVFP4_SOURCE_SPEC,
    )

    g = torch.tensor(4.0)
    assert _ingest_global(_NVFP4_CT_SOURCE_SPEC, g).item() == 0.25
    assert _ingest_global(_NVFP4_SOURCE_SPEC, g).item() == 4.0


def test_dense_pass_skips_routed_experts_but_keeps_the_shared_expert():
    """The dense pass excludes routed experts by this regex; the shared-expert MLP and the
    router gate are dense weights and must survive it."""
    from freetoken.models.qwen3_5_moe.weight import _NVFP4_EXPERT_RE

    assert _NVFP4_EXPERT_RE.search(
        "model.language_model.layers.5.mlp.experts.7.gate_proj.weight_packed"
    )
    for dense in (
        "model.language_model.layers.5.mlp.shared_expert.gate_proj.weight_packed",
        "model.language_model.layers.5.mlp.shared_expert_gate.weight",
        "model.language_model.layers.5.mlp.gate.weight",
    ):
        assert not _NVFP4_EXPERT_RE.search(dense), dense


def _ct_quant_ignoring(*modules: str) -> dict:
    return {**_CT_NVFP4_QUANT, "ignore": list(modules)}


def test_gdn_out_proj_follows_the_ignore_list():
    """This export ignores the whole linear_attn block, so the GDN out_proj is plain bf16.
    Building it as an Nvfp4DenseLinear anyway made serving die on a missing weight_scale."""
    # _NUM_LAYERS=8 with full_attention_interval=4 -> layers 3 and 7 are full attention.
    linear_ids = [i for i in range(_NUM_LAYERS) if (i + 1) % 4 != 0]
    quant = _ct_quant_ignoring(
        *[f"model.language_model.layers.{i}.linear_attn.out_proj" for i in linear_ids],
        *[f"model.language_model.layers.{i}.linear_attn.in_proj_qkv" for i in linear_ids],
        "lm_head",
    )
    cfg = parse_config(_hf_config(quant))
    assert cfg.linear_attn_quant == "none"
    assert cfg.attn_quant == "nvfp4"  # self_attn q/k/v/o are still packed FP4


def test_partially_ignored_gdn_out_proj_is_rejected():
    """One verdict drives every GDN layer, so an export that quantizes only some of them
    cannot be served -- fail loudly rather than load garbage into the rest."""
    quant = _ct_quant_ignoring("model.language_model.layers.0.linear_attn.out_proj")
    with pytest.raises(ValueError, match="partially quantized"):
        parse_config(_hf_config(quant))


def test_gdn_out_proj_stays_nvfp4_when_only_in_proj_is_ignored():
    """The Qwen3.6-27B shape: in_proj_* are bf16 by contract but out_proj is packed FP4."""
    quant = _ct_quant_ignoring(
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
        "lm_head",
    )
    cfg = parse_config(_hf_config(quant))
    assert cfg.linear_attn_quant == "nvfp4"


def test_modelopt_checkpoint_leaves_the_gdn_verdict_unset():
    """Only compressed-tensors exports carry an ignore list; modelopt keeps following
    attn_quant, so the field stays None."""
    assert parse_config(_hf_config(_MODELOPT_NVFP4_QUANT)).linear_attn_quant is None


def test_compressed_tensors_fuses_both_dense_mlp_layouts():
    """A MoE checkpoint stores the dense MLP under ``.mlp.shared_expert.``; without that
    entry gate/up never fused and the model asked for a gate_up_proj nothing supplied."""
    from freetoken.models.loader import ct_nvfp4_fuse
    from freetoken.models.qwen3_5_moe.weight import _CT_NVFP4_FUSE

    parts = (torch.zeros(4, 2), torch.zeros(4, 1), torch.zeros(4))
    buf: dict = {}
    base = "model.layers.5.mlp.shared_expert"
    assert ct_nvfp4_fuse(f"{base}.gate_proj", parts, buf, _CT_NVFP4_FUSE) == []
    emitted = ct_nvfp4_fuse(f"{base}.up_proj", parts, buf, _CT_NVFP4_FUSE)
    assert [n for n, _ in emitted] == [
        f"{base}.gate_up_proj.weight",
        f"{base}.gate_up_proj.weight_scale",
        f"{base}.gate_up_proj.weight_global",
    ]
    assert not buf
    # The bare dense layout still fuses on its own, and the two never cross-match.
    dense = "model.layers.5.mlp"
    assert ct_nvfp4_fuse(f"{dense}.gate_proj", parts, buf, _CT_NVFP4_FUSE) == []
    assert [n for n, _ in ct_nvfp4_fuse(f"{dense}.up_proj", parts, buf, _CT_NVFP4_FUSE)][0] == (
        f"{dense}.gate_up_proj.weight"
    )
    assert not buf
