from __future__ import annotations

import re
from typing import Iterator

import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import (
    MergeRule,
    ShardReader,
    iter_merged_tensors,
    iter_stacked_experts,
    nvfp4_parts_ct,
)
from freetoken.models.nvfp4_banks import Nvfp4ExpertSourceSpec
from freetoken.utils import cached_load_hf_config
from tqdm import tqdm

from .config import parse_config

# Routed MLP experts: streamed into the offload cache, never into the resident model.
_ROUTED_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
_ROUTED_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")
# MoVA value-experts: stacked into resident banks here. The leading "v_" keeps these out
# of _ROUTED_EXPERT_RE (and out of the engine's is_routed_expert) by construction.
_V_EXPERT_RE = re.compile(r"^model\.layers\.(?P<layer>\d+)\.self_attn\.v_experts\.(?P<expert>\d+)\.")

_NVFP4_EXPERT_KEY_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\."
    r"(?P<kind>weight_packed|weight_scale|weight_global_scale)$"
)
# llm-compressor export: weight_packed | weight_scale | weight_global_scale, the global
# being the quant-side scale (reciprocated at ingest). input_global_scale is deliberately
# absent -- the routed-expert paths are W4A16 and never quantize activations.
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_NVFP4_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: (
        None
        if layer < config.first_k_dense_replace or layer >= config.num_layers
        else layer - config.first_k_dense_replace
    ),
    desc="K2-Horizon NVFP4 experts",
    kind_map={"weight_packed": "weight", "weight_global_scale": "weight_scale_2"},
    global_reciprocal=True,
)

_MERGE_RULES = {
    ".gate_proj": MergeRule(".gate_up_proj", "gate", ("gate", "up")),
    ".up_proj": MergeRule(".gate_up_proj", "up", ("gate", "up")),
}


def _mova_banks(
    reader: ShardReader, config, layer: int
) -> Iterator[tuple[str, torch.Tensor]]:
    """One layer's value-experts, stacked into the ``[E, ...]`` banks the resident MoVA
    layer declares.

    The checkpoint keeps one tensor per expert; the kernels want a single bank indexed by
    expert id. The per-tensor ``weight_global_scale`` is broadcast across output rows and
    reciprocated, which is the same shape every other NVFP4 bank in the engine arrives in.
    """
    base = f"model.layers.{layer}.self_attn.v_experts"
    experts = range(config.k2_args.num_value_experts)

    # Asked of the checkpoint rather than of config.quant, which the engine only attaches
    # after parse_config; an export may quantize the MLP experts and leave these dense.
    if not reader.has(f"{base}.0.weight_packed"):
        yield f"{base}.weight", torch.stack(
            [reader.get_tensor(f"{base}.{e}.weight") for e in experts]
        )
        return

    parts = [nvfp4_parts_ct(reader, f"{base}.{e}") for e in experts]
    yield f"{base}.weight_packed", torch.stack([p[0] for p in parts])
    # stacked through uint8: the copy is the same either way and fp8 stack is not
    # available on every torch build
    yield f"{base}.weight_scale", torch.stack([p[1].view(torch.uint8) for p in parts]).view(
        torch.float8_e4m3fn
    )
    yield f"{base}.weight_global_scale", torch.stack([p[2] for p in parts])


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Resident weights for K2-Horizon.

    MoVA's value-experts are resident and so come through here, not the offload cache:
    as NVFP4 they are under 4 GiB, and routing them per token inside attention gives the
    streaming machinery nothing to prefetch against. The routed MLP experts keep the
    normal split (``nvfp4_expert_spec`` for a quantized checkpoint, stacked bf16 here).
    """
    config = parse_config(cached_load_hf_config(model_path))
    if get_tp_info().size > 1:
        raise NotImplementedError("k2_horizon weight loading supports TP=1 only")

    reader = ShardReader(model_path, device)
    try:
        def selected(want_experts: bool) -> Iterator[tuple[str, torch.Tensor]]:
            for file in tqdm(reader.files(), desc="Loading weights", disable=not want_experts):
                for name in reader.names_in(file):
                    if (_ROUTED_EXPERT_RE.search(name) is not None) != want_experts:
                        continue
                    # value-experts are emitted as stacked banks below, not one by one
                    if _V_EXPERT_RE.match(name):
                        continue
                    yield name, reader.get_tensor(name)

        if include_non_moe:
            # No merge rules here: K2HorizonMLP keeps gate_proj and up_proj apart, so the
            # dense layers and shared experts load straight off the checkpoint names.
            yield from selected(want_experts=False)
            for layer in range(config.first_k_dense_replace, config.num_layers):
                yield from _mova_banks(reader, config, layer)

        if include_moe_experts:
            # Only reached for a bf16 checkpoint; a quantized one streams the experts
            # through nvfp4_expert_spec instead. The bank wants one fused gate_up row.
            merged = iter_merged_tensors(
                selected(want_experts=True), _MERGE_RULES, model_name="k2_horizon"
            )
            yield from iter_stacked_experts(
                merged,
                num_experts=config.num_experts,
                model_name="k2_horizon",
                expert_pattern=_ROUTED_EXPERT_PATTERN,
            )
    finally:
        reader.close()


def nvfp4_expert_spec(model_path: str, config):
    return _NVFP4_SOURCE_SPEC


__all__ = ["iter_weights", "nvfp4_expert_spec"]
