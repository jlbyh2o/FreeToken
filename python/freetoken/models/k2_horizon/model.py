from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GroupedRMSNormFused,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.utils import nvtx_annotate

from freetoken.models.blocks import BaseLLMModel

from .attention import K2HorizonAttention, K2HorizonMoVAAttention
from .moe import K2HorizonMLP, K2HorizonSparseBlock

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class K2HorizonDecoderLayer(BaseOP):
    """A sparse layer pairs MoVA attention with the routed MoE block; the leading dense
    layers pair a plain value projection with a plain MLP. The checkpoint never mixes the
    two, so one flag selects both halves."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        is_sparse = layer_id >= config.first_k_dense_replace
        attn_cls = K2HorizonMoVAAttention if is_sparse else K2HorizonAttention
        self.self_attn = attn_cls(config, layer_id, prefix=f"{prefix}.self_attn")
        self.mlp = (
            K2HorizonSparseBlock(config, layer_id, prefix=f"{prefix}.mlp")
            if is_sparse
            else K2HorizonMLP(
                config.hidden_size,
                config.intermediate_size,
                quant_config=config.quant,
                prefix=f"{prefix}.mlp",
            )
        )
        groups = config.k2_args.layernorm_num_groups
        self.input_layernorm = GroupedRMSNormFused(
            config.hidden_size, config.rms_norm_eps, groups
        )
        self.post_attention_layernorm = GroupedRMSNormFused(
            config.hidden_size, config.rms_norm_eps, groups
        )
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class K2HorizonModel(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [
                K2HorizonDecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = GroupedRMSNormFused(
            config.hidden_size, config.rms_norm_eps, config.k2_args.layernorm_num_groups
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class K2HorizonForCausalLM(BaseLLMModel):
    model_cls = K2HorizonModel

    def __init__(self, config: ModelConfig):
        self.model = self.model_cls(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant,
            prefix="lm_head",
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


__all__ = ["K2HorizonForCausalLM"]
