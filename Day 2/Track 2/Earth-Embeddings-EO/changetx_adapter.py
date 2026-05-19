"""Transformers-native ChangeTX bi-temporal retrieval adapter.

This module is intentionally stored in the workshop repository, not on the
Hugging Face artifact repo. The Hub bundle contains weights and data only; the
notebook imports this local class and calls `from_pretrained(bundle_dir)`.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, SiglipConfig, SiglipModel
from transformers.modeling_outputs import ModelOutput


class ChangeTXBiTemporalConfig(PretrainedConfig):
    model_type = "changetx_siglip_bitemporal"

    def __init__(
        self,
        siglip_config: dict[str, Any] | None = None,
        embedding_dim: int = 768,
        fusion_hidden_mult: float = 2.0,
        fusion_dropout: float = 0.0,
        use_temporal_pos_encoding: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if siglip_config is None:
            siglip_config = SiglipConfig().to_dict()
        self.siglip_config = siglip_config
        self.embedding_dim = int(embedding_dim)
        self.fusion_hidden_mult = float(fusion_hidden_mult)
        self.fusion_dropout = float(fusion_dropout)
        self.use_temporal_pos_encoding = bool(use_temporal_pos_encoding)


class ChangeTXBiTemporalOutput(ModelOutput):
    pre_features: torch.Tensor | None = None
    post_features: torch.Tensor | None = None
    image_features: torch.Tensor | None = None
    text_features: torch.Tensor | None = None


class ChangeTXFusionHead(nn.Module):
    def __init__(self, embed_dim: int, hidden_mult: float = 2.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = max(1, int(embed_dim * hidden_mult))
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=max(0.0, float(dropout))),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, pre_features: torch.Tensor, post_features: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([pre_features, post_features], dim=-1))


class ChangeTXBiTemporalModel(PreTrainedModel):
    model_type = "changetx_siglip_bitemporal"
    config_class = ChangeTXBiTemporalConfig
    base_model_prefix = "changetx"
    main_input_name = "pre_pixel_values"
    supports_gradient_checkpointing = False

    def __init__(self, config: ChangeTXBiTemporalConfig) -> None:
        super().__init__(config)
        self.siglip = SiglipModel(SiglipConfig.from_dict(config.siglip_config))
        self.embedding_dim = int(config.embedding_dim)
        self.use_temporal_pos_encoding = bool(config.use_temporal_pos_encoding)
        self.fusion_head = ChangeTXFusionHead(
            embed_dim=self.embedding_dim,
            hidden_mult=float(config.fusion_hidden_mult),
            dropout=float(config.fusion_dropout),
        )
        if self.use_temporal_pos_encoding:
            self.pre_pos = nn.Parameter(torch.zeros(self.embedding_dim))
            self.post_pos = nn.Parameter(torch.zeros(self.embedding_dim))
        else:
            self.register_parameter("pre_pos", None)
            self.register_parameter("post_pos", None)
        self.post_init()

    def encode_image(self, pixel_values: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        features = self.siglip.vision_model(pixel_values=pixel_values).pooler_output
        if normalize:
            features = F.normalize(features.float(), dim=-1)
        return features

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        normalize: bool = False,
    ) -> torch.Tensor:
        features = self.siglip.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        if normalize:
            features = F.normalize(features.float(), dim=-1)
        return features

    def encode_fused_image(
        self,
        pre_features: torch.Tensor,
        post_features: torch.Tensor,
        normalize: bool = False,
    ) -> torch.Tensor:
        if self.use_temporal_pos_encoding:
            pre_features = pre_features + self.pre_pos
            post_features = post_features + self.post_pos
        fused = self.fusion_head(pre_features, post_features)
        if normalize:
            fused = F.normalize(fused.float(), dim=-1)
        return fused

    def encode_pair(
        self,
        pre_pixel_values: torch.Tensor,
        post_pixel_values: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        pre_features = self.encode_image(pre_pixel_values, normalize=False)
        post_features = self.encode_image(post_pixel_values, normalize=False)
        return self.encode_fused_image(pre_features, post_features, normalize=normalize)

    def forward(
        self,
        pre_pixel_values: torch.Tensor | None = None,
        post_pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        normalize: bool = True,
        **_: Any,
    ) -> ChangeTXBiTemporalOutput:
        image_features = None
        pre_features = None
        post_features = None
        text_features = None

        if pre_pixel_values is not None and post_pixel_values is not None:
            pre_features = self.encode_image(pre_pixel_values, normalize=False)
            post_features = self.encode_image(post_pixel_values, normalize=False)
            image_features = self.encode_fused_image(pre_features, post_features, normalize=normalize)

        if input_ids is not None:
            text_features = self.encode_text(input_ids, attention_mask=attention_mask, normalize=normalize)

        return ChangeTXBiTemporalOutput(
            pre_features=pre_features,
            post_features=post_features,
            image_features=image_features,
            text_features=text_features,
        )
