"""ArtemisVLMConfig — composite config for the Artemis vision-language model.

A Qwen3-VL vision tower + an A-series (Mistral) text decoder, joined by a
learned projector. The text decoder is not modified — its config is carried
through unchanged so reload behavior matches the underlying decoder.
"""
from __future__ import annotations

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig


class ArtemisVLMConfig(PretrainedConfig):
    """Composite config: a Qwen3-VL vision tower + an A-series (Mistral) text decoder."""

    model_type = "artemis_vlm"
    sub_configs = {"vision_config": Qwen3VLVisionConfig, "text_config": AutoConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id: int = 22,   # repurposed <|image_pad|> in the A-series tokenizer
        video_token_id: int = 23,   # repurposed <|video_pad|>
        projector_hidden_act: str = "gelu",
        **kwargs,
    ):
        if vision_config is None:
            vision_config = Qwen3VLVisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = Qwen3VLVisionConfig(**vision_config)
        self.vision_config = vision_config

        if text_config is None:
            text_config = AutoConfig.for_model("mistral")
        elif isinstance(text_config, dict):
            text_config = AutoConfig.for_model(**text_config)
        self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.projector_hidden_act = projector_hidden_act
        super().__init__(**kwargs)
