"""Schneewolf Labs — Project Artemis.

ArtemisVLM is a LLaVA-style graft that adds vision-language capability to
any Mistral-family text decoder (A-series, Mahou, etc.) without modifying
the decoder. See README.md for the architecture.
"""
from __future__ import annotations

from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_artemis_vlm import ArtemisVLMConfig
from .data_collator import ArtemisDataCollator
from .losses import artemis_loss_fn
from .modeling_artemis_vlm import (
    ArtemisVLMForConditionalGeneration,
    ArtemisVLMProjector,
)
from .processing_artemis_vlm import ArtemisVLMProcessor

__version__ = "0.1.0"

__all__ = [
    "ArtemisVLMConfig",
    "ArtemisVLMProjector",
    "ArtemisVLMForConditionalGeneration",
    "ArtemisVLMProcessor",
    "ArtemisDataCollator",
    "artemis_loss_fn",
    "__version__",
]

# Register with transformers so `AutoConfig.from_pretrained(...)` and
# `AutoModelForCausalLM.from_pretrained(...)` resolve `model_type="artemis_vlm"`
# checkpoints to our classes (no trust_remote_code dance required once this
# package is imported).
AutoConfig.register("artemis_vlm", ArtemisVLMConfig)
AutoModelForCausalLM.register(ArtemisVLMConfig, ArtemisVLMForConditionalGeneration)
