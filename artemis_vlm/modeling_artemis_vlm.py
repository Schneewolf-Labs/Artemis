"""Schneewolf Labs — Project Artemis: ArtemisVLM model.

A LLaVA-style vision-language wrapper that grafts Qwen3-VL's vision stack
onto an *unmodified* Schneewolf Labs A-series (Mistral) decoder.

Data flow (Path B):
    image
      -> Qwen3VLVisionModel (SigLIP-2 ViT + internal patch merger)
      -> .pooler_output            # merged: (sum_image_tokens, vision out_hidden)
      -> 2-layer MLP projector     # vision out_hidden -> text hidden
      -> spliced into text embeds at <|image_pad|> (image_token_id) positions
      -> A-series decoder (vanilla 1-D RoPE, byte-for-byte unchanged) -> logits

Deliberately NOT Qwen3-VL's decoder: no Interleaved-MRoPE, no DeepStack.
The vision tower's `deepstack_features` are intentionally ignored.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from .configuration_artemis_vlm import ArtemisVLMConfig


class ArtemisVLMProjector(nn.Module):
    """Fresh 2-layer MLP bridging vision out_hidden -> text hidden.

    Trained from scratch in Stage-1 alignment — there's no warm-start path
    because the (vision_out_dim, text_hidden_dim) pair is unique per graft.
    """

    def __init__(self, in_dim: int, out_dim: int, act: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = ACT2FN[act]
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class ArtemisVLMForConditionalGeneration(PreTrainedModel, GenerationMixin):
    config_class = ArtemisVLMConfig
    base_model_prefix = "artemis"
    _no_split_modules = ["Qwen3VLVisionBlock", "MistralDecoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    # transformers 5.x looks this up during `_finalize_model_loading` to
    # subtract tied keys from missing_keys. ArtemisVLM has no tied weights
    # (the A-series decoder is untied embed/lm_head by construction), so an
    # empty dict is the correct declaration.
    all_tied_weights_keys: dict = {}

    # Grad-checkpointing is forwarded to the language_model only (the trainable
    # decoder); the vision tower is frozen during Stage-1/2 and doesn't need it.
    # The class attr lets PreTrainedModel.gradient_checkpointing_enable accept
    # the call; the methods below explicitly delegate. Required by grimoire's
    # evaluate() which disables and re-enables grad-ckpt around eval.
    _supports_gradient_checkpointing = True

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.language_model.gradient_checkpointing_disable()

    def __init__(self, config: ArtemisVLMConfig, vision_model=None, language_model=None):
        super().__init__(config)
        # Pre-built submodules may be injected (assembly path) to avoid
        # double-instantiating a 12B decoder; otherwise build from config
        # (the from_pretrained path).
        self.visual = vision_model if vision_model is not None else Qwen3VLVisionModel(config.vision_config)
        self.language_model = (
            language_model if language_model is not None
            else AutoModelForCausalLM.from_config(config.text_config)
        )
        self.multi_modal_projector = ArtemisVLMProjector(
            config.vision_config.out_hidden_size,
            config.text_config.hidden_size,
            config.projector_hidden_act,
        )
        self.vocab_size = config.text_config.vocab_size

    # --- embedding passthrough (delegate to the A-series decoder) ---
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    # --- vision path (mirrors transformers Qwen3VLModel.get_image_features,
    #     but uses only the MERGED pooler_output; DeepStack ignored) ---
    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor):
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True)
        image_embeds = vision_output.pooler_output            # merged: (sum_tokens, out_hidden)
        return self.multi_modal_projector(image_embeds)        # (sum_tokens, text_hidden)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask=None,
        pixel_values: torch.FloatTensor = None,
        image_grid_thw: torch.LongTensor = None,
        inputs_embeds=None,
        labels=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values, image_grid_thw)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            mask = input_ids == self.config.image_token_id
            n_tokens = int(mask.sum())
            if n_tokens != image_features.shape[0]:
                raise ValueError(
                    f"Image placeholder tokens ({n_tokens}) != image features "
                    f"({image_features.shape[0]}). The processor's <|image_pad|> "
                    f"expansion must equal grid.prod()//spatial_merge_size**2."
                )
            mask = mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(mask, image_features)

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    # --- staged-freeze control (Project Artemis training recipe) ---
    def set_training_stage(self, stage: str, unfreeze_vision_top_n: int = 0):
        """Stage-1: train only the projector (freeze ViT + decoder) — connector alignment.
        Stage-2: + decoder full fine-tune (projector stays trainable); ViT frozen unless
        `unfreeze_vision_top_n` top blocks are opened. Returns (#trainable, #total)."""
        if stage not in ("stage1", "stage2"):
            raise ValueError("stage must be 'stage1' or 'stage2'")
        for p in self.visual.parameters():
            p.requires_grad = False
        for p in self.language_model.parameters():
            p.requires_grad = (stage == "stage2")
        for p in self.multi_modal_projector.parameters():
            p.requires_grad = True
        if unfreeze_vision_top_n > 0:
            blocks = getattr(self.visual, "blocks", None)
            if blocks is not None:
                for blk in list(blocks)[-unfreeze_vision_top_n:]:
                    for p in blk.parameters():
                        p.requires_grad = True
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return tr, tot

    # --- generation: inject the image only on the first step ---
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None,
        pixel_values=None, image_grid_thw=None, is_first_iteration=False, **kwargs
    ):
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values,
            attention_mask=attention_mask, is_first_iteration=is_first_iteration,
            **kwargs
        )
        # transformers >= 5 pre-initializes the KV cache before prefill, so
        # `past_key_values is None` never holds inside generate(); the prefill
        # signal is `is_first_iteration` (same gating as upstream Llava).
        # `past_key_values is None` is kept for direct calls outside generate(),
        # and cache-less generation re-runs the full prompt every step, so it
        # needs the pixels every step.
        if is_first_iteration or past_key_values is None or not kwargs.get("use_cache", True):
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_grid_thw"] = image_grid_thw
        else:
            model_inputs["pixel_values"] = None
            model_inputs["image_grid_thw"] = None
        return model_inputs

    @classmethod
    def from_a2_and_vision(
        cls,
        text_model_path: str,
        vision_model: Qwen3VLVisionModel | None = None,
        vision_config: Qwen3VLVisionConfig | None = None,
        image_token_id: int = 22,
        video_token_id: int = 23,
        torch_dtype=torch.bfloat16,
    ) -> "ArtemisVLMForConditionalGeneration":
        """Assemble ArtemisVLM from a trained A-series checkpoint + a Qwen3-VL vision
        tower (pretrained module passed in, or random from `vision_config`).

        Loads the text decoder exactly once (no random double-instantiation).

        The name keeps `from_a2_and_vision` for back-compat with existing Merlina
        call sites; despite the name, any Mistral-class checkpoint works
        (`flammenai/Mahou-1.5-mistral-nemo-12B` etc.) since the projector adapts
        to the text model's hidden size automatically.
        """
        language_model = AutoModelForCausalLM.from_pretrained(text_model_path, dtype=torch_dtype)
        if vision_model is None:
            vision_model = Qwen3VLVisionModel(vision_config or Qwen3VLVisionConfig())
        config = ArtemisVLMConfig(
            vision_config=vision_model.config.to_dict(),
            text_config=language_model.config.to_dict(),
            image_token_id=image_token_id,
            video_token_id=video_token_id,
        )
        model = cls(config, vision_model=vision_model, language_model=language_model)
        model.multi_modal_projector.to(language_model.device, torch_dtype)
        model.visual.to(language_model.device, torch_dtype)
        return model
