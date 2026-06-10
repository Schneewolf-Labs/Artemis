"""Regression test: generate() must inject image features at prefill.

The bug (fixed in 0.1.5): `prepare_inputs_for_generation` gated image
injection on `past_key_values is None`. transformers >= 5 pre-initializes
the cache *before* the prefill forward, so that condition never held inside
`generate()` — `pixel_values` was silently dropped on every step (forward
only validates the placeholder/feature count when pixel_values is not None)
and generation was image-blind. The prefill signal in transformers 5.x is
the `is_first_iteration` kwarg.

Unlike the four hardware smoke scripts, this is a real pytest test: tiny
random configs, CPU-only, no checkpoint or tokenizer required. It skips
only when the ML stack itself is absent (the no-stack collection CI job).

Run: pytest tests/test_generate_image_injection.py -v
"""
import pytest

torch = pytest.importorskip("torch", reason="needs the ML stack (CPU is fine)")
pytest.importorskip("transformers", reason="needs the ML stack (CPU is fine)")

from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

from artemis_vlm import ArtemisVLMConfig, ArtemisVLMForConditionalGeneration

IMG_PAD = 22


def tiny_artemis():
    """Random-weight ArtemisVLM small enough for CPU (<1M params)."""
    vision_config = Qwen3VLVisionConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=32,
        num_position_embeddings=64,  # must be a perfect square (8x8 pos grid)
        deepstack_visual_indexes=[],  # unused by Artemis (Path B)
    )
    text_config = MistralConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    config = ArtemisVLMConfig(
        vision_config=vision_config.to_dict(),
        text_config=text_config.to_dict(),
        image_token_id=IMG_PAD,
    )
    torch.manual_seed(0)
    return ArtemisVLMForConditionalGeneration(config).eval()


def image_batch(model):
    """Synthetic prompt + pixels honoring the placeholder/feature contract."""
    vcfg = model.config.vision_config
    grid_thw = torch.tensor([[1, 4, 4]])  # 16 patches
    n_patches = int(grid_thw.prod())
    patch_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size**2
    pixel_values = torch.randn(n_patches, patch_dim)
    n_pads = n_patches // vcfg.spatial_merge_size**2  # 4 merged image tokens
    input_ids = torch.tensor([[1, 5, 6] + [IMG_PAD] * n_pads + [7, 8]])
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask, pixel_values, grid_thw


def count_vision_calls(model):
    """Shadow get_image_features with a counting wrapper; returns the counter."""
    calls = {"n": 0}
    orig = model.get_image_features

    def counting(pixel_values, image_grid_thw):
        calls["n"] += 1
        return orig(pixel_values, image_grid_thw)

    model.get_image_features = counting
    return calls


def test_generate_injects_image_at_prefill_with_cache():
    """The regression: with the default KV cache, the vision tower must run
    exactly once (prefill) — 0 means generate() is image-blind, >1 would
    trip forward's count contract on decode steps."""
    model = tiny_artemis()
    input_ids, attention_mask, pixel_values, grid_thw = image_batch(model)
    calls = count_vision_calls(model)

    new_tokens = 4
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
            max_new_tokens=new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=None,  # tiny random model: don't stop early
            pad_token_id=0,
        )

    assert calls["n"] == 1, (
        f"get_image_features ran {calls['n']} times during generate(); "
        "expected exactly once at prefill. 0 == the pre-0.1.5 image-blind bug."
    )
    assert out.shape == (1, input_ids.shape[1] + new_tokens)


def test_generate_injects_image_without_cache():
    """use_cache=False re-runs the full prompt every step, so pixels must be
    forwarded on every step (and the count contract must hold each time)."""
    model = tiny_artemis()
    input_ids, attention_mask, pixel_values, grid_thw = image_batch(model)
    calls = count_vision_calls(model)

    new_tokens = 3
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
            max_new_tokens=new_tokens,
            do_sample=False,
            use_cache=False,
            eos_token_id=None,
            pad_token_id=0,
        )

    assert calls["n"] >= 1, "image features never computed with use_cache=False"
    assert out.shape == (1, input_ids.shape[1] + new_tokens)
