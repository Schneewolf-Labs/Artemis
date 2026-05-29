# Changelog

All notable changes to Artemis will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-05-29

### Added
- ArtemisMix-shape Stage 2 dataset reader support
- A3 checkpoint loader for `vlm_stage2` continuation

## [0.1.0] - 2026-05-24

### Added
- Initial public release of `artemis-vlm`. LLaVA-style graft that adds
  vision-language capability to Mistral-family decoders by composing
  Qwen3-VL's vision tower + a fresh 2-layer MLP projector + an
  unmodified language model.
- **`ArtemisVLMConfig`** — composite config (`Qwen3VLVisionConfig` +
  text config + `image_token_id`/`video_token_id`).
- **`ArtemisVLMForConditionalGeneration`** with `from_a2_and_vision()`
  helper and `set_training_stage("stage1"|"stage2")` for the two-stage
  recipe (projector-only alignment → full multimodal FFT).
- **`ArtemisVLMProcessor`** — wraps `Qwen2VLImageProcessor` with
  patch/temporal/merge sourced from the model's vision_config; mirrors
  Qwen3VLProcessor's `<|image_pad|>` token expansion.
- **`ArtemisDataCollator`** — multimodal batching with prefix-trick
  label masking (prompt + image tokens → -100).
- **`artemis_loss_fn`** — grimoire-compatible `(loss, metrics)` adapter.
- Extracted from Merlina's `src/artemis_vlm.py` (v1.6.0) so the model
  classes can be installed independently of the Merlina FastAPI app.
