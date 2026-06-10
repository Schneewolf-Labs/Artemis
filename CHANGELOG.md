# Changelog

All notable changes to Artemis will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.5] - 2026-06-10

### Fixed
- **`generate()` was image-blind under transformers >= 5.**
  `prepare_inputs_for_generation` gated image injection on
  `past_key_values is None`, but transformers 5.x pre-initializes the KV
  cache *before* prefill, so `pixel_values` was silently dropped on every
  step (forward only validates the placeholder/feature count when
  `pixel_values is not None`). The gate now uses the `is_first_iteration`
  prefill signal (same as upstream Llava), keeps the `None` check for
  direct calls outside `generate()`, and forwards pixels on every step
  when `use_cache=False`. Workaround on earlier versions:
  `generate(..., use_cache=False)`. Training (`forward` with labels) was
  never affected.

### Added
- CPU-only pytest regression test (`tests/test_generate_image_injection.py`)
  that counts vision-tower invocations through a real `generate()` call —
  tiny random configs, no checkpoint needed, runs in CI.
- `tests.yml`: new `unit` CI job with a CPU ML stack (torch + transformers)
  that actually runs the pytest suite; the no-stack collection job is
  unchanged.

## [0.1.4] - 2026-06-06

### Added
- Collator: per-row `enable_thinking`, auto-detected from whether the final
  assistant turn carries a `<think>...</think>` block (`auto_detect_thinking`,
  default on). Non-reasoning rows bake the empty think wrapper into the
  prompt prefix so the model never trains on emitting a stray `</think>`.

## [0.1.3] - 2026-05-29

### Fixed
- Modeling: `gradient_checkpointing_enable()/disable()` now forward to the
  `language_model` (the trainable decoder), so trainers that toggle
  grad-checkpointing around eval work with ArtemisVLM.

## [0.1.2] - 2026-05-29

### Fixed
- **Release workflow**: `pytest tests/ --collect-only` exited 5 ("no tests
  collected") because all four test files skip at module load — the v0.1.1
  release publish job failed on this. Replaced with a package-import check
  that actually validates a fresh-install wheel.
- **Tests workflow**: same exit-5 problem; accepts exit 5 explicitly now
  while still failing on real import / collection errors.

### Note
- v0.1.1 was tagged but never reached PyPI due to the workflow bug. 0.1.2
  is the first PyPI release; nothing else changed in the package itself.

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
