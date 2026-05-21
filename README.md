# Artemis — Schneewolf Labs

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A LLaVA-style graft that adds vision-language capability to any Mistral-family
text decoder *without modifying the decoder*. Built originally for the
Schneewolf Labs A-series, but architecturally Mistral-Nemo agnostic — point it
at any Mistral-class checkpoint (A2, A3, Mahou, Flammades, etc.) and you get
an ArtemisVLM around it.

## Path B by design

```
       PIL Image
           │
           ▼
   ┌───────────────────────┐
   │  Qwen3-VL ViT         │  patches → ViT layers → merger
   │  (FROZEN, pixels only)│
   └───────────────────────┘
           │  N vectors of dim out_hidden_size
           ▼
   ┌───────────────────────┐
   │  Projector (trained)  │  2-layer MLP, out_hidden → text_hidden
   │  ~45M params          │
   └───────────────────────┘
           │  N vectors in the text decoder's hidden space
           ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  Mistral-family decoder (FROZEN in Stage-1, full-FT Stage-2)   │
   │  At each <|image_pad|> position, OVERWRITE the embedding with │
   │  the next projector vector. Then run as a normal decoder.     │
   └───────────────────────────────────────────────────────────────┘
           │
           ▼
       text output (decoder's own vocab — Qwen vocab never seen)
```

The vision tower processes pixels (no text tokens). The projector bridges
*hidden spaces*, not token spaces. The decoder is byte-identical to the
underlying Mistral checkpoint — its vocab, weights, chat template, reasoning,
tool calling, and identity are preserved by construction.

## Install

```bash
pip install artemis-vlm
```

Or, from source:

```bash
git clone https://github.com/Schneewolf-Labs/Artemis.git
cd Artemis
pip install -e .
```

Requires `transformers>=5.0.0`, `torch>=2.5.0`, `Pillow`.

## Quick start — load a pretrained Artemis checkpoint

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import artemis_vlm  # registers ArtemisVLM with AutoConfig / AutoModel

REPO = "schneewolflabs/A3-preview"  # or any ArtemisVLM checkpoint

model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16).to("cuda").eval()
tok = AutoTokenizer.from_pretrained(REPO)
processor = artemis_vlm.ArtemisVLMProcessor(
    tokenizer=tok, vision_config=model.visual.config,
    min_pixels=32 * 32, max_pixels=512 * 512,
)

from PIL import Image
image = Image.open("photo.jpg")
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "Describe this image in detail."},
]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
batch = processor(text=text, images=[image], return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**batch, max_new_tokens=200, do_sample=False)
print(tok.decode(out[0][batch["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Quick start — build a new graft from your own checkpoints

```python
import torch
import artemis_vlm
from transformers import Qwen3VLForConditionalGeneration

# Take the vision tower from a pretrained Qwen3-VL checkpoint
qv = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", dtype=torch.bfloat16,
)
vision = qv.model.visual
del qv  # free the Qwen3-VL decoder we don't need

# Graft onto any Mistral-class text checkpoint
model = artemis_vlm.ArtemisVLMForConditionalGeneration.from_a2_and_vision(
    "schneewolflabs/A2",  # or any Mistral-Nemo finetune
    vision_model=vision,
    image_token_id=22,    # repurposed <|image_pad|> in A-series Tekken vocab
    torch_dtype=torch.bfloat16,
)

# Stage-1: train only the projector (~45M params)
trainable, total = model.set_training_stage("stage1")
print(f"Stage-1: trainable={trainable/1e6:.1f}M / total={total/1e9:.2f}B")
```

## Training (Stage-1 / Stage-2)

`set_training_stage("stage1")` freezes the ViT and the decoder, leaving only
the projector trainable — the "alignment" phase. `set_training_stage("stage2")`
unfreezes the decoder for the visual-instruction phase.

The recommended trainer is [Schneewolf-Labs/Merlina](https://github.com/Schneewolf-Labs/Merlina),
which exposes Artemis training as `training_mode: "vlm_stage1"` / `"vlm_stage2"`
on its REST API. The `ArtemisDataCollator` here is `data_collator=`-compatible
with any trainer that consumes a custom collator (Grimoire, accelerate-driven
loops, HF `Trainer`).

## Key implementation notes

- **Merged vision features.** `Qwen3VLVisionModel.forward()` returns pre-merge
  features on `last_hidden_state` and merged features on `pooler_output`. We
  use `pooler_output` (matches the merger's downstream-consumer contract).
- **Patch / merge sizes come from `vision_config`.** Qwen3-VL uses
  `patch_size=16`; Qwen2-VL's image processor defaults to `patch_size=14`. The
  processor sources patch / temporal / merge from `vision_config` so the
  `<|image_pad|>` expansion count can never drift from the model's merged
  feature count.
- **Image token splice.** At each `<|image_pad|>` position in the prompt, the
  input embedding is overwritten with the next projector vector (via
  `masked_scatter`). The decoder sees a normal token sequence where some
  embeddings happen to come from vision instead of `embed_tokens`.
- **DeepStack / Interleaved-MRoPE are intentionally NOT used.** Those are
  decoder-modification ("Path A") tricks. We chose Path B (composition).
- **Untied weights.** A-series decoders have untied `embed_tokens` and
  `lm_head`. `ArtemisVLMForConditionalGeneration.all_tied_weights_keys = {}`
  is declared explicitly for transformers 5.x compatibility.

## Tests

Four hardware-bound smoke tests live in `tests/`. They require a real
checkpoint on disk + an ML stack + a CUDA device, so they skip cleanly under
`pytest` (CI won't try to run them) and are meant to be invoked as
`python tests/test_artemis_<name>.py` on the development machine.

```bash
python tests/test_artemis_vlm.py        # model assembly + forward
python tests/test_artemis_processor.py  # chat template ↔ pad expansion
python tests/test_artemis_collator.py   # multimodal batching
python tests/test_artemis_stage_gen.py  # staged-freeze + generate()
```

## Published checkpoints

| Checkpoint | Status | Notes |
|---|---|---|
| `schneewolflabs/A3-preview` | public, apache-2.0 | 25k-sample Stage-1 smoke (proof-of-concept) |
| `schneewolflabs/A3` | training (Stage-1, 1M samples) | first real release |
| `schneewolflabs/Artemis` | planned (post Stage-2) | named flagship |

## License

Apache 2.0 — see [LICENSE](LICENSE).
