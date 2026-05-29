"""ArtemisVLMProcessor — image+text processor for ArtemisVLM.

Wraps Qwen2-VL's image processor (which Qwen3-VL itself reuses) and the
A-series tokenizer. Critically, patch/temporal/merge sizes are sourced from
the model's `vision_config` so the processor's `<|image_pad|>` expansion can
never drift from the model's merged feature count (the contract the model's
forward enforces).

Expansion mirrors transformers' `Qwen3VLProcessor.__call__`: each single
`<|image_pad|>` (the chat template emits one per image, framed by
`<|vision_start|>`/`<|vision_end|>`) is replaced by
`grid_thw[i].prod() // merge_size**2` copies.
"""
from __future__ import annotations


class ArtemisVLMProcessor:
    def __init__(
        self,
        tokenizer,
        vision_config,
        image_token: str = "<|image_pad|>",
        video_token: str = "<|video_pad|>",
        min_pixels: int = 32 * 32,
        max_pixels: int = 512 * 512,
    ):
        from transformers import Qwen2VLImageProcessor

        self.tokenizer = tokenizer
        # patch/temporal/merge from vision_config (can't drift from the model);
        # min/max_pixels cap dynamic-resolution token blow-up on large images
        # (e.g. 2560x1920 uncapped -> ~4800 image tokens).
        self.image_processor = Qwen2VLImageProcessor(
            patch_size=vision_config.patch_size,
            temporal_patch_size=vision_config.temporal_patch_size,
            merge_size=vision_config.spatial_merge_size,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        self.merge_length = vision_config.spatial_merge_size ** 2
        self.image_token = image_token
        self.video_token = video_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def __call__(self, text=None, images=None, return_tensors="pt", **kwargs):
        from transformers.feature_extraction_utils import BatchFeature

        image_inputs = {}
        grid = None
        if images:  # non-empty list; text-only rows pass [] (or None) -> no pixel_values
            image_inputs = self.image_processor(images=images, return_tensors=return_tensors)
            grid = image_inputs["image_grid_thw"]

        if text is None:
            return BatchFeature(data=image_inputs)
        if isinstance(text, str):
            text = [text]

        if grid is not None:
            idx = 0
            out_text = []
            for t in text:
                while self.image_token in t:
                    n = int(grid[idx].prod()) // self.merge_length
                    t = t.replace(self.image_token, "<|placeholder|>" * n, 1)
                    idx += 1
                out_text.append(t.replace("<|placeholder|>", self.image_token))
            text = out_text

        text_inputs = self.tokenizer(text, return_tensors=return_tensors, **kwargs)
        return BatchFeature(data={**text_inputs, **image_inputs})
