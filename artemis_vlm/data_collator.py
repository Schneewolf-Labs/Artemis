"""ArtemisDataCollator — multimodal collator for ArtemisVLM training.

Each feature is a dict::

    {"images": [PIL.Image, ...], "messages": [chat turns ...]}

where `messages` is a chat list (image placeholders + text); the final
turn (assistant) is the training target. Produces a batch consumable by
`ArtemisVLMForConditionalGeneration.forward`:

  input_ids / attention_mask / labels  -- right-padded across the batch
                                           (prompt + <|image_pad|> = -100)
  pixel_values                          -- flat concat over all images
  image_grid_thw                        -- (num_images_in_batch, 3)

Label masking mirrors the validated text path (prefix trick): tokenize the
prompt (everything but the last assistant turn, with `add_generation_prompt`)
to get its length under the *same* image expansion, mask that prefix, and
additionally mask every `<|image_pad|>` position (vision input, not a target).
"""
from __future__ import annotations


class ArtemisDataCollator:
    def __init__(self, processor, label_pad: int = -100):
        self.proc = processor
        self.tok = processor.tokenizer
        self.label_pad = label_pad
        self.img_id = processor.image_token_id
        self.pad_id = (
            self.tok.pad_token_id if self.tok.pad_token_id is not None
            else self.tok.eos_token_id
        )

    def _ids(self, messages, images, add_gen):
        text = self.proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_gen
        )
        return self.proc(text=text, images=images, return_tensors="pt")

    def __call__(self, features):
        import torch

        seqs, labels, pvs, grids = [], [], [], []
        for f in features:
            msgs, imgs = f["messages"], f.get("images")
            full = self._ids(msgs, imgs, add_gen=False)
            prompt = self._ids(msgs[:-1], imgs, add_gen=True)  # everything but target
            ids = full["input_ids"][0]
            plen = prompt["input_ids"].shape[1]
            lab = ids.clone()
            lab[:plen] = self.label_pad                          # mask prompt
            lab[ids == self.img_id] = self.label_pad             # mask image placeholders
            seqs.append(ids)
            labels.append(lab)
            if "pixel_values" in full:
                pvs.append(full["pixel_values"])
                grids.append(full["image_grid_thw"])

        maxlen = max(s.size(0) for s in seqs)
        input_ids, attn, lbl = [], [], []
        for s, l in zip(seqs, labels):
            pad = maxlen - s.size(0)
            input_ids.append(torch.cat([s, torch.full((pad,), self.pad_id, dtype=s.dtype)]))
            attn.append(torch.cat([torch.ones(s.size(0), dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
            lbl.append(torch.cat([l, torch.full((pad,), self.label_pad, dtype=l.dtype)]))
        batch = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attn),
            "labels": torch.stack(lbl),
        }
        if pvs:
            batch["pixel_values"] = torch.cat(pvs, dim=0)        # flat over all images
            batch["image_grid_thw"] = torch.cat(grids, dim=0)
        return batch
