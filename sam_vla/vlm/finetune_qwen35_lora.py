"""Phase B of the LoRA fine-tuning plan (see /home/gpu/.claude/plans/
prancy-plotting-lerdorf.md): LoRA-finetune Qwen3.5-9B on the hindsight-
labeled rollout dataset nav/benchmark/collect_lora_dataset.py produced
(nav/benchmark/lora_dataset/train.jsonl), targeting the diagnosed
bottleneck (the advance/stop decision) without needing the full 90-scene
MP3D dataset -- see finetune_sam2_lora.py for this repo's existing LoRA
precedent (SAM2's Hiera trunk instead of a causal VLM here).

Qwen3.5-9B is a hybrid architecture: 8/32 language-decoder layers are
standard softmax self-attention (q_proj/k_proj/v_proj/o_proj), the other 24
are a linear-attention variant (linear_attn.in_proj_a/b/qkv/z) -- confirmed
live via model.named_modules(). This LoRA targets only the well-understood
standard modules across all 32 layers (q/k/v/o_proj where present,
gate/up/down_proj in every layer) and the vision tower is left untouched --
adapting the language model's reasoning over features it already extracts
is enough to shift point/advance decisions without the added risk of
LoRA-wrapping the less-understood linear-attention parameterization.

Every training row's target JSON always includes a well-formed "advance"
field (required by qwen35_response_parser.parse_point_goal_history_response
at inference time) -- rows with target.advance=None (non-terminal-
instruction ticks, no ground truth available) default to False for
training purposes: "keep going" is the correct answer for the large
majority of mid-trajectory frames anyway, and this avoids needing
sub-string-level loss masking to skip just the advance token span. Rows
with target.advance=True (rare -- the whole point of this dataset) are
oversampled (see --true-advance-oversample) since they're both the
scarcest and most valuable label for the diagnosed bottleneck.

    conda activate qwen_vlm
    python -m sam_vla.vlm.finetune_qwen35_lora --dataset nav/benchmark/lora_dataset/train.jsonl \\
        --out-dir sam_vla/vlm/qwen35_lora_runs/exp1 [--epochs 3] [--lr 1e-4] [--true-advance-oversample 5]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from sam_vla.vlm.qwen35_model_runner import load_qwen35_model

DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class LoraPointGoalDataset(Dataset):
    def __init__(self, jsonl_path: str, true_advance_oversample: int = 5):
        rows = [json.loads(line) for line in open(jsonl_path)]
        oversampled = []
        for row in rows:
            oversampled.append(row)
            if row["target"]["advance"] is True:
                oversampled.extend([row] * (true_advance_oversample - 1))
        self.rows = oversampled

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        target = dict(row["target"])
        if target["advance"] is None:
            target["advance"] = False  # see module docstring
        target["reasoning"] = ""  # not supervised -- keep target strings short/stable
        image = Image.open(row["frame_path"]).convert("RGB")
        return {"image": image, "prompt": row["prompt"], "target_json": json.dumps(target)}


def _build_example(processor, image, prompt: str, target_json: str, device) -> dict:
    prompt_messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
    ]
    prompt_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_enc = processor(text=[prompt_text], images=[image], return_tensors="pt")

    full_text = prompt_text + target_json + processor.tokenizer.eos_token
    full_enc = processor(text=[full_text], images=[image], return_tensors="pt")

    labels = full_enc["input_ids"].clone()
    prompt_len = prompt_enc["input_ids"].shape[1]
    labels[:, :prompt_len] = -100

    return {
        "input_ids": full_enc["input_ids"].to(device),
        "attention_mask": full_enc["attention_mask"].to(device),
        "pixel_values": full_enc["pixel_values"].to(device) if "pixel_values" in full_enc else None,
        "image_grid_thw": full_enc.get("image_grid_thw", None),
        # Qwen3.5's M-RoPE needs this alongside image_grid_thw (confirmed
        # live: forward() raises ValueError without it) -- the processor
        # already returns it, just wasn't being forwarded.
        "mm_token_type_ids": full_enc.get("mm_token_type_ids", None),
        "labels": labels.to(device),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--true-advance-oversample", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[!] loading base Qwen3.5-9B model")
    model, processor = load_qwen35_model()
    model.train()

    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=list(DEFAULT_TARGET_MODULES),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = LoraPointGoalDataset(args.dataset, args.true_advance_oversample)
    n_val = max(1, int(len(dataset) * args.val_frac))
    indices = list(range(len(dataset)))
    random.Random(0).shuffle(indices)
    val_indices, train_indices = indices[:n_val], indices[n_val:]
    print(f"[!] dataset: {len(dataset)} examples ({len(train_indices)} train / {len(val_indices)} val)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    device = next(model.parameters()).device

    def run_batch(idx: int) -> torch.Tensor:
        item = dataset[idx]
        example = _build_example(processor, item["image"], item["prompt"], item["target_json"], device)
        kwargs = {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "labels": example["labels"],
        }
        if example["pixel_values"] is not None:
            kwargs["pixel_values"] = example["pixel_values"]
        if example["image_grid_thw"] is not None:
            kwargs["image_grid_thw"] = example["image_grid_thw"].to(device)
        if example["mm_token_type_ids"] is not None:
            kwargs["mm_token_type_ids"] = example["mm_token_type_ids"].to(device)
        out = model(**kwargs)
        return out.loss

    for epoch in range(args.epochs):
        random.Random(epoch).shuffle(train_indices)
        model.train()
        total_loss, n = 0.0, 0
        for step, idx in enumerate(train_indices):
            loss = run_batch(idx)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n += 1
            if step % 20 == 0:
                print(f"[epoch {epoch}] step {step}/{len(train_indices)} loss={loss.item():.4f}")
        print(f"[epoch {epoch}] mean train loss = {total_loss / max(n, 1):.4f}")

        model.eval()
        val_loss, n_val_done = 0.0, 0
        with torch.no_grad():
            for idx in val_indices:
                val_loss += run_batch(idx).item()
                n_val_done += 1
        print(f"[epoch {epoch}] mean val loss = {val_loss / max(n_val_done, 1):.4f}")

        ckpt_dir = out_dir / f"epoch{epoch}"
        model.save_pretrained(str(ckpt_dir))
        print(f"[!] saved adapter to {ckpt_dir}")


if __name__ == "__main__":
    main()
