# sam

Standalone SAM2 training/eval harness, separate from `sam_vla/perception/`'s LoRA finetuning
pipeline. `sam/sam/` is a vendored clone of Meta's [SAM2](https://github.com/facebookresearch/segment-anything-2)
repo (own `README.md`/`LICENSE`, unmodified) providing the base `sam2` package; the files at
`sam/` top level are this project's training/eval/inference code written against that package.

| file | what |
|---|---|
| `train_sam2_simple_fast.py` | fine-tunes SAM2 for 4-class semantic segmentation (sand/rock/etc.) |
| `evaluate_sam2_simple_fast.py` | evaluation harness for a trained checkpoint |
| `inference.py` | single-frame inference entry point — imported by `scripts/utils/sam_annotation_adapter.py` and `scripts/vlm_nav_tests/vlm_nav_demo.py` as `sam/sam/inference.py` |
| `requirements.txt` | env deps for this harness — also covers `peft` (used by `sam_vla.perception.finetune_sam2_lora`) and `rasterio` (used by `sam_vla.env.terrain`), so it doubles as a rough superset for the `sam2`/`sam3` conda envs |

This predates `sam_vla/perception/finetune_sam2_lora.py`, which is the currently-preferred
LoRA-based finetuning path reading segmentation-sweep run directories directly. Nothing here is
imported by `sam_vla` except `sam/sam/inference.py` (via the adapter above) and the vendored
`sam2` package itself.
