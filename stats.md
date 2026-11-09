# SAM2 Training

## Run 1

### command used:

```
python -m sam_vla.perception.finetune_sam2_lora --run-dir output/2026-07-30_160329_segmentation-sweep-goal_grid_obsna_seed0/ --out-dir sam_lora_runs/exp1 --encoder-mode lora --epochs 20
```

### Stats

```
classes: ['background', 'small_rock', 'big_rock', 'bedrock', 'hole_in_ground'] (5)
inputs ->train: 1576 frames, val: 175 frames
encoder_mode=lora, trainable params: 2,291,869 / 214,995,149 (1.07%)

```

### start

<!-- epoch 1/20 train_loss=1.6588 val_loss=1.0078 val_mIoU=0.5354 -->

```
epoch 1/20 train_loss=1.9283 val_loss=1.1120 val_mIoU=0.5601
```

### end

<!-- epoch 20/20 train_loss=0.6541 val_loss=0.6616 val_mIoU=0.7995 -->

```
epoch 20/20 train_loss=0.6550 val_loss=0.6599 val_mIoU=0.8055
```

### best

<!-- epoch 19/20 train_loss=0.6552 val_loss=0.6607 val_mIoU=0.8027 -->

```
epoch 20/20 train_loss=0.6550 val_loss=0.6599 val_mIoU=0.8055
```

### Confusion Matrix & Eval

```
training took 6670.51 seconds
[eval] confusion matrix on validation set (175 frames):
                  background  small_rock    big_rock     bedrock  hole_in_gr
background       183,309,518           0      37,864           0           0
small_rock                 0           0           0           0           0
big_rock              34,447           0     118,971           0           0
bedrock                    0           0           0           0           0
hole_in_ground             0           0           0           0           0
[eval] overall pixel accuracy (val set): 0.9996

[eval] pixel accuracy on 10 random image(s) from the dataset:
    idx=   275  accuracy=0.9997
    idx=  1165  accuracy=0.9999
    idx=  1735  accuracy=0.9996
    idx=  1643  accuracy=0.9996
    idx=  1564  accuracy=0.9996
    idx=   129  accuracy=0.9999
    idx=   522  accuracy=0.9997
    idx=   241  accuracy=0.9998
    idx=  1014  accuracy=0.9999
    idx=  1558  accuracy=0.9999
[eval] mean accuracy over 10 random image(s): 0.9997
```

## Run 2

### command used:

```
python -m sam_vla.perception.finetune_sam2_lora --run-dir output/2026-07-31_180313_segmentation-sweep-goal_grid_obsna_seed10/ --out-dir sam_lora_runs/exp2 --encoder-mode lora --epochs 20
```

### Stats

```
classes: ['background', 'small_rock', 'big_rock', 'bedrock', 'hole_in_ground'] (5)
inputs -> train: 12238 frames, val: 1359 frames
encoder_mode=lora, trainable params: 2,291,869 / 214,995,149 (1.07%)

```

### start

```
epoch 1/20 train_loss=1.0274 val_loss=0.7834 val_mIoU=0.6817
```

### end

```
epoch 20/20 train_loss=0.0372 val_loss=0.0379 val_mIoU=0.8681
```

### best

```
epoch 20/20 train_loss=0.0372 val_loss=0.0379 val_mIoU=0.8681
```

### conf matrix and stats

```
training took 51662.67 seconds

[eval] confusion matrix on validation set (1359 frames):
                  background  small_rock    big_rock     bedrock  hole_in_gr
background      1,423,568,814           0     226,142           0           0
small_rock                 0           0           0           0           0
big_rock             147,115           0   1,072,713           0           0
bedrock                    0           0           0           0           0
hole_in_ground             0           0           0           0           0
[eval] overall pixel accuracy (val set): 0.9997

[eval] pixel accuracy on 10 random image(s) from the dataset:
    idx=  2201  accuracy=0.9998
    idx=  9325  accuracy=0.9999
    idx= 13144  accuracy=0.9999
    idx= 12513  accuracy=0.9999
    idx=  1033  accuracy=0.9999
    idx=  4179  accuracy=0.9999
    idx=  1931  accuracy=0.9997
    idx=  8117  accuracy=0.9997
    idx= 12467  accuracy=0.9999
    idx=  7364  accuracy=0.9998
[eval] mean accuracy over 10 random image(s): 0.9998
```

```

16 real Qwen calls across every bearing bucket (ahead, side, directly behind), with actual rendered frames from the sim, no mocking. The geometry math checks out (I hand-verified the bearing/distance numbers against the trig by hand — correct), but the model's placement barely tracks its own input:

bearing	expected	got (u, frac of width)
-102.5° (right, behind)	near right edge (~0.85)	u=376, 0.59
-147.5° (right, far behind)	pinned at right edge	u=376, 0.59 — identical to the -102.5° case
+167.5° (left, far behind)	pinned at left edge	u=300, 0.47 — barely off-center
+77.5° (left)	clearly left of center	u=320, dead center — directly violates the prompt's "NOT near u=320" instruction
-12.5° (small, right)	barely off-center	u=376, 0.59 — same offset as the -102.5°/-147.5° cases
+32.5° (left)	somewhat left	u=320, dead center
Side (left/right) is usually right, but magnitude is essentially decoupled from the actual bearing — a 12° bearing and a 147° bearing produce the same pixel offset, and two cases land exactly on the disallowed center pixel despite the prompt explicitly forbidding that. This isn't a bug in the conversion code (belief_to_bearing_range_uncertainty's numbers check out); it's Qwen2.5-VL-3B failing to do proportional numeric reasoning from text, which is exactly your symptom — clustered near center, never near the sides, occasionally nonsensical.

The good news: the deterministic fallback path (project_or_clamp_body_point_to_pixel, which you already extended to clamp to the correct edge instead of disappearing) doesn't have this problem — it's exact trig, always lands on the correct side with proportional magnitude, and already exists behind --no-ghost-mask-vlm.
```
