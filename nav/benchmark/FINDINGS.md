# R2R-CE benchmark vs. InternVLA-N1: findings (2026-08-26/27)

## Goal

The user wanted our own navigation stack (Qwen3.5-9B point-goal navigator + NavDP diffusion
policy, `nav/vlm_point_navigator.py` + `nav/rover_controller.py`) to actually **beat**
InternVLA-N1-DualVLN on a standard VLN benchmark, not just be compared to it informally.

## Benchmark

Real R2R-CE (VLN-CE, Krantz et al. 2020) episodes, freely downloaded (`R2R_VLNCE_v1-3`, no
registration needed) — the only scene we have license-free access to is one MP3D example scene,
`17DRP5sb8fy` (full MP3D requires Matterport's own gated registration, not completable
autonomously). 6 real `val_seen` episodes for this scene were used as the held-out eval set
throughout (`nav/benchmark/r2rce_17DRP5sb8fy_val_seen.json`); a separate 75 real `train`-split
episodes for the same scene (`r2rce_17DRP5sb8fy_train.json`) were used only for later fine-tuning
data, never for eval — a clean split matching the original dataset's own design.

Standard metrics: NE (navigation error), SR (success, <3m from goal), OSR (oracle success rate,
best point along the path), SPL (success weighted by path length vs. geodesic distance).

All final numbers are **n=18** (3 repeated runs × 6 episodes) — repetition was necessary because
early single-run numbers turned out to be dominated by noise (see below).

## Result summary

| Configuration | SR | SPL | OSR |
|---|---|---|---|
| Ours, original (pre-session) | 0.17 | 0.07 | 1.00 |
| Ours, + stop-logic fixes + prompt tightening + look-down | **0.556 ± 0.497** | 0.412 ± 0.395 | 0.94 |
| Ours, + majority-vote advance confirmation | 0.33 | 0.18 | 0.83 (reverted — worse) |
| Ours, + split point/advance into two calls (greedy advance) | 0.39 ± 0.49 | 0.30 ± 0.39 | — (not significant vs. above, reverted) |
| Ours, + LoRA fine-tune v1 (345 rows, 14 positive advance examples) | 0.33 | — | — (not significant, reverted) |
| Ours, + LoRA fine-tune v2 (1170 rows, 52 positive advance examples) | **0.222 ± 0.416** | **0.098 ± 0.188** | 1.00 (**significantly worse**, p<0.05, reverted) |
| **InternVLA-N1-DualVLN** | **1.000 ± 0.000** | **0.691 ± 0.222** | 1.00 |

**Current shipped state**: the heuristic-fixes configuration (row 2) — best measured, no adapter
loaded (`QWEN35_LORA_ADAPTER` unset).

## What actually explains the gap

OSR is ~0.83–1.00 for our system throughout — it reliably gets physically close to the real goal.
The entire SR/SPL gap to InternVLA-N1 is a **stopping-reliability** problem, not a navigation or
perception problem: InternVLA-N1 has zero variance across 18 identical-episode runs (a real STOP
token, trained end-to-end); ours succeeds roughly half the time on the exact same episode with no
code change in between, because the completion judgment is a *sampled* (temperature=0.7) output
from a frozen general-purpose VLM never trained on navigation-outcome supervision.

## What was tried, in order, and why each was abandoned

1. **Geometry-based stuck-force-advance safety net** (`nav/rover_controller.py`'s
   `VLM_STUCK_POLLS`/`VLM_STUCK_RADIUS_M`) + **tightened prompt wording** distinguishing stative
   ("wait at X") from progressive ("walk past X") instructions + **look-down retry** on
   found=False. Real, measured improvement (SR 0.17 → 0.56). Kept.
2. **Majority-vote confirmation before trusting advance=true**, plus a minimum-progress gate.
   Hypothesis: suppress false-positive early stops. Result: worse (0.56 → 0.33) — the 2-of-3
   threshold suppressed real completions more than it filtered bad ones. Reverted.
3. **Split point-selection (sampled) from advance-judgment (greedy, do_sample=False) as two
   separate Qwen calls.** Hypothesis: sampling noise in the terminal decision itself was the
   variance source. Result: no statistically significant difference from asking jointly (z≈1.0),
   despite doubling inference cost per advance check. Disproved the specific hypothesis — the
   variance is evidently compounding across the whole sampled trajectory, not isolated to one
   decision. Reverted (`nav/vlm_point_navigator.py`, `qwen35_prompts.py`,
   `qwen35_response_parser.py`, `qwen35_server.py`, `qwen35_client.py` all reverted to the
   single-call schema).
4. **LoRA fine-tune on self-collected, hindsight-labeled rollout data**
   (`nav/benchmark/collect_lora_dataset.py` drives real production rollouts through the 75 real
   train-split episodes and labels each frame using the episode's own ground-truth goal/reference
   path — privileged info used only for training labels, never at inference). `sam_vla/vlm/
   finetune_qwen35_lora.py` LoRA-adapts Qwen3.5-9B's standard attention+MLP projections (8/32
   full-attention layers + all 32 layers' MLP; the other 24 layers use an unusual linear-attention
   parameterization, left untouched). Two attempts:
   - v1: 345 rows / 14 positive advance examples → SR 0.33, not significantly different from
     baseline.
   - v2: after a 3-way parallel collection round (fixed port-isolation across
     `qwen_server_manager.py`/`qwen35_server_manager.py` — neither previously threaded `--port`
     to its spawned subprocess, a real bug fixed along the way), 1170 rows / 52 positive examples
     → SR **0.222, significantly worse than baseline (p<0.05)**.

   More and better-balanced data made the model measurably worse, not better — ruling out "not
   enough data" as the explanation. Most likely causes, untested:
   - Every training target forces `"reasoning": ""` (not supervised) — may be teaching the model
     to skip reasoning generally, degrading output quality beyond just the point/advance fields.
   - Training examples are single-turn (no history field — a known v1 simplification, see the
     dataset collection script's docstring for why), but the model is deployed with real
     multi-turn history at inference time; this train/inference mismatch could be actively
     confusing it rather than helping.
   - LoRA hyperparameters (r=8, lr=1e-4, 3 epochs) were not tuned/searched at all.

   Both adapters reverted; `qwen35_server.py` runs the frozen base model by default
   (`QWEN35_LORA_ADAPTER` env var opts in, unset by default).

## Honest bottom line

Across four different technical approaches (heuristics, confirmation voting, decision-splitting,
two rounds of LoRA fine-tuning), our system was not brought to InternVLA-N1's level, and one
approach (LoRA v2) made it significantly worse. The one approach that helped (row 1 above) is a
real, measured, ~3x SR improvement over the original baseline, and is what's currently live. If
fine-tuning is revisited, the reasoning-field and single-turn/multi-turn mismatch issues above are
the most likely next things to fix before assuming more data alone will help.

## Reusable infrastructure produced

- `nav/benchmark/r2rce_common.py`, `run_ours_episode.py`, `run_internvla_episode.py` — the
  standard-metric (NE/SR/OSR/SPL) R2R-CE eval harness for both systems, `--rep` for multi-seed
  statistical runs.
- `nav/benchmark/collect_lora_dataset.py` — hindsight-labeled rollout dataset collection from real
  production rollouts, with `--qwen-port`/`--qwen35-port`/`--navdp-port` for parallel workers.
- `sam_vla/vlm/finetune_qwen35_lora.py` — the LoRA fine-tuning loop itself, reusable for a future
  attempt with the fixes noted above.
- Port-passthrough fixes in `qwen_server_manager.py`/`qwen35_server_manager.py` (previously
  silently ignored a custom `port=`), needed for anything to run multiple Qwen instances
  concurrently.
