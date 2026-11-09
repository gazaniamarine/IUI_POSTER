# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Mars-rover navigation research on top of Habitat-Sim, built around the `marsyard2022` terrain scene. The
pipeline runs, in rough order: DEM/heightmap → 3D terrain asset (Blender) → interactive teleop / automated
rollouts in Habitat → segmentation dataset capture → SAM2-LoRA finetuning → VLM/VLA and NavDP-diffusion
navigation policies, with a CBF safety layer and a Bayesian goal-belief tracker. There is also a standalone
HCI-study module (`vl_direction`) that asks a VLM (InternVL) for discrete directional guidance, and an
offline harness (`belief_exp`) for tuning the belief tracker's parameters against the real NavDP belief
classes.

`exp.md` is a long-form written explainer of the NavDP belief system (state → refinement → encoding →
policy integration → runtime control) — read it before touching anything belief-related instead of
re-deriving it from `navdp/`. `next.md` currently documents the base-station out-and-back (two-goal)
navigation plan — read it before touching goal-sequencing/route logic instead of re-deriving it.
`vl_direction/DESIGN.md` holds that module's own design doc (moved out of `next.md`, which no longer
tracks it). `stats.md` logs SAM2-LoRA training runs.

## Conda environments

This repo spans several conda environments; there is no single environment with everything installed.
Match the script to its env, not the other way around:

| env | used for |
|---|---|
| `habitat` | habitat-sim itself: `kb_teleop*.py`, `mesh_annotation_tool.py`, `sam_vla` env/rollout code, `vl_direction` orchestration (has tkinter + scipy) |
| `sam2` / `sam3` | SAM2/SAM3 perception code (`sam/`, `sam_vla/perception/*`), and `belief_exp/*` (only needed because `navdp.extensions` transitively imports torch-using modules at load time, not because belief_exp itself needs a GPU) |
| `vl` | InternVL3-8B backend for `vl_direction` (torch 2.11+cu128, transformers 5.x) — spawned automatically as a subprocess by `InternVLServerManager`, you don't activate it yourself |
| `qwen_vlm` | Qwen VLM backend for `sam_vla/vlm/*` — same pattern, spawned by `QwenServerManager` (env name is read from code; if `conda env list` doesn't show it, it needs creating before Qwen-based rollouts will run) |

`conda_env.py` is the shared helper (`resolve_conda_base()`) that scripts use to locate the base conda
install without hardcoding a machine-specific path — reuse it rather than hardcoding `~/miniconda3`.

Blender (with its Python API) is required for `obj2glb.py` / `dem2glb.py`, run via
`blender --background --python <script>.py -- <args>`, not a conda env.

## Common commands

### 3D asset pipeline (heightmap → Habitat-loadable GLB)

```bash
python hm2obj.py --heightmap <hm.png> --texture <tex.png> --size-x <m> --size-y <m> --size-z <m> [--out out.obj] [--stride 4]
blender --background --python obj2glb.py -- <in.obj> <out.glb>
# or, DEM (.tif) straight to GLB:
python dem2glb.py ...
```

**Both exporters must pass `export_yup=False`** to Blender's glTF export — Habitat-sim assumes raw
Z-up for scenes with no `.stage_config.json`, and Blender's default (`export_yup=True`) double-rotates
the mesh, producing black-void renders and 0-hit raycasts. See "Known issues" below before regenerating
`assets/marsyard2022.glb`.

### Interactive teleop

```bash
python kb_teleop.py            # base Tkinter teleop, WASD + QE clearance, Space=record, P=snapshot
python kb_teleop_vl.py         # same, with vl_direction wired in live (prints VL prompt/response/latency per frame)
python ht_vel_server.py        # UDP server variant (127.0.0.1:5055)
python ht_vel_client.py vel <linear_x> <angular_y> [--rate hz] [--duration s]
python pix2vid.py [--input <mars_teleop_out*>] [--fps 15]   # compile recorded frames to MP4
```

### Segmentation dataset pipeline (`next.md` steps 1–6)

```bash
# Step 1: annotate objects on the terrain texture (habitat env: has tkinter+scipy)
/home/gpu/miniconda3/envs/habitat/bin/python mesh_annotation_tool.py --out-dir annotations/<name>

# Steps 3-6, in order:
python -m sam_vla.run_segmentation_sweep --out-dir output/
python -m sam_vla.perception.filter_empty_segmentation_frames --run-dir <run_dir>
python -m sam_vla.perception.spot_check_segmentation --run-dir <run_dir>
python -m sam_vla.perception.export_annotations --run-dir <run_dir>

# LoRA finetune (sam2 env; reads run_dir directly, does not need export_annotations first)
conda run -n sam2 python -m sam_vla.perception.finetune_sam2_lora \
    --run-dir output/<run_id> --out-dir sam_lora_runs/exp<N> --encoder-mode lora --epochs 20
```

Sanity-check any new sweep by overlaying `masks_category/*.png` on `rgb/*.png` for a few frames before
spending a training run on it (see "Known issues" — this pipeline has burned real GPU-hours twice).

### Belief-tracker tuning harness (`belief_exp/`)

Reads the real `navdp.extensions.SubgoalBeliefBank`/`RouteManager` — never copies or reimplements them.
Must run in an env with torch (`sam2`/`sam3`), even though the harness itself is numpy-only:

```bash
conda run -n sam2 python belief_exp/inspect_one.py                    # single-config step trace
conda run -n sam2 python belief_exp/sweep.py --configs-n 200 --episodes-per-config 60 --out belief_exp/results/sweep_001.csv
```

### Navigation rollouts

```bash
python rollout_navdp_policy.py --navdp-root <path> --ckpt <ckpt.pt> --scene assets/marsyard2022.glb \
    --terrain-obj assets/marsyard2022.obj --scene-height-flip-z --cbf --cbf-mode cone ... --out <name>
python -m sam_vla.run_navdp_rollout ...      # modular equivalent, see Architecture
./sam_vla/run_qwen_vla_rollout.sh --scene_path <glb> --heightmap_path <hm.png> --out_dir <name>
```

`usage` (repo root) has real, previously-run example invocations with full flag sets worth copying from.

### Tests

```bash
pytest vl_direction/tests/                 # prompt/parser/contract tests, no live model needed (MockInternVLClient)
python test_goal_pixel_ratio.py
python test_resolve_vlm_selection.py
python qwen_vlm_smoke_test.py              # needs a live qwen_vlm server
```

There's no repo-wide test runner/CI config — tests are invoked directly per-module. `vl_direction/tests/`
is the one real pytest suite; everything else is a standalone smoke/verification script.

## Architecture

### Two generations of rollout code

The top-level `rollout_navdp*.py`, `rollout_navdp_policy.py`, `vlm_nav_interactive.py`, `vlm_nav_demo.py`
are large, self-contained legacy scripts (originally written to be copy-pasted into this repo from a
sibling NavDP repo — see their docstrings). `sam_vla/` is the modular rewrite of the same
capabilities — prefer extending `sam_vla/` for new work; only touch the top-level scripts for legacy
compatibility or when a flag exists there but hasn't been ported yet.

### `sam_vla/` package layout

- `env/` — Habitat-sim wrapper. `habitat_env.py:MarsHabitatEnv` is the central sim handle; `terrain.py`'s
  `HeightmapGrid` is the *sole authority* on world-Y from the heightmap (every other height computation
  must match its normalize-then-subtract-mean convention). `annotation_meshes.py` loads the mesh
  annotations produced by `mesh_annotation_tool.py` as semantic scene objects; `rock_generation.py`
  places procedural rock fields.
- `core/` — sim-independent domain types and math: `types.py` (Action/GoalSpec/Pose/Observation),
  `goal_geometry.py` (bbox↔world backprojection, mesh IDs), `belief_tracking.py`
  (`BeliefGoalTracker`, distinct from `navdp`'s belief system — this is the rollout-loop-facing wrapper),
  `pose_integrator.py`, `lifecycle.py` (`ServiceRegistry`).
- `policy/` — pluggable policies sharing `base_policy.py`: `navdp_policy.py` (diffusion policy),
  `qwen_vla_policy.py`, `qwen_discrete_direction_policy.py`.
- `safety/` — `cbf_avoidance.py` (control-barrier-function cone steering) + `safety_filter.py`, consumed
  by rollout loops regardless of which policy is driving.
- `vlm/` — Qwen VLM client/server plumbing (`qwen_server_manager.py` spawns the `qwen_vlm` conda env
  subprocess; `qwen_prompts.py` / `qwen_response_parser.py` handle prompt templating and output parsing).
- `perception/` — SAM2/SAM3 segmentation: dataset capture (`segmentation_capture.py`), LoRA finetuning
  (`finetune_sam2_lora.py`), inference (`predict_lora.py`), CLIP-based goal classification
  (`clip_goal_classifier.py`, `sam3_goal_tracker.py`), and export to COCO/YOLO formats
  (`export_annotations.py`).
- `goal_resolution/` — turns a task description into a concrete goal (`first_frame_resolver.py` = fixed
  first-frame bbox; `goal_vocabulary_resolver.py` = open-vocabulary via SAM3+CLIP).
- `logging/` — `episode_logger.py` (per-run JSONL) and `rollout_logger.py`.

Entry-point scripts (`run_navdp_rollout.py`, `run_qwen_vla_rollout.py`, `run_segmentation_sweep.py`) live
at the top of `sam_vla/` and wire these submodules together; the submodules themselves have no
`__main__`.

### `vl_direction/` — standalone VL-directional-guidance module

Deliberately decoupled from `sam_vla`: a pure function `directive_engine.query(mode, frames, context,
episode_id) -> VLDirectiveResult` is the *only* integration point, backed by InternVL via
`InternVLClient` (swappable backend). Three modes — `cbf` (LEFT/RIGHT go-around), `exploration`
(LEFT/RIGHT/FRONT/BACK prior), `uncertainty` (stateful heading-request sub-flow driven externally by
`uncertainty_session.py`, not looped internally) — each with its own prompt template under `prompts/`
and closed-vocabulary output enforced by `parser.py` (never trust free text past `raw_response`). Exists
to support an HCI study (VL-autonomy vs. human-teleop-intervention), so `intervention/mode_flag.py` +
`logging/hci_metrics.py` track session mode and aggregate success/intervention metrics. Full design
rationale is in the module's own docstrings and `vl_direction/DESIGN.md` — read
`vl_direction/directive_engine.py` and `schemas.py` first when extending it.

### `belief_exp/` — offline belief-parameter tuning harness

Not part of the runtime path; a research tool that drives the *real* `navdp.extensions.SubgoalBeliefBank`
+ `RouteManager` through simulated occlusion/noise scenarios (`scenario.py`) to find well-calibrated
covariance parameters (`sigma_init`, `sigma_visible`, `odom_noise`, `decay_factor`,
`large_uncertainty`). See `belief_exp/README.md` for the full parameter/metric glossary — in particular,
`calibration_nll`/`coverage_1sigma`/`coverage_2sigma` measure whether `Sigma` honestly reflects real
error, not just whether it's small.

### `navdp/`

Treated as an external, read-from dependency — belief_exp's README explicitly notes nothing under
`navdp/` is copied or modified, only imported. `navdp/scripts/` has its own rollout/training entry points
independent of `sam_vla`.

## Known issues to check before trusting output

- **GLB export must use `export_yup=False`.** If a freshly-regenerated scene GLB renders half-black or a
  segmentation sweep returns `"objects": []` for every frame, check this first (both `dem2glb.py` and
  `obj2glb.py` already have the fix, but any new conversion script needs it too). DEM→mesh height
  normalization must also subtract the mean, matching `sam_vla/env/terrain.py`'s `HeightmapGrid` exactly,
  or the camera ends up embedded below the surface uniformly.
- **Annotation masks are thin silhouette slivers, not filled objects — currently unresolved.**
  `mesh_annotation_tool.py`'s hulls are flat, terrain-following patches occluded almost everywhere by the
  real rock geometry, so every SAM2-LoRA checkpoint trained via `run_segmentation_sweep.py` output so far
  (`sam_lora_runs/exp1`–`exp5`) has learned a thin curved-line artifact, not rock appearance — expect
  near-zero detection confidence on real (e.g. `kb_teleop`) frames. Don't spend a training run on new
  sweep output without first overlaying `masks_category/` on `rgb/` and confirming filled (not sliver)
  masks. A related RGB-baking/label-leakage bug (the hull rendering into the training RGB itself) was
  fixed 2026-08-01 via two-pass rendering in `habitat_env.py:get_full_observation` — that fix is
  independent and does not address the mask-shape problem above.
- **`sam_lora_runs/exp1` was trained on pre-fix broken renders** and should not be used for anything;
  `exp2` onward were trained on valid renders (but still hit the mask-shape issue above).
