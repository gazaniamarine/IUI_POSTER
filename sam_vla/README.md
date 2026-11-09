# sam_vla

Modular rewrite of the top-level `rollout_navdp*.py` / `vlm_nav_*.py` legacy scripts — prefer
extending this package for new work; only touch the top-level scripts for legacy compatibility or
flags not yet ported here. See the root [README.md](../README.md) and [CLAUDE.md](../CLAUDE.md) for
the full pipeline and conda-env matrix.

## Layout

- `env/` — Habitat-sim wrapper. `habitat_env.py:MarsHabitatEnv` is the central sim handle;
  `terrain.py`'s `HeightmapGrid` is the sole authority on world-Y from the heightmap; every other
  height computation must match its normalize-then-subtract-mean convention.
  `annotation_meshes.py` loads mesh annotations from `mesh_annotation_tool.py`; `rock_generation.py`
  places procedural rock fields.
- `core/` — sim-independent domain types and math: `types.py` (Action/GoalSpec/Pose/Observation),
  `goal_geometry.py` (bbox↔world backprojection), `belief_tracking.py` (`BeliefGoalTracker`,
  distinct from `navdp`'s belief system), `pose_integrator.py`, `lifecycle.py` (`ServiceRegistry`).
- `policy/` — pluggable policies sharing `base_policy.py`: `navdp_policy.py` (diffusion policy),
  `qwen_vla_policy.py`, `qwen_discrete_direction_policy.py`.
- `safety/` — `cbf_avoidance.py` (control-barrier-function cone steering) + `safety_filter.py`,
  consumed by rollout loops regardless of which policy is driving.
- `vlm/` — Qwen VLM client/server plumbing; `qwen_server_manager.py` spawns the `qwen_vlm` conda
  env subprocess.
- `perception/` — SAM2/SAM3 segmentation: dataset capture, LoRA finetuning (`finetune_sam2_lora.py`),
  inference (`predict_lora.py`), CLIP-based goal classification, COCO/YOLO export.
- `goal_resolution/` — task description → concrete goal (`first_frame_resolver.py` = fixed
  first-frame bbox; `goal_vocabulary_resolver.py` = open-vocabulary via SAM3+CLIP).
- `logging/` — `episode_logger.py` (per-run JSONL) and `rollout_logger.py`.

Entry points (`run_navdp_rollout.py`, `run_qwen_vla_rollout.py`, `run_exploration_rollout.py`,
`run_segmentation_sweep.py`) live at the top level and wire submodules together; the submodules
themselves have no `__main__`.

## Usage

```bash
python -m sam_vla.run_navdp_rollout ...
python -m sam_vla.run_segmentation_sweep --out-dir output/
./sam_vla/run_qwen_vla_rollout.sh --scene_path <glb> --heightmap_path <hm.png> --out_dir <name>
```

See the root README's "Navigation rollouts" and "Segmentation dataset pipeline" sections for full
command sequences.
