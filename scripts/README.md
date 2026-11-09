# scripts

Everything here used to live at the repo root; it's grouped by purpose now. Nothing under
`scripts/` is a Python package (no relative imports between the three subdirs) — each file is run
directly, `python scripts/<subdir>/<file>.py`, from whichever conda env its docstring calls for.

## `habitat_tests/`

Bare Habitat-Sim interaction: teleop UIs and raw sensor smoke tests. Runs in the `habitat` env.

| file | what |
|---|---|
| `kb_teleop.py` | base Tkinter WASD teleop (Space=record, P=snapshot, Q/E=clearance probe) |
| `kb_teleop_env.py` | same UI, but driven through `sam_vla.env.habitat_env.MarsHabitatEnv` with annotation meshes registered — matches `sam_vla.run_segmentation_sweep`'s exact sim setup, for eyeballing what a sweep would capture |
| `kb_teleop_vl.py` | same UI, with `vl_direction` wired in live against a synthetic obstacle field (prints VL prompt/response/latency per frame) |
| `ht_vel_server.py` / `ht_vel_client.py` | UDP velocity-command server (`127.0.0.1:5055`) + CLI client, for driving the sim from a separate process |
| `rgbd_test.py` / `rgbd_drive.py` | minimal raw-`habitat_sim` RGBD sensor checks, no rover/teleop scaffolding |
| `test_goal_pixel_ratio.py` | standalone test for `goal_pixel_ratio()` (imported from `scripts/vlm_nav_tests/rollout_navdp.py`) against a saved mask image |

## `vlm_nav_tests/`

Legacy self-contained NavDP-rollout and Qwen-VLM scripts — the generation of code `sam_vla/` is the
modular rewrite of. Prefer extending `sam_vla/` for new work; these stay for reference and for flags
not yet ported over. `rollout_navdp_policy.py` here is the archived version — the actively-edited
copy lives at the repo root as `rollout_navdp1.py`.

| file | what |
|---|---|
| `rollout_navdp.py`, `rollout_navdp2.py`, `rollout_navdp_policy.py` | earlier iterations of the Mars NavDP/S2DiT rollout adapter (full CBF cone-safety layer, belief-adapter hookup) — see the root `rollout_navdp1.py` for the current one |
| `qwen_search_rollout.py` | manual left/right/straight/back search via a "ghost" goal mask, auto-handoff to the real goal once visible — no VLM in the loop |
| `qwen_search_dino.py` | same, but hands off once GroundingDINO visually confirms the goal instead of using known-world-point geometry |
| `qwen_vlm_server.py` / `qwen_vlm_client.py` | persistent Qwen2.5-VL-3B inference server (`qwen_vlm` env) + stdlib-only TCP client, so callers without torch installed can poll it at a few Hz |
| `qwen_vlm_smoke_test.py` | one-shot load-model-and-generate check for the `qwen_vlm` env |
| `test_qwen_vlm_persistent.py` | measures per-call latency against the persistent server |
| `vlm_query.py` | standalone Qwen2.5-VL query runner, invoked as a subprocess from `vlm_nav_interactive.py` (which runs in the torch-less `habitat` env) |
| `vlm_nav_demo.py` | free-camera scene viewer that runs captured frames through the trained SAM2 model (`sam/sam/inference.py`) |
| `vlm_nav_interactive.py` | interactive frame capture + labelme annotation handoff for VLM navigation, built on `vlm_nav_demo.py`'s scene config |
| `verify_vlm_nav_setup.py` | checks required files (scene, heightmap, `labels.txt`) exist before running the above |
| `test_resolve_vlm_selection.py` | standalone test for `resolve_vlm_selection()` against an already-captured frame + annotation pair |
| `test_exploration_policy.py` | standalone test for `sam_vla.policy.exploration_policy.ExplorationPolicy`'s leg state machine, using `vl_direction.client.MockInternVLClient` — no Habitat-Sim or live model needed |
| `round1_walk_rgbd.py` | early raw walk/RGBD scratch script |

## `utils/`

One-off/asset-pipeline utilities, not part of a rollout loop.

| file | what |
|---|---|
| `hm2obj.py` | heightmap PNG → textured `.obj` terrain mesh |
| `obj2glb.py` / `dem2glb.py` | Blender-run `.obj`/DEM `.tif` → Habitat-loadable `.glb` (must pass `export_yup=False`) |
| `upscale_dem.py` | bicubic-upsamples a DEM `.tif` (e.g. 257×257 → 1025×1025) for a smoother mesh |
| `mesh_annotation_tool.py` | interactive top-down annotation of objects on the terrain texture, output feeds `sam_vla`'s segmentation sweep |
| `generate_rock_env.py` | generates a reusable random rock-field layout (meshes + JSON manifest) under `rock_envs/<name>/`, so obstacle placement stays fixed across ablation runs |
| `sam2_auto_masks_folder.py` | runs SAM2 automatic-mask-generation over a folder of frames |
| `sam_annotation_adapter.py` | adapts single-frame SAM inference (`sam/sam/inference.py`) output into the labelme-style annotation JSON `resolve_vlm_selection()` expects |
| `pix2vid.py` | compiles recorded teleop frames into an MP4 |
| `log_reader.py` | parses rollout log directories under `logs/` into summary tables (single-episode or batch/CSV) |
