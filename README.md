# Mars Habitat Sim

Mars-rover navigation research built on Habitat-Sim around the `marsyard2022` terrain. Pipeline:
DEM/heightmap → 3D terrain asset (Blender) → interactive teleop / automated rollouts in Habitat →
segmentation dataset capture → SAM2-LoRA finetuning → VLM/VLA and NavDP-diffusion navigation
policies, with a CBF safety layer and a Bayesian goal-belief tracker.

---

## Layout

| path | what |
|---|---|
| `rollout_navdp1.py` | current self-contained NavDP/S2DiT rollout script (Mars terrain adapter + full CBF cone-safety layer) — the actively-edited legacy rollout entry point |
| [sam_vla/](sam_vla/README.md) | modular rewrite of rollout/perception/policy code — prefer this for new work |
| [vl_direction/](vl_direction/DESIGN.md) | standalone HCI-study module: VLM (InternVL) discrete directional guidance |
| [belief_exp/](belief_exp/README.md) | offline harness tuning the NavDP belief tracker's covariance parameters |
| [scripts/](scripts/README.md) | teleop, legacy rollout/VLM scripts, and one-off asset-pipeline utilities |
| [sam/](sam/README.md) | standalone SAM2 training/eval harness + vendored SAM2 package (separate from `sam_vla`'s LoRA path) |
| `navdp/` | external dependency, imported not modified — own rollout/training entry points |
| `assets/` | `marsyard2022` terrain mesh/texture + Habitat's `scene_datasets`/`versioned_data` |
| `rock_envs/` | saved procedural rock-field layouts (mesh + JSON manifest) from `scripts/utils/generate_rock_env.py`, kept fixed across ablation runs |
| `annotations/` | output of `scripts/utils/mesh_annotation_tool.py` |
| `exp.md` | long-form explainer of the NavDP belief system — read before touching belief code |
| `next.md` | base-station out-and-back (two-goal) navigation plan — read before touching goal-sequencing |
| `stats.md` | SAM2-LoRA training run log |
| `usage` | real, previously-run example invocations with full flag sets, worth copying from |

---

## Conda environments

No single environment has everything installed — match the script to its env:

| env | used for |
|---|---|
| `habitat` | habitat-sim itself: teleop scripts, `mesh_annotation_tool.py`, `vl_direction` orchestration |
| `sam2` / `sam3` | SAM2/SAM3 perception code (`sam/`, `sam_vla/perception/*`), `belief_exp/*` |
| `vl` | InternVL3-8B backend for `vl_direction` — spawned automatically, don't activate directly |
| `qwen_vlm` | Qwen VLM backend for `sam_vla/vlm/*` and `scripts/vlm_nav_tests/qwen_*` — spawned automatically, don't activate directly |

Blender (with its Python API) is required for the asset-conversion scripts, run via
`blender --background --python <script>.py -- <args>`, not a conda env.

---

## 3D asset pipeline (heightmap → Habitat-loadable GLB)

```bash
python scripts/utils/hm2obj.py --heightmap <hm.png> --texture <tex.png> --size-x <m> --size-y <m> --size-z <m> [--out out.obj] [--stride 4]
blender --background --python scripts/utils/obj2glb.py -- <in.obj> <out.glb>
# or, DEM (.tif) straight to GLB:
python scripts/utils/dem2glb.py ...
# optional: upsample a DEM before converting, for a smoother mesh
python scripts/utils/upscale_dem.py <input.tif> <output.tif> [target_res]
```

Both exporters must pass `export_yup=False` to Blender's glTF export — Habitat-sim assumes raw
Z-up for scenes with no `.stage_config.json`; Blender's default (`export_yup=True`) double-rotates
the mesh, producing black-void renders and 0-hit raycasts. Both `dem2glb.py`/`obj2glb.py` already
have the fix; any new conversion script needs it too.

---

## Interactive teleop

```bash
python scripts/habitat_tests/kb_teleop.py         # base Tkinter teleop, WASD + QE clearance, Space=record, P=snapshot
python scripts/habitat_tests/kb_teleop_env.py      # same UI, driven through sam_vla.env.habitat_env.MarsHabitatEnv (matches a segmentation sweep's exact sim setup)
python scripts/habitat_tests/kb_teleop_vl.py       # same, with vl_direction wired in live
python scripts/habitat_tests/ht_vel_server.py      # UDP server variant (127.0.0.1:5055)
python scripts/habitat_tests/ht_vel_client.py vel <linear_x> <angular_y> [--rate hz] [--duration s]
python scripts/utils/pix2vid.py [--input <mars_teleop_out*>] [--fps 15]   # compile recorded frames to MP4
```

---

## Navigation rollouts

```bash
python rollout_navdp1.py --navdp-root <path> --ckpt <ckpt.pt> --scene assets/marsyard2022.glb \
    --terrain-obj assets/marsyard2022.obj --scene-height-flip-z --cbf --cbf-mode cone ... --out <name>
python -m sam_vla.run_navdp_rollout ...      # modular equivalent, see sam_vla/README.md
./sam_vla/run_qwen_vla_rollout.sh --scene_path <glb> --heightmap_path <hm.png> --out_dir <name>
```

`usage` (repo root) has real, previously-run example invocations with full flag sets worth copying
from. Earlier iterations of `rollout_navdp1.py`, plus the manual/DINO ghost-mask search scripts and
Qwen-VLM server plumbing, live under [scripts/vlm_nav_tests/](scripts/README.md).

---

## Segmentation dataset pipeline

```bash
# annotate objects on the terrain texture (habitat env)
python scripts/utils/mesh_annotation_tool.py --out-dir annotations/<name>

python -m sam_vla.run_segmentation_sweep --out-dir output/
python -m sam_vla.perception.filter_empty_segmentation_frames --run-dir <run_dir>
python -m sam_vla.perception.spot_check_segmentation --run-dir <run_dir>
python -m sam_vla.perception.export_annotations --run-dir <run_dir>

conda run -n sam2 python -m sam_vla.perception.finetune_sam2_lora \
    --run-dir output/<run_id> --out-dir sam_lora_runs/exp<N> --encoder-mode lora --epochs 20
```

Sanity-check any new sweep by overlaying `masks_category/*.png` on `rgb/*.png` before spending a
training run on it — this has burned real GPU-hours before. See [sam/README.md](sam/README.md) for
the older, non-LoRA SAM2 training harness this pipeline superseded.

---

## Belief-tracker tuning

```bash
conda run -n sam2 python belief_exp/inspect_one.py
conda run -n sam2 python belief_exp/sweep.py --configs-n 200 --episodes-per-config 60 --out belief_exp/results/sweep_001.csv
```

See [belief_exp/README.md](belief_exp/README.md) for the parameter/metric glossary.

---

## Tests

```bash
pytest vl_direction/tests/                                    # prompt/parser/contract tests, no live model needed
python scripts/habitat_tests/test_goal_pixel_ratio.py
python scripts/vlm_nav_tests/test_resolve_vlm_selection.py
python scripts/vlm_nav_tests/test_exploration_policy.py        # no Habitat-Sim or live model needed
python scripts/vlm_nav_tests/qwen_vlm_smoke_test.py            # needs a live qwen_vlm server
```

No repo-wide test runner/CI config — tests are invoked directly per-module.
