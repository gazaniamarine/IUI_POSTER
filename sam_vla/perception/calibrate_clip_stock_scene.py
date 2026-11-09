"""
Standalone calibration script for --clip-goal-thresh / --clip-reid-thresh
against a stock Habitat-Sim scene, instead of the Mars terrain -- per user
direction in the approved multi-goal plan: Mars's monochrome rock-colored
palette makes it a poor testbed for visually validating CLIP classification
decisions.

Does NOT go through MarsHabitatEnv (sam_vla/env/habitat_env.py), which is
tightly coupled to the Mars heightmap/terrain/rock-generation pipeline
(heightmap_path, terrain.py, rock_generation.py) -- this script drives
habitat_sim directly against a standard bundled example scene instead,
reusing only the small generic sensor/pose helpers from sam_vla.env.sim_utils
(make_sensor, set_agent_pose, rgb_depth), not the Mars-specific parts of that
package.

One-time setup (run once, from the `habitat` conda env, which already has
habitat_sim installed): download Habitat-Sim's bundled example test scenes --

    conda run -n habitat python -m habitat_sim.utils.datasets_download \\
        --uids habitat_test_scenes --data-path <path-to-download-into>

then pass one of the resulting .glb files (e.g.
<path>/habitat-test-scenes/skokloft-tallahassee.glb) as --scene-glb below.

Cross-env note: this script needs BOTH habitat_sim (to capture frames) AND
the sam3/torch/open_clip stack (for Sam3GoalTracker/ClipGoalClassifier). At
the time this was written no single conda env in this repo has both (see
next.md's plan: `habitat` has habitat_sim only, `sam3` has torch/sam3/
open_clip only). If no merged env exists yet, split the run across two
invocations instead of one:

    conda run -n habitat python -m sam_vla.perception.calibrate_clip_stock_scene \\
        --scene-glb <path> --dump-frames-dir /tmp/calib_frames --capture-only

    conda run -n sam3 python -m sam_vla.perception.calibrate_clip_stock_scene \\
        --load-frames-dir /tmp/calib_frames --vocab-terms "chair,door,picture frame,window,table"

Usage (single merged env, once one exists):
    conda run -n <merged_env> python -m sam_vla.perception.calibrate_clip_stock_scene \\
        --scene-glb <path> --vocab-terms "chair,door,picture frame,window,table"
"""

import argparse
import colorsys
import hashlib
import os

import numpy as np
from PIL import Image, ImageDraw


def _color_for_goal_id(goal_id: str) -> tuple[int, int, int]:
    """Deterministic distinct-ish RGB color per goal_id (same object keeps the
    same color across frames, so re-ID is visible at a glance in the GIF)."""
    digest = hashlib.sha256(goal_id.encode("utf-8")).digest()
    hue = digest[0] / 255.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def _annotate_frame(
    frame: np.ndarray, mask_infos: list[dict], alpha: float = 0.4
) -> np.ndarray:
    """Alpha-blends each tracked goal's mask in its own color and draws a
    category/score/goal_id label near its bbox. mask_infos items:
    {"mask": bool ndarray, "goal_id": str, "category": str, "score": float}."""
    overlaid = np.asarray(frame, dtype=np.float32).copy()
    for info in mask_infos:
        color = np.asarray(_color_for_goal_id(info["goal_id"]), dtype=np.float32)
        mask = info["mask"]
        overlaid[mask] = (1.0 - alpha) * overlaid[mask] + alpha * color
    overlaid = np.clip(overlaid, 0, 255).astype(np.uint8)

    img = Image.fromarray(overlaid)
    draw = ImageDraw.Draw(img)
    for info in mask_infos:
        ys, xs = np.nonzero(info["mask"])
        if ys.size == 0:
            continue
        color = _color_for_goal_id(info["goal_id"])
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        label = f"{info['category']} {info['score']:.2f} {info['goal_id']}"
        text_y = max(y0 - 12, 0)
        draw.rectangle([x0, text_y, x0 + 8 * len(label), text_y + 11], fill=(0, 0, 0))
        draw.text((x0 + 1, text_y), label, fill=color)
    return np.asarray(img, dtype=np.uint8)


def save_gif(frames: list[np.ndarray], out_path: str, fps: float = 2.0) -> None:
    images = [Image.fromarray(f) for f in frames]
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
    )


def build_stock_sim(
    scene_glb: str, width: int = 640, height: int = 480, hfov_deg: float = 90.0
):
    """Minimal habitat_sim.Simulator against a stock example scene: one RGB
    sensor, no depth/semantic (not needed for CLIP calibration)."""
    import habitat_sim
    from habitat_sim.agent import ActionSpec, ActuationSpec, AgentConfiguration

    from sam_vla.env.sim_utils import make_sensor

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_glb

    rgb_spec = make_sensor("rgb", habitat_sim.SensorType.COLOR, height, width, hfov_deg)

    agent_cfg = AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec]
    agent_cfg.action_space = {
        "move_forward": ActionSpec("move_forward", ActuationSpec(amount=0.25)),
        "turn_left": ActionSpec("turn_left", ActuationSpec(amount=15.0)),
        "turn_right": ActionSpec("turn_right", ActuationSpec(amount=15.0)),
    }

    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def capture_frames(sim, script: list[str], out_dir: str = None) -> list[np.ndarray]:
    """Walk a short scripted path (list of action names), capturing the RGB
    frame after each action (plus the initial frame). Returns a list of
    (H, W, 3) uint8 arrays; also writes them to out_dir if given, so a
    separate process/env can pick them up (see module docstring)."""
    obs = sim.get_sensor_observations()
    frames = [np.asarray(obs["rgb"])[:, :, :3].astype(np.uint8)]
    for action in script:
        obs = sim.step(action)
        frames.append(np.asarray(obs["rgb"])[:, :, :3].astype(np.uint8))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(os.path.join(out_dir, f"{i:03d}.jpg"))
    return frames


def load_frames(frames_dir: str) -> list[np.ndarray]:
    paths = sorted(p for p in os.listdir(frames_dir) if p.endswith(".jpg"))
    if not paths:
        raise FileNotFoundError(f"no .jpg frames found in {frames_dir}")
    return [
        np.array(Image.open(os.path.join(frames_dir, p)).convert("RGB")) for p in paths
    ]


def calibrate(
    frames: list[np.ndarray],
    vocab_terms: list[str],
    window_frames: int,
    seg_interval: int,
    clip_reid_thresh: float,
    checkpoint_path: str = None,
) -> list[np.ndarray]:
    """Runs frames through Sam3GoalTracker.resegment (every seg_interval
    frames) + ClipGoalClassifier.classify/match_or_mint, printing per-mask
    category/score/re-ID decisions for manual --clip-goal-thresh /
    --clip-reid-thresh tuning by eye -- same classifier/tracker code as the
    live multi-goal rollout, just against an easier-to-eyeball scene.

    Returns one annotated frame per input frame (mask fill + bbox + label,
    colored per goal_id so re-ID is visible at a glance), holding each goal's
    last-known mask between resegment cycles -- the same "hold between
    cycles" behavior run_navdp_rollout.py uses for the policy's goal_mask
    channel -- so the sequence can be stitched into a GIF/video."""
    from sam_vla.core.types import TrackedGoal
    from sam_vla.perception.clip_goal_classifier import ClipGoalClassifier
    from sam_vla.perception.sam3_goal_tracker import Sam3GoalTracker

    tracker = Sam3GoalTracker(
        vocab_terms=vocab_terms,
        window_frames=window_frames,
        checkpoint_path=checkpoint_path,
    )
    classifier = ClipGoalClassifier(goal_vocabulary=vocab_terms)
    tracked_goals: dict[str, TrackedGoal] = {}
    last_known: dict[str, dict] = {}
    annotated_frames: list[np.ndarray] = []

    for step, frame in enumerate(frames):
        tracker.push_frame(frame)
        if step % seg_interval == 0:
            masks = tracker.resegment(step)
            if not masks:
                print(f"[step {step}] no masks returned")
            else:
                height, width = frame.shape[:2]
                for obj_id, mask in masks.items():
                    ys, xs = np.nonzero(mask)
                    if ys.size == 0:
                        continue
                    crop = frame[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
                    category, score, embedding = classifier.classify(crop)
                    goal_id = classifier.match_or_mint(
                        embedding, tracked_goals, clip_reid_thresh
                    )
                    is_new = goal_id not in tracked_goals
                    if is_new:
                        tracked_goals[goal_id] = TrackedGoal(
                            goal_id=goal_id,
                            category=category,
                            clip_embedding=embedding,
                            clip_score=score,
                            first_seen_step=step,
                            bbox_norm=(
                                float(xs.min()) / width,
                                float(ys.min()) / height,
                                float(xs.max() + 1) / width,
                                float(ys.max() + 1) / height,
                            ),
                        )
                    last_known[goal_id] = {
                        "mask": mask,
                        "category": category,
                        "score": score,
                        "goal_id": goal_id,
                    }
                    print(
                        f"[step {step}] sam3_obj_id={obj_id} area={int(mask.sum())} "
                        f"-> category={category!r} score={score:.3f} goal_id={goal_id} "
                        f"{'(NEW)' if is_new else '(re-ID)'}"
                    )

        annotated_frames.append(_annotate_frame(frame, list(last_known.values())))

    return annotated_frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--scene-glb",
        default=None,
        help="path to a habitat_test_scenes .glb (required unless --load-frames-dir is given)",
    )
    parser.add_argument(
        "--vocab-terms",
        default="chair,door,picture frame,window,table",
        help="comma-separated stock-scene object terms",
    )
    parser.add_argument(
        "--script-length",
        type=int,
        default=12,
        help="number of scripted turn+forward steps to walk",
    )
    parser.add_argument("--window-frames", type=int, default=5)
    parser.add_argument("--seg-interval", type=int, default=3)
    parser.add_argument("--clip-reid-thresh", type=float, default=0.9)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="path to a local SAM3.1 checkpoint (default: download from HF)",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--dump-frames-dir",
        default=None,
        help="write captured frames here for a later --load-frames-dir run in another env",
    )
    parser.add_argument(
        "--load-frames-dir",
        default=None,
        help="skip habitat_sim capture; load already-dumped frames from here instead",
    )
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="only capture+dump frames (no SAM3/CLIP); use in an env that has habitat_sim but not torch",
    )
    parser.add_argument(
        "--gif-path",
        default=None,
        help="stitch the per-goal mask+label annotated frames into an animated GIF at this path",
    )
    parser.add_argument("--gif-fps", type=float, default=2.0)
    parser.add_argument(
        "--annotate-dir",
        default=None,
        help="also dump each annotated frame as a PNG here",
    )
    args = parser.parse_args()

    vocab_terms = [t.strip() for t in args.vocab_terms.split(",") if t.strip()]

    if args.load_frames_dir:
        frames = load_frames(args.load_frames_dir)
        print(f"loaded {len(frames)} frames from {args.load_frames_dir}")
    else:
        if not args.scene_glb:
            parser.error("--scene-glb is required unless --load-frames-dir is given")
        sim = build_stock_sim(args.scene_glb, width=args.width, height=args.height)
        script = (["turn_left", "move_forward"] * args.script_length)[
            : args.script_length
        ]
        frames = capture_frames(sim, script, out_dir=args.dump_frames_dir)
        sim.close()
        print(
            f"captured {len(frames)} frames"
            + (f", dumped to {args.dump_frames_dir}" if args.dump_frames_dir else "")
        )

    if args.capture_only:
        return

    annotated_frames = calibrate(
        frames,
        vocab_terms,
        args.window_frames,
        args.seg_interval,
        args.clip_reid_thresh,
        checkpoint_path=args.checkpoint,
    )

    if args.annotate_dir:
        os.makedirs(args.annotate_dir, exist_ok=True)
        for i, frame in enumerate(annotated_frames):
            Image.fromarray(frame).save(os.path.join(args.annotate_dir, f"{i:03d}.png"))
        print(f"wrote {len(annotated_frames)} annotated frames to {args.annotate_dir}")

    if args.gif_path:
        save_gif(annotated_frames, args.gif_path, fps=args.gif_fps)
        print(f"wrote GIF to {args.gif_path}")


if __name__ == "__main__":
    main()
