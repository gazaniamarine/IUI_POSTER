"""
Headless capture of a frozen benchmark frame set for the vl_direction model
ablation (see run_ablation.py). Reuses kb_teleop_vl's synthetic obstacle
field, projection, and overlay so every candidate model sees the exact same
scenes kb_teleop_vl.py would show a human operator -- but renders them once,
deterministically, with no Tkinter window and no VL calls, so the same
frame set can be replayed against many model servers without re-running
habitat-sim (and its GPU/rendering cost) for each one.

Deliberately does NOT import kb_teleop_vl's VLTeleopApp (Tkinter-coupled) --
just its module-level obstacle/projection helpers and constants, which are
plain functions/values. The per-obstacle nearest/visible scan below mirrors
VLTeleopApp._project_obstacles exactly (small enough, and tied enough to a
class instance, that importing it isn't simpler than reproducing it).

Run in the "habitat" conda env (needs habitat_sim), from the repo root:
    conda activate habitat && python -m vl_direction.ablation.capture_benchmark_frames
"""

import json
import math
from pathlib import Path

import numpy as np
import quaternion
from PIL import Image

import kb_teleop as kb
import kb_teleop_vl as kbvl

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "benchmark_frames"

# Approach distances (< CBF_DISTANCE_THRESHOLD_M) for the "obstacle visible
# and close" scenarios, and for the "close but facing away" ones that should
# make kb_teleop_vl's real dispatch logic fall back cbf -> exploration.
_CBF_VISIBLE_APPROACH_M = [1.5, 2.5, 3.0]
_CBF_FALLBACK_APPROACH_M = [2.0, 2.5]
_NUM_EXPLORATION_FAR = 3
_NUM_UNCERTAINTY = 2

# Candidate grid to search for "far from every obstacle" rover placements --
# deterministic (fixed obstacle seed + fixed grid), not a random search.
_FAR_CANDIDATE_COORDS = [x for x in range(-20, 21, 5)]


def _project_obstacles(obstacles, x, y, z, yaw):
    """Mirrors kb_teleop_vl.VLTeleopApp._project_obstacles."""
    circles = []
    nearest_any = None
    nearest_visible = None

    for ox, oy, oz in obstacles:
        edge_distance = math.hypot(ox - x, oz - z) - kbvl.OBSTACLE_RADIUS_M
        if nearest_any is None or edge_distance < nearest_any["edge_distance"]:
            nearest_any = {"edge_distance": edge_distance}

        projected = kbvl.project_point(ox, oy, oz, x, y, z, yaw)
        if projected is None:
            continue
        pixel_x, pixel_y, depth = projected
        pixel_radius = float(
            np.clip(
                kbvl._FOCAL_PX * kbvl.OBSTACLE_RADIUS_M / depth,
                kbvl.OVERLAY_MIN_PIXEL_RADIUS,
                kbvl.OVERLAY_MAX_PIXEL_RADIUS,
            )
        )
        x1, y1 = pixel_x - pixel_radius, pixel_y - pixel_radius
        x2, y2 = pixel_x + pixel_radius, pixel_y + pixel_radius
        x1c, y1c = max(0.0, x1), max(0.0, y1)
        x2c, y2c = min(float(kbvl._FRAME_W), x2), min(float(kbvl._FRAME_H), y2)
        if x2c <= x1c or y2c <= y1c:
            continue

        circles.append((pixel_x, pixel_y, pixel_radius))
        bbox = (int(x1c), int(y1c), int(math.ceil(x2c)), int(math.ceil(y2c)))
        if nearest_visible is None or edge_distance < nearest_visible["edge_distance"]:
            nearest_visible = {"edge_distance": edge_distance, "bbox": bbox}

    return circles, nearest_any, nearest_visible


def _make_obstacles():
    rng = np.random.default_rng(kbvl.OBSTACLE_SEED)
    obstacle_xz = kbvl.make_obstacles(
        rng, kbvl.NUM_OBSTACLES, kbvl.OBSTACLE_SPAWN_HALF_EXTENT
    )
    return [
        (ox, kb.terrain_height_at(ox, oz) + kbvl.OBSTACLE_RADIUS_M, oz)
        for ox, oz in obstacle_xz
    ]


def _build_scenarios(obstacles):
    """Returns a list of {label, x, z, yaw_deg} rover placements covering
    cbf-visible, cbf-fallback (near but facing away), and far-from-everything
    (exploration / uncertainty) cases."""
    scenarios = []

    for i, approach in enumerate(_CBF_VISIBLE_APPROACH_M):
        ox, _oy, oz = obstacles[i % len(obstacles)]
        scenarios.append(
            {
                "label": f"cbf_visible_{i}",
                "x": ox,
                "z": oz + kbvl.OBSTACLE_RADIUS_M + approach,
                "yaw_deg": 0.0,  # forward = -Z, faces straight at the obstacle
            }
        )

    for i, approach in enumerate(_CBF_FALLBACK_APPROACH_M):
        ox, _oy, oz = obstacles[(i + len(_CBF_VISIBLE_APPROACH_M)) % len(obstacles)]
        scenarios.append(
            {
                "label": f"cbf_fallback_{i}",
                "x": ox,
                "z": oz + kbvl.OBSTACLE_RADIUS_M + approach,
                "yaw_deg": 180.0,  # forward = +Z, faces away from the obstacle
            }
        )

    # Search the fixed grid for points whose nearest obstacle is well clear
    # of CBF_DISTANCE_THRESHOLD_M -- these become the exploration/uncertainty
    # scenarios (no visual anchor nearby).
    far_candidates = []
    for x in _FAR_CANDIDATE_COORDS:
        for z in _FAR_CANDIDATE_COORDS:
            if abs(x) > kb.BOUNDARY_LIMIT - 1 or abs(z) > kb.BOUNDARY_LIMIT - 1:
                continue
            _circles, nearest_any, _nearest_visible = _project_obstacles(
                obstacles, x, kb.terrain_height_at(x, z) + kb.INITIAL_CLEARANCE, z, 0.0
            )
            edge_distance = (
                nearest_any["edge_distance"] if nearest_any else float("inf")
            )
            far_candidates.append((edge_distance, x, z))
    far_candidates.sort(key=lambda t: -t[0])

    needed_far = _NUM_EXPLORATION_FAR + _NUM_UNCERTAINTY
    chosen_far = far_candidates[:needed_far]
    if (
        len(chosen_far) < needed_far
        or chosen_far[-1][0] <= kbvl.CBF_DISTANCE_THRESHOLD_M
    ):
        raise RuntimeError(
            f"could not find {needed_far} rover placements clear of every obstacle "
            f"(best candidates: {far_candidates[:needed_far]}); widen _FAR_CANDIDATE_COORDS"
        )

    # Face the terrain center from these far-out placements -- a fixed
    # arbitrary yaw risks pointing past the finite terrain mesh's edge into
    # the unrendered black void beyond it (caught by inspecting a captured
    # frame: uncertainty_1 came back solid black before this fix).
    for i in range(_NUM_EXPLORATION_FAR):
        _edge, x, z = chosen_far[i]
        yaw_deg = math.degrees(math.atan2(x, z))
        scenarios.append(
            {"label": f"exploration_far_{i}", "x": x, "z": z, "yaw_deg": yaw_deg}
        )

    for i in range(_NUM_UNCERTAINTY):
        _edge, x, z = chosen_far[_NUM_EXPLORATION_FAR + i]
        yaw_deg = math.degrees(math.atan2(x, z))
        scenarios.append(
            {"label": f"uncertainty_{i}", "x": x, "z": z, "yaw_deg": yaw_deg}
        )

    return scenarios


def capture():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sim = kb.make_sim()
    agent = sim.initialize_agent(0)

    obstacles = _make_obstacles()
    print(
        f"[capture_benchmark_frames] {len(obstacles)} obstacles: "
        + ", ".join(f"({o[0]:.1f},{o[2]:.1f})" for o in obstacles)
    )

    scenarios = _build_scenarios(obstacles)
    print(
        f"[capture_benchmark_frames] {len(scenarios)} scenarios: "
        + ", ".join(s["label"] for s in scenarios)
    )

    manifest = []
    for scenario in scenarios:
        x, z = scenario["x"], scenario["z"]
        yaw = math.radians(scenario["yaw_deg"])
        y = kb.terrain_height_at(x, z) + kb.INITIAL_CLEARANCE

        state = agent.get_state()
        state.position = np.array([x, y, z], dtype=np.float32)
        state.rotation = quaternion.from_rotation_vector([0.0, yaw, 0.0])
        agent.set_state(state)

        obs = sim.get_sensor_observations()
        rgb, _depth_vis, _depth_rgb = kb.rgb_depth_from_obs(obs)

        circles, nearest_any, nearest_visible = _project_obstacles(
            obstacles, x, y, z, yaw
        )
        annotated_rgb = kbvl.overlay_obstacles(rgb, circles)

        frame_path = OUT_DIR / f"{scenario['label']}.png"
        Image.fromarray(annotated_rgb).save(frame_path)

        manifest.append(
            {
                "label": scenario["label"],
                "frame_file": frame_path.name,
                "pose": {"x": x, "y": y, "z": z, "yaw_deg": scenario["yaw_deg"]},
                "nearest_any_edge_distance_m": (
                    nearest_any["edge_distance"] if nearest_any else None
                ),
                "nearest_visible_bbox_xyxy": (
                    nearest_visible["bbox"] if nearest_visible else None
                ),
                "frame_wh": [kbvl._FRAME_W, kbvl._FRAME_H],
            }
        )
        print(
            f"  {scenario['label']}: nearest_any={manifest[-1]['nearest_any_edge_distance_m']}, "
            f"nearest_visible_bbox={manifest[-1]['nearest_visible_bbox_xyxy']}"
        )

    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    sim.close()
    print(
        f"[capture_benchmark_frames] wrote {len(manifest)} frames + manifest to {OUT_DIR}"
    )


if __name__ == "__main__":
    capture()
