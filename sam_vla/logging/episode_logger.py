"""Structured, per-episode JSON logging for ablation rollouts.

One `EpisodeLogger` instance per episode. It knows nothing about ablation
semantics (goal mode, steering mode, CBF, ...) -- it just persists whatever
`config` dict it's constructed with, plus whatever the rollout loop reports
through `log_frame` / `log_qwen_query` / `log_cbf_event`. Keeping it dumb like
this means it doesn't need to change when new ablation flags get added.

Writes are non-blocking: JSON Lines files (frames/qwen_queries/cbf_events)
are appended to by a single background thread pulling off a queue, so
`log_*` calls from the physics/render loop only do cheap bookkeeping (dict
construction + a queue.put) and never touch disk themselves. `config.json`,
`obstacles.json`, and `summary.json` are small, one-shot, pretty-printed
writes done directly since they don't recur every step.
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_SENTINEL = object()


def _json_default(obj: Any) -> Any:
    """Tolerate numpy scalars/arrays showing up in logged payloads -- the
    rollout loop this feeds is numpy-heavy, and a stray np.float32 buried in
    a nested dict/list should not crash logging mid-episode."""
    item = getattr(obj, "item", None)
    if callable(item) and hasattr(obj, "shape") and obj.shape == ():
        return item()
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slugify(value: Any) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", str(value)).strip("-").lower()
    return s or "na"


def make_run_id(config: Dict[str, Any], timestamp: Optional[datetime] = None) -> str:
    """`<timestamp>_<goal_mode>-goal_<steering_mode>_obs<count>_seed<seed>`, so
    the ablation condition is identifiable from the folder name alone."""
    ts = (timestamp or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    goal_mode = _slugify(config.get("goal_mode", "na"))
    steering_mode = _slugify(config.get("steering_mode", "na"))
    obstacle_count = _slugify(config.get("obstacle_count", "na"))
    seed = _slugify(config.get("obstacle_seed", "na"))
    return f"{ts}_{goal_mode}-goal_{steering_mode}_obs{obstacle_count}_seed{seed}"


class EpisodeLogger:
    _JSONL_FILES = ("frames", "qwen_queries", "cbf_events", "segmentation_frames")

    def __init__(
        self,
        run_id: str,
        config: Dict[str, Any],
        log_root: str = "logs",
        save_frames: bool = False,
        flush_interval_s: float = 1.0,
    ):
        self.run_id = run_id
        self.run_dir = Path(log_root) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._save_frames = save_frames
        self.frames_dir = self.run_dir / "frames"
        if save_frames:
            self.frames_dir.mkdir(parents=True, exist_ok=True)

        self._start_monotonic = time.monotonic()
        self._config = dict(config)
        self._config.setdefault("run_id", run_id)
        self._config.setdefault("timestamp_start", _now_iso())
        self._write_json_now("config.json", self._config)

        # Stats accumulated synchronously on the caller's thread (cheap, no I/O)
        # so finalize() can summarize without waiting on the background writer.
        self._total_steps = 0
        self._last_distance_to_goal: Optional[float] = None
        self._closest_approach: Dict[str, float] = {}
        self._counts = {"cbf_events": 0, "qwen_queries": 0, "goal_proximity_events": 0}

        self._files = {
            name: open(self.run_dir / f"{name}.jsonl", "a", encoding="utf-8")
            for name in self._JSONL_FILES
        }
        self._queue: "queue.Queue" = queue.Queue()
        self._flush_interval_s = flush_interval_s
        self._closed = False
        self._worker = threading.Thread(
            target=self._drain_loop, name=f"episode-logger-{run_id}", daemon=True
        )
        self._worker.start()

    # -- internal -----------------------------------------------------------

    def _write_json_now(self, filename: str, obj: Dict[str, Any]) -> None:
        (self.run_dir / filename).write_text(
            json.dumps(obj, indent=2, default=_json_default)
        )

    def _elapsed(self) -> float:
        return time.monotonic() - self._start_monotonic

    def _enqueue(self, kind: str, payload: Dict[str, Any]) -> None:
        self._queue.put((kind, payload))

    def _drain_loop(self) -> None:
        last_flush = time.monotonic()
        while True:
            try:
                item = self._queue.get(timeout=self._flush_interval_s)
            except queue.Empty:
                item = None

            if item is _SENTINEL:
                self._flush_all()
                break
            if item is not None:
                self._write_item(item)

            now = time.monotonic()
            if now - last_flush >= self._flush_interval_s:
                self._flush_all()
                last_flush = now

        for fh in self._files.values():
            fh.close()

    def _write_item(self, item) -> None:
        kind, payload = item
        if kind == "image":
            step, image = payload
            try:
                import imageio.v3 as iio

                iio.imwrite(self.frames_dir / f"frame_{step:06d}.png", image)
            except Exception as e:  # pragma: no cover - best-effort side channel
                print(f"EpisodeLogger: failed to save frame {step}: {e}")
            return
        self._files[kind].write(
            json.dumps(payload, separators=(",", ":"), default=_json_default) + "\n"
        )

    def _flush_all(self) -> None:
        for fh in self._files.values():
            fh.flush()

    # -- public API -----------------------------------------------------------

    def write_obstacles(
        self,
        obstacle_list: Iterable[Dict[str, Any]],
        goal_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        payload = {
            "seed": seed if seed is not None else self._config.get("obstacle_seed"),
            "obstacles": list(obstacle_list),
            "goal_id": goal_id,
        }
        self._write_json_now("obstacles.json", payload)

    def log_frame(
        self,
        step: int,
        position,
        orientation,
        action: Any,
        distances_to_obstacles: Optional[Dict[str, float]] = None,
        cbf_active: bool = False,
        goal_belief: Optional[Dict[str, Any]] = None,
        distance_to_goal: Optional[float] = None,
        yaw_to_goal: Optional[float] = None,
    ) -> None:
        distances_to_obstacles = dict(distances_to_obstacles or {})
        if distance_to_goal is None and goal_belief is not None:
            distance_to_goal = goal_belief.get("range")
        if yaw_to_goal is None and goal_belief is not None:
            yaw_to_goal = goal_belief.get("bearing")

        entry = {
            "step": step,
            "t": self._elapsed(),
            "position": list(position),
            "orientation": list(orientation),
            "action": action,
            "distance_to_goal": distance_to_goal,
            "yaw_to_goal": yaw_to_goal,
            "distances_to_obstacles": distances_to_obstacles,
            "cbf_active": bool(cbf_active),
            "goal_belief": goal_belief,
        }
        self._enqueue("frames", entry)

        self._total_steps = max(self._total_steps, step + 1)
        if distance_to_goal is not None:
            self._last_distance_to_goal = distance_to_goal
        for obs_id, dist in distances_to_obstacles.items():
            prev = self._closest_approach.get(obs_id)
            if prev is None or dist < prev:
                self._closest_approach[obs_id] = dist

    def log_rendered_frame(self, step: int, image) -> None:
        """Save a rendered RGB snapshot (with overlays) to frames/, if the
        logger was constructed with save_frames=True; no-op otherwise."""
        if not self._save_frames:
            return
        self._enqueue("image", (step, image))

    def log_qwen_query(
        self,
        step: int,
        query_type: str,
        trigger: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        latency_ms: float,
    ) -> None:
        entry = {
            "step": step,
            "t": self._elapsed(),
            "query_type": query_type,
            "trigger": trigger,
            "input": input_data,
            "output": output_data,
            "latency_ms": latency_ms,
        }
        self._enqueue("qwen_queries", entry)
        self._counts["qwen_queries"] += 1
        if trigger == "goal_proximity":
            self._counts["goal_proximity_events"] += 1

    def log_cbf_event(
        self,
        step: int,
        obstacle_id: str,
        distance: float,
        nominal_action: Any,
        overridden_action: Any,
        mode: str,
    ) -> None:
        entry = {
            "step": step,
            "t": self._elapsed(),
            "obstacle_id": obstacle_id,
            "distance": distance,
            "nominal_action": nominal_action,
            "overridden_action": overridden_action,
            "mode": mode,
        }
        self._enqueue("cbf_events", entry)
        self._counts["cbf_events"] += 1
        prev = self._closest_approach.get(obstacle_id)
        if prev is None or distance < prev:
            self._closest_approach[obstacle_id] = distance

    def log_segmentation_frame(
        self,
        frame_id: str,
        rgb_path: str,
        instance_mask_path: str,
        category_mask_path: str,
        objects: List[Dict[str, Any]],
        camera_pose: Dict[str, float],
    ) -> None:
        """One next.md Step 4 record per captured frame (mesh_id -> category
        segmentation dataset generation). Mirrors log_cbf_event/log_qwen_query
        exactly: cheap dict construction + enqueue, no disk I/O on this thread
        -- the background drain loop appends to segmentation_frames.jsonl.
        Pixel data (PNGs) must already be written by the caller (see
        perception.segmentation_capture.write_segmentation_assets) before
        calling this -- this method only persists the metadata record,
        keeping it agnostic to how/where the caller chose to save pixels."""
        entry = {
            "frame_id": frame_id,
            "rgb_path": rgb_path,
            "instance_mask_path": instance_mask_path,
            "category_mask_path": category_mask_path,
            "objects": objects,
            "camera_pose": camera_pose,
        }
        self._enqueue("segmentation_frames", entry)

    def finalize(self, summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Write summary.json (caller-provided fields win over computed ones)
        and shut down the background writer. Safe to call at most once;
        __exit__ calls it automatically if the episode ended via exception."""
        computed = {
            "run_id": self.run_id,
            "timestamp_end": _now_iso(),
            "total_steps": self._total_steps,
            "final_distance_to_goal": self._last_distance_to_goal,
            "closest_approach_per_obstacle": dict(self._closest_approach),
            "num_cbf_interventions": self._counts["cbf_events"],
            "num_qwen_queries": self._counts["qwen_queries"],
            "num_goal_proximity_events": self._counts["goal_proximity_events"],
        }
        computed.update(summary or {})
        self._write_json_now("summary.json", computed)
        self.close()
        return computed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(_SENTINEL)
        self._worker.join(timeout=10.0)

    def __enter__(self) -> "EpisodeLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self._closed:
            self.finalize(
                {
                    "success": False,
                    "termination_reason": "exception" if exc_type else "unfinalized",
                }
            )
        return False


if __name__ == "__main__":
    import shutil
    import tempfile

    tmp_root = tempfile.mkdtemp(prefix="episode_logger_demo_")
    try:
        config = {
            "scene_glb": "assets/marsyard2022.glb",
            "goal_mode": "vlm",
            "goal_coord": None,
            "steering_mode": "nudge",
            "cbf_enabled": True,
            "obstacle_count": 5,
            "obstacle_seed": 17,
            "obstacle_distance_threshold_X": 2.0,
            "goal_distance_threshold_Y": 1.5,
            "max_steps": 500,
            "agent_height_offset": 2.0,
        }
        run_id = make_run_id(config)
        print(f"run_id: {run_id}")

        logger = EpisodeLogger(run_id, config, log_root=tmp_root, save_frames=False)

        obstacles = [
            {
                "id": f"rock_{i:02d}",
                "position": [float(i * 2), 0.0, float(i)],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "radius": 0.4,
                "is_goal": (i == 3),
            }
            for i in range(5)
        ]
        logger.write_obstacles(obstacles, goal_id="rock_03")

        for step in range(20):
            dist_to_goal = max(0.1, 10.0 - step * 0.5)
            distances = {
                o["id"]: max(0.2, 5.0 - abs(step - i * 2) * 0.3)
                for i, o in enumerate(obstacles)
            }
            near_obstacle = min(distances.values()) < 2.0
            cbf_active = near_obstacle and step % 3 == 0

            logger.log_frame(
                step=step,
                position=[step * 0.1, 0.0, step * 0.05],
                orientation=[0.0, 0.0, 0.0, 1.0],
                action={"v": 0.3, "w": 0.1 if step % 4 else -0.1},
                distances_to_obstacles=distances,
                cbf_active=cbf_active,
                goal_belief={
                    "range": dist_to_goal,
                    "bearing": 12.5 - step,
                    "source": "observed",
                },
            )

            if cbf_active:
                nearest_id = min(distances, key=distances.get)
                logger.log_cbf_event(
                    step=step,
                    obstacle_id=nearest_id,
                    distance=distances[nearest_id],
                    nominal_action={"v": 0.3, "w": 0.1},
                    overridden_action={"v": 0.1, "w": 0.4},
                    mode="orbit",
                )

            if dist_to_goal < 3.0 and step % 5 == 0:
                t0 = time.monotonic()
                time.sleep(0.001)
                logger.log_qwen_query(
                    step=step,
                    query_type="action_suggestion",
                    trigger="goal_proximity",
                    input_data={
                        "goal_belief": {"range": dist_to_goal, "bearing": 12.5 - step}
                    },
                    output_data={
                        "action": "forward",
                        "reasoning": "goal roughly ahead",
                    },
                    latency_ms=(time.monotonic() - t0) * 1000.0,
                )

        summary = logger.finalize(
            {
                "success": True,
                "termination_reason": "goal_reached",
            }
        )
        print("summary:", json.dumps(summary, indent=2))

        run_dir = Path(tmp_root) / run_id
        print(f"\n{run_dir} contents:")
        for p in sorted(run_dir.iterdir()):
            print(f"  {p.name} ({p.stat().st_size} bytes)")

        assert (run_dir / "config.json").exists()
        assert (run_dir / "obstacles.json").exists()
        assert (run_dir / "summary.json").exists()
        with open(run_dir / "frames.jsonl") as f:
            frame_lines = f.readlines()
        assert (
            len(frame_lines) == 20
        ), f"expected 20 frame lines, got {len(frame_lines)}"
        json.loads(frame_lines[0])  # each line parses independently
        print("\nOK: all files present, frames.jsonl has 20 well-formed lines.")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
