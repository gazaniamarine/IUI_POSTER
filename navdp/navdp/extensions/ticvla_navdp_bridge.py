"""Latency-aware TIC-VLA -> NavDP bridge.

This file is intentionally additive: it does not modify NavDP, TIC-VLA, or the
Mars rollout scripts. It implements the TIC-VLA "think in control" pattern as a
wrapper:

    delayed RGB history + instruction --slow async--> TIC-VLA reasoning/waypoints
    latest cached TIC-VLA output + current pose --fast--> ghost goal mask
    RGB-D + ghost mask + proprio --fast--> existing NavDP PolicyRunner

The first bridge mode is waypoint_to_ghost: TIC-VLA predicts robot-frame
waypoints (forward, left), and the bridge converts the first stable waypoint
into a world-anchored ghost goal mask that NavDP already knows how to follow.
"""
from __future__ import annotations

import math
import sys
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from navdp.extensions.ghost_geometry import gc_make_mask, gc_project


@dataclass
class TICVLABridgeConfig:
    """Runtime configuration for the slow TIC-VLA side."""

    ticvla_root: Optional[str] = None
    model_path: str = "InternVL3-1B"
    checkpoint_path: Optional[str] = None
    device: str = "cuda"
    vlm_every: int = 10
    history_frames: int = 4
    image_dir: Optional[str] = None
    mask_radius_px: int = 18
    waypoint_index: int = 0
    max_ghost_distance_m: float = 5.0
    min_ghost_distance_m: float = 0.8
    fallback_distance_m: float = 2.5
    mock: bool = False


@dataclass
class TICVLAState:
    """Latest completed slow-model output."""

    response: str
    prompt: str
    waypoints: np.ndarray
    started_step: int
    completed_step: int
    latency_s: float
    source: str = "ticvla"


@dataclass
class NavDPBridgeOutput:
    """Fast-step conditioning emitted for NavDP."""

    goal_mask: np.ndarray
    ghost_world: Optional[np.ndarray]
    latest_state: Optional[TICVLAState]
    refreshed: bool
    busy: bool


def robot_frame_to_world(
    pose_xyz_yaw: Sequence[float],
    forward_m: float,
    left_m: float,
) -> np.ndarray:
    """Convert robot-frame (forward, left) offset to world XYZ.

    Pose convention: (x, y, z, yaw), matching the Mars/Habitat convention used
    elsewhere in this repo: forward = (-sin(yaw), -cos(yaw)) in XZ.
    """
    x, y, z, yaw = [float(v) for v in pose_xyz_yaw]
    fwd_x, fwd_z = -math.sin(yaw), -math.cos(yaw)
    left_x, left_z = math.cos(yaw), -math.sin(yaw)
    return np.asarray(
        [
            x + forward_m * fwd_x + left_m * left_x,
            y,
            z + forward_m * fwd_z + left_m * left_z,
        ],
        dtype=np.float32,
    )


def world_goal_mask(
    ghost_world: Optional[np.ndarray],
    pose_xyz_yaw: Sequence[float],
    image_shape: Sequence[int],
    intrinsics: dict[str, float],
    radius_px: int,
) -> np.ndarray:
    """Project a world ghost point into the current frame and render a mask."""
    h, w = int(image_shape[0]), int(image_shape[1])
    if ghost_world is None:
        return np.zeros((h, w), dtype=np.uint8)
    pose = np.asarray(pose_xyz_yaw, dtype=np.float32)
    u, v, _z_cam = gc_project(ghost_world, pose[:3], float(pose[3]), intrinsics)
    mask = gc_make_mask(h, w, u, v, float(radius_px))
    return (mask > 0).astype(np.uint8)


class AsyncTICVLANavDPBridge:
    """Run TIC-VLA slowly in the background and expose fast NavDP masks.

    The class can run in mock mode first. In mock mode it returns deterministic
    waypoints from the instruction ("left", "right", otherwise forward), so the
    rest of the bridge can be tested without installing/loading InternVL.
    """

    def __init__(self, config: TICVLABridgeConfig):
        self.cfg = config
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Optional[Future] = None
        self._model = None
        self._latest: Optional[TICVLAState] = None
        self._frame_paths: list[Path] = []
        self._step = 0
        root = Path(config.image_dir) if config.image_dir else Path(tempfile.mkdtemp(prefix="ticvla_navdp_"))
        root.mkdir(parents=True, exist_ok=True)
        self.image_dir = root

    @property
    def latest_state(self) -> Optional[TICVLAState]:
        return self._latest

    def close(self) -> None:
        if self._future is not None and not self._future.done():
            self._future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def step(
        self,
        *,
        rgb: np.ndarray,
        pose_xyz_yaw: Sequence[float],
        intrinsics: dict[str, float],
        instruction: str,
        robot_state: Optional[Sequence[float]] = None,
        time_delay: float = 0.0,
    ) -> NavDPBridgeOutput:
        """Advance bridge one fast step and return the current NavDP goal mask."""
        rgb = np.asarray(rgb, dtype=np.uint8)
        current_path = self._save_frame(rgb, self._step)
        self._frame_paths.append(current_path)
        keep = max(int(self.cfg.history_frames) * max(int(self.cfg.vlm_every), 1), int(self.cfg.history_frames))
        self._frame_paths = self._frame_paths[-keep:]

        refreshed = self._poll_future(completed_step=self._step)
        if self._should_start_slow_step():
            delayed = self._sample_history_paths(current_path)
            self._future = self._executor.submit(
                self._run_slow_model,
                delayed,
                str(current_path),
                instruction,
                None if robot_state is None else list(robot_state),
                int(self._step),
                float(time_delay),
            )

        ghost = self._latest_to_ghost(pose_xyz_yaw)
        mask = world_goal_mask(
            ghost,
            pose_xyz_yaw,
            rgb.shape[:2],
            intrinsics,
            radius_px=int(self.cfg.mask_radius_px),
        )
        out = NavDPBridgeOutput(
            goal_mask=mask,
            ghost_world=ghost,
            latest_state=self._latest,
            refreshed=refreshed,
            busy=self._future is not None and not self._future.done(),
        )
        self._step += 1
        return out

    def _should_start_slow_step(self) -> bool:
        every = max(int(self.cfg.vlm_every), 1)
        return self._step % every == 0 and (self._future is None or self._future.done())

    def _save_frame(self, rgb: np.ndarray, step: int) -> Path:
        from PIL import Image

        path = self.image_dir / f"frame_{step:06d}.jpg"
        Image.fromarray(rgb).save(path, quality=92)
        return path

    def _sample_history_paths(self, current_path: Path) -> list[str]:
        n = max(int(self.cfg.history_frames), 1)
        paths = list(self._frame_paths)
        if not paths:
            paths = [current_path]
        if len(paths) >= n:
            idx = np.linspace(0, len(paths) - 1, n).round().astype(int).tolist()
            sampled = [paths[i] for i in idx]
        else:
            sampled = [paths[0]] * (n - len(paths)) + paths
        sampled[-1] = current_path
        return [str(p) for p in sampled]

    def _poll_future(self, completed_step: int) -> bool:
        if self._future is None or not self._future.done():
            return False
        state = self._future.result()
        self._latest = TICVLAState(
            response=state.response,
            prompt=state.prompt,
            waypoints=state.waypoints,
            started_step=state.started_step,
            completed_step=int(completed_step),
            latency_s=state.latency_s,
            source=state.source,
        )
        self._future = None
        return True

    def _latest_to_ghost(self, pose_xyz_yaw: Sequence[float]) -> Optional[np.ndarray]:
        if self._latest is None or self._latest.waypoints.size == 0:
            return robot_frame_to_world(pose_xyz_yaw, self.cfg.fallback_distance_m, 0.0)
        wp = np.asarray(self._latest.waypoints, dtype=np.float32)
        if wp.ndim == 3:
            wp = wp[0]
        idx = int(np.clip(int(self.cfg.waypoint_index), 0, max(wp.shape[0] - 1, 0)))
        forward = float(wp[idx, 0])
        left = float(wp[idx, 1]) if wp.shape[1] > 1 else 0.0
        dist = math.hypot(forward, left)
        if not np.isfinite(dist) or dist < 1e-6:
            forward, left, dist = self.cfg.fallback_distance_m, 0.0, self.cfg.fallback_distance_m
        if dist > float(self.cfg.max_ghost_distance_m):
            scale = float(self.cfg.max_ghost_distance_m) / dist
            forward *= scale
            left *= scale
        elif dist < float(self.cfg.min_ghost_distance_m):
            scale = float(self.cfg.min_ghost_distance_m) / max(dist, 1e-6)
            forward *= scale
            left *= scale
        return robot_frame_to_world(pose_xyz_yaw, forward, left)

    def _run_slow_model(
        self,
        delayed_image_paths: list[str],
        current_image_path: str,
        instruction: str,
        robot_state: Optional[list[float]],
        started_step: int,
        time_delay: float,
    ) -> TICVLAState:
        t0 = time.perf_counter()
        if self.cfg.mock:
            waypoints = self._mock_waypoints(instruction)
            return TICVLAState(
                response=f"mock waypoint for instruction: {instruction}",
                prompt="mock",
                waypoints=waypoints,
                started_step=int(started_step),
                completed_step=int(started_step),
                latency_s=time.perf_counter() - t0,
                source="mock",
            )

        model = self._ensure_model_loaded()
        import torch

        rs = torch.zeros(5, dtype=torch.float32) if robot_state is None else torch.tensor(robot_state, dtype=torch.float32)
        response, waypoints, prompt = model.predict(
            delayed_image_paths=delayed_image_paths,
            current_image_path=current_image_path,
            instruction=instruction,
            robot_state=rs,
            time_delay=float(time_delay),
        )
        if hasattr(waypoints, "detach"):
            waypoints_np = waypoints.detach().float().cpu().numpy()
        else:
            waypoints_np = np.asarray(waypoints, dtype=np.float32)
        return TICVLAState(
            response=str(response),
            prompt=str(prompt),
            waypoints=waypoints_np.astype(np.float32),
            started_step=int(started_step),
            completed_step=int(started_step),
            latency_s=time.perf_counter() - t0,
            source="ticvla",
        )

    def _ensure_model_loaded(self):
        if self._model is not None:
            return self._model
        if not self.cfg.ticvla_root:
            raise ValueError("ticvla_root is required unless TICVLABridgeConfig.mock=True")
        root = Path(self.cfg.ticvla_root).expanduser().resolve()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from ticvla.models.ticvla import TICVLA

        self._model = TICVLA(model_path=self.cfg.model_path, train_vlm=False)
        self._model.eval()
        if self.cfg.checkpoint_path:
            self._load_checkpoint(self.cfg.checkpoint_path)
        return self._model

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        import torch

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        candidates = []
        candidates.append(state_dict)
        candidates.append({k.removeprefix("model."): v for k, v in state_dict.items()})
        candidates.append({k.removeprefix("module."): v for k, v in state_dict.items()})
        last_error = None
        for sd in candidates:
            try:
                self._model.load_state_dict(sd, strict=False)
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if hasattr(self._model, "load_vlm_checkpoint"):
            self._model.load_vlm_checkpoint(checkpoint_path)
            return
        raise RuntimeError(f"could not load TIC-VLA checkpoint {checkpoint_path}: {last_error}")

    @staticmethod
    def _mock_waypoints(instruction: str) -> np.ndarray:
        text = instruction.lower()
        left = 0.0
        if "left" in text:
            left = 1.5
        elif "right" in text:
            left = -1.5
        elif "avoid" in text or "around" in text:
            left = 1.0
        forward = 2.5 if "stop" not in text else 0.8
        return np.asarray([[[forward, left], [3.5, left], [4.5, 0.5 * left]]], dtype=np.float32)


class TICVLANavDPController:
    """One-object wrapper that turns TIC-VLA thinking into NavDP actions."""

    def __init__(
        self,
        *,
        navdp_ckpt: str,
        bridge_config: TICVLABridgeConfig,
        device: str = "cuda",
        navdp_weights: str = "model",
        sample_steps: int = 20,
        smoothing: str = "ema",
        max_forward_speed: float = 0.6,
        max_lateral_speed: float = 0.0,
        max_yaw_rate: float = 0.7,
    ):
        from navdp.deploy.policy_runner import PolicyRunner

        self.bridge = AsyncTICVLANavDPBridge(bridge_config)
        self.policy = PolicyRunner(
            ckpt_path=navdp_ckpt,
            device=device,
            weights=navdp_weights,
            sample_steps=sample_steps,
            smoothing=smoothing,
            max_forward_speed=max_forward_speed,
            max_lateral_speed=max_lateral_speed,
            max_yaw_rate=max_yaw_rate,
        )

    def reset(self, goal_name: str = "ticvla_goal") -> None:
        self.policy.reset(goal_name)

    def close(self) -> None:
        self.bridge.close()

    def step(
        self,
        *,
        rgb: np.ndarray,
        depth: np.ndarray,
        pose_xzy: Sequence[float],
        intrinsics: dict[str, float],
        instruction: str,
        obstacle_mask: Optional[np.ndarray] = None,
        camera_y: float = 0.0,
        hz: float = 10.0,
    ) -> dict:
        x, z, yaw = [float(v) for v in pose_xzy]
        bridge_out = self.bridge.step(
            rgb=rgb,
            pose_xyz_yaw=(x, camera_y, z, yaw),
            intrinsics=intrinsics,
            instruction=instruction,
        )
        action = self.policy.step(
            depth=depth,
            goal_mask=bridge_out.goal_mask,
            pose=pose_xzy,
            obstacle_mask=obstacle_mask,
            intrinsics=intrinsics,
            hz=hz,
        )
        return {
            "action": action,
            "goal_mask": bridge_out.goal_mask,
            "ghost_world": bridge_out.ghost_world,
            "ticvla_state": bridge_out.latest_state,
            "ticvla_refreshed": bridge_out.refreshed,
            "ticvla_busy": bridge_out.busy,
        }
