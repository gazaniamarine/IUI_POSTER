# from __future__ import annotations

# import argparse
# import io
# import json
# import math
# import os
# import socket
# import subprocess
# import sys
# import time
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Any, Optional, Sequence

# import habitat_sim
# import numpy as np
# import quaternion
# import requests
# from habitat_sim.agent import AgentConfiguration
# from PIL import Image, ImageDraw

# from belief_heading_recovery import belief_heading_recovery_action
# from belief_pixel_goal import GaussianGoalBelief


# HERE = Path(__file__).resolve().parent
# SIZE_X = 50.0
# SIZE_Z = 50.0
# SIZE_Y = 4.820803273566
# MESH_GOAL_ID = 10000
# MESH_OBSTACLE_ID = 2


# @dataclass(frozen=True)
# class NavDPS2DiffOutput:
#     trajectory: np.ndarray
#     all_trajectories: np.ndarray
#     all_values: np.ndarray
#     selected_index: int
#     fallback_stop: bool
#     escape_turn: bool
#     valid_obstacle_points: int
#     selected_circulation_sign: float
#     candidate_circulation_signs: np.ndarray
#     selected_barrier_energy: float
#     selected_circulation_energy: float
#     minimum_clearance: np.ndarray
#     selected_minimum_clearance: float
#     mean_guidance_noise_correction: float
#     final_guidance_noise_correction: float
#     maximum_guidance_noise_correction: float
#     mean_final_effective_sample_size: float


# class NavDPS2DiffClient:
#     def __init__(self, server_url: str, timeout: float = 180.0):
#         self.server_url = server_url.rstrip("/")
#         self.timeout = float(timeout)

#     def reset(
#         self,
#         intrinsic: np.ndarray,
#         *,
#         stop_threshold: float = -3.0,
#         batch_size: int = 1,
#     ) -> str:
#         intrinsic = np.asarray(intrinsic, dtype=np.float32)
#         if intrinsic.shape != (3, 3):
#             raise ValueError(f"intrinsic must have shape [3,3], got {intrinsic.shape}")
#         response = requests.post(
#             f"{self.server_url}/navigator_reset",
#             json={
#                 "intrinsic": intrinsic.tolist(),
#                 "stop_threshold": float(stop_threshold),
#                 "batch_size": int(batch_size),
#             },
#             timeout=self.timeout,
#         )
#         self._raise_for_error(response)
#         return str(response.json().get("algo", ""))

#     def plan(
#         self,
#         *,
#         goal_xy: np.ndarray,
#         rgb: np.ndarray,
#         depth: np.ndarray,
#         obstacle_pixels: np.ndarray,
#         goal_mode: str = "point",
#         forced_circulation_sign: float = 0.0,
#     ) -> NavDPS2DiffOutput:
#         goal_xy = np.asarray(goal_xy, dtype=np.float32).reshape(-1)
#         if goal_xy.shape != (2,):
#             raise ValueError(f"goal_xy must have shape [2], got {goal_xy.shape}")
#         if goal_mode not in {"point", "pixel"}:
#             raise ValueError("goal_mode must be point or pixel")
#         forced_circulation_sign = float(forced_circulation_sign)
#         if forced_circulation_sign not in {-1.0, 0.0, 1.0}:
#             raise ValueError("forced_circulation_sign must be -1, 0, or +1")

#         rgb = np.asarray(rgb, dtype=np.uint8)
#         if rgb.ndim != 3 or rgb.shape[-1] < 3:
#             raise ValueError(f"rgb must have shape [H,W,3], got {rgb.shape}")
#         rgb = rgb[..., :3]

#         depth = np.asarray(depth, dtype=np.float32)
#         if depth.ndim == 3 and depth.shape[-1] == 1:
#             depth = depth[..., 0]
#         if depth.shape != rgb.shape[:2]:
#             raise ValueError(f"depth/rgb shape mismatch: {depth.shape} vs {rgb.shape[:2]}")

#         if goal_mode == "pixel":
#             if not np.all(np.isfinite(goal_xy)) or not np.allclose(
#                 goal_xy, np.round(goal_xy)
#             ):
#                 raise ValueError("PixelGoal must be integer [u,v]")
#             goal_xy = np.round(goal_xy).astype(np.int64)
#             if not (0 <= goal_xy[0] < rgb.shape[1] and 0 <= goal_xy[1] < rgb.shape[0]):
#                 raise ValueError("PixelGoal lies outside the RGB image")

#         pixels = np.asarray(obstacle_pixels)
#         if pixels.size == 0:
#             pixels = np.zeros((0, 2), dtype=np.int32)
#         else:
#             pixels = pixels.reshape(-1, 2)
#             if not np.all(np.isfinite(pixels)):
#                 raise ValueError("obstacle pixels must be finite")
#             if not np.allclose(pixels, np.round(pixels)):
#                 raise ValueError("obstacle pixels must be integer [u,v] coordinates")
#             pixels = np.round(pixels).astype(np.int32)

#         rgb_bytes = io.BytesIO()
#         Image.fromarray(rgb, mode="RGB").save(rgb_bytes, format="JPEG", quality=95)
#         depth_u16 = np.clip(depth * 10000.0, 0.0, 65535.0).astype(np.uint16)
#         depth_bytes = io.BytesIO()
#         Image.fromarray(depth_u16).save(depth_bytes, format="PNG")

#         endpoint = "pixelgoal_step" if goal_mode == "pixel" else "pointgoal_step"
#         response = requests.post(
#             f"{self.server_url}/{endpoint}",
#             files={
#                 "image": ("image.jpg", rgb_bytes.getvalue(), "image/jpeg"),
#                 "depth": ("depth.png", depth_bytes.getvalue(), "image/png"),
#             },
#             data={
#                 "goal_data": json.dumps(
#                     {
#                         "goal_x": [float(goal_xy[0])],
#                         "goal_y": [float(goal_xy[1])],
#                         "obstacle_pixels": [pixels.tolist()],
#                         "forced_circulation_signs": [forced_circulation_sign],
#                     }
#                 )
#             },
#             timeout=self.timeout,
#         )
#         self._raise_for_error(response)
#         payload = response.json()
#         diagnostics = payload["s2diff"]
#         trajectory = np.asarray(payload["trajectory"], dtype=np.float32)
#         all_trajectories = np.asarray(payload["all_trajectory"], dtype=np.float32)
#         all_values = np.asarray(payload["all_values"], dtype=np.float32)

#         return NavDPS2DiffOutput(
#             trajectory=trajectory[0],
#             all_trajectories=all_trajectories[0],
#             all_values=all_values[0],
#             selected_index=int(diagnostics["selected_index"][0]),
#             fallback_stop=bool(diagnostics["fallback_stop"][0]),
#             escape_turn=bool(diagnostics["escape_turn"][0]),
#             valid_obstacle_points=int(diagnostics["valid_obstacle_points"][0]),
#             selected_circulation_sign=float(
#                 diagnostics["selected_circulation_sign"][0]
#             ),
#             candidate_circulation_signs=np.asarray(
#                 diagnostics["candidate_circulation_signs"][0], dtype=np.float32
#             ),
#             selected_barrier_energy=float(diagnostics["selected_barrier_energy"][0]),
#             selected_circulation_energy=float(
#                 diagnostics["selected_circulation_energy"][0]
#             ),
#             minimum_clearance=np.asarray(
#                 diagnostics["minimum_clearance"][0], dtype=np.float32
#             ),
#             selected_minimum_clearance=float(
#                 diagnostics["selected_minimum_clearance"][0]
#             ),
#             mean_guidance_noise_correction=float(
#                 diagnostics["mean_guidance_noise_correction"][0]
#             ),
#             final_guidance_noise_correction=float(
#                 diagnostics["final_guidance_noise_correction"][0]
#             ),
#             maximum_guidance_noise_correction=float(
#                 diagnostics["maximum_guidance_noise_correction"][0]
#             ),
#             mean_final_effective_sample_size=float(
#                 diagnostics.get("mean_final_effective_sample_size", [0.0])[0]
#             ),
#         )

#     @staticmethod
#     def _raise_for_error(response: requests.Response) -> None:
#         try:
#             payload = response.json()
#         except ValueError:
#             payload = None
#         if isinstance(payload, dict) and "error" in payload:
#             raise RuntimeError(str(payload["error"]))
#         response.raise_for_status()


# @dataclass(frozen=True)
# class QwenHomotopyDecision:
#     side: str
#     circulation_sign: float
#     confidence: float
#     obstacle_relevant: bool
#     queried_qwen: bool
#     raw_response: Optional[str]
#     repeated_sides: tuple[str, ...]
#     repeated_confidences: tuple[float, ...]
#     consistency_rate: float
#     used_fallback: bool


# class QwenHomotopyClient:
#     """HTTP client for the isolated visual-Qwen process."""

#     def __init__(self, server_url: str, timeout: float = 300.0) -> None:
#         self.server_url = server_url.rstrip("/")
#         self.timeout = float(timeout)

#     def reset(self) -> None:
#         response = requests.post(f"{self.server_url}/reset", timeout=self.timeout)
#         self._raise_for_error(response)

#     def step(
#         self, overlaid_rgb: np.ndarray, obstacle_mask: np.ndarray
#     ) -> QwenHomotopyDecision:
#         image_bytes = io.BytesIO()
#         Image.fromarray(np.asarray(overlaid_rgb, dtype=np.uint8)).save(
#             image_bytes, format="PNG"
#         )
#         mask_bytes = io.BytesIO()
#         Image.fromarray(
#             (np.asarray(obstacle_mask) > 0).astype(np.uint8) * 255
#         ).save(mask_bytes, format="PNG")
#         response = requests.post(
#             f"{self.server_url}/select",
#             files={
#                 "image": ("overlay.png", image_bytes.getvalue(), "image/png"),
#                 "obstacle_mask": ("mask.png", mask_bytes.getvalue(), "image/png"),
#             },
#             timeout=self.timeout,
#         )
#         self._raise_for_error(response)
#         payload = response.json()
#         return QwenHomotopyDecision(
#             side=str(payload["side"]),
#             circulation_sign=float(payload["circulation_sign"]),
#             confidence=float(payload["confidence"]),
#             obstacle_relevant=bool(payload["obstacle_relevant"]),
#             queried_qwen=bool(payload["queried_qwen"]),
#             raw_response=payload.get("raw_response"),
#             repeated_sides=tuple(payload.get("repeated_sides", [])),
#             repeated_confidences=tuple(
#                 float(value) for value in payload.get("repeated_confidences", [])
#             ),
#             consistency_rate=float(payload.get("consistency_rate", 1.0)),
#             used_fallback=bool(payload.get("used_fallback", False)),
#         )

#     @staticmethod
#     def _raise_for_error(response: requests.Response) -> None:
#         try:
#             payload = response.json()
#         except ValueError:
#             payload = None
#         if isinstance(payload, dict) and "error" in payload:
#             raise RuntimeError(str(payload["error"]))
#         response.raise_for_status()

# def port_is_open(host: str, port: int) -> bool:
#     try:
#         with socket.create_connection((host, port), timeout=1.0):
#             return True
#     except OSError:
#         return False


# def wait_for_server(
#     process: subprocess.Popen[Any], host: str, port: int, timeout: float
# ) -> None:
#     deadline = time.time() + float(timeout)
#     while time.time() < deadline:
#         if process.poll() is not None:
#             raise RuntimeError(
#                 f"NavDP/S2Diff server exited with code {process.returncode}"
#             )
#         if port_is_open(host, port):
#             return
#         time.sleep(1.0)
#     raise TimeoutError(f"NavDP server did not open port {port} within {timeout}s")


# def stop_server(process: Optional[subprocess.Popen[Any]]) -> None:
#     if process is None or process.poll() is not None:
#         return
#     process.terminate()
#     try:
#         process.wait(timeout=10.0)
#     except subprocess.TimeoutExpired:
#         process.kill()
#         process.wait()


# def start_qwen_homotopy_server(
#     args: argparse.Namespace,
# ) -> Optional[subprocess.Popen[Any]]:
#     if not args.qwen_homotopy or not args.start_qwen_homotopy_server:
#         return None
#     if port_is_open(args.qwen_homotopy_host, args.qwen_homotopy_port):
#         raise RuntimeError(
#             f"Qwen homotopy port {args.qwen_homotopy_port} is already in use; "
#             "pass --no-start-qwen-homotopy-server to use an existing service"
#         )
#     server_file = HERE / "qwen_homotopy_server.py"
#     if not server_file.is_file():
#         raise FileNotFoundError(f"Qwen homotopy server not found: {server_file}")
#     command = [
#         str(args.qwen_homotopy_python),
#         str(server_file),
#         "--host",
#         str(args.qwen_homotopy_host),
#         "--port",
#         str(args.qwen_homotopy_port),
#         "--model-id",
#         str(args.qwen_model_id),
#         "--device",
#         str(args.qwen_device),
#         "--minimum-obstacle-pixels",
#         str(args.homotopy_minimum_obstacle_pixels),
#         "--release-clear-frames",
#         str(args.homotopy_release_clear_frames),
#         "--consistency-repeats",
#         str(args.homotopy_consistency_repeats),
#     ]
#     print("[qwen-server]", " ".join(command), flush=True)
#     process = subprocess.Popen(command, cwd=str(HERE))
#     wait_for_server(
#         process,
#         args.qwen_homotopy_host,
#         args.qwen_homotopy_port,
#         args.qwen_homotopy_timeout,
#     )
#     return process

# def start_server(args: argparse.Namespace) -> Optional[subprocess.Popen[Any]]:
#     if not args.start_server:
#         return None
#     if port_is_open(args.server_host, args.server_port):
#         raise RuntimeError(
#             f"port {args.server_port} is already in use; use --no-start-server "
#             "to connect to an existing guided server"
#         )

#     navdp_root = Path(args.navdp_root).expanduser().resolve()
#     checkpoint = Path(args.navdp_checkpoint).expanduser().resolve()
#     server_dir = navdp_root / "baselines" / "navdp"
#     server_file = server_dir / "navdp_s2diff_server.py"
#     if not server_file.is_file():
#         raise FileNotFoundError(f"guided server not found: {server_file}")
#     if not checkpoint.is_file():
#         raise FileNotFoundError(f"NavDP checkpoint not found: {checkpoint}")

#     command = [
#         str(args.navdp_python),
#         str(server_file),
#         "--checkpoint",
#         str(checkpoint),
#         "--device",
#         str(args.navdp_device),
#         "--planner-mode",
#         str(args.planner_mode),
#         "--seed",
#         str(args.seed),
#         "--port",
#         str(args.server_port),
#         "--candidates",
#         str(args.candidates),
#         "--particles",
#         str(args.particles),
#         "--particle-std",
#         str(args.particle_std),
#         "--gradient-steps",
#         str(args.gradient_steps),
#         "--gradient-step-size",
#         str(args.gradient_step_size),
#         "--guidance-strength",
#         str(args.guidance_strength),
#         "--temperature",
#         str(args.temperature),
#         "--safe-distance",
#         str(args.safe_distance),
#         "--hard-collision-distance",
#         str(args.hard_collision_distance),
#         "--robot-radius",
#         str(args.robot_radius),
#         "--safety-weight",
#         str(args.safety_weight),
#         "--barrier-weight",
#         str(args.barrier_weight),
#         "--barrier-rate",
#         str(args.barrier_rate),
#         "--circulation-weight",
#         str(args.circulation_weight),
#         "--circulation-activation-distance",
#         str(args.circulation_activation_distance),
#         "--circulation-activation-sharpness",
#         str(args.circulation_activation_sharpness),
#         "--minimum-circulation-progress",
#         str(args.minimum_circulation_progress),
#         "--blocking-alignment-threshold",
#         str(args.blocking_alignment_threshold),
#         "--circulation-switch-weight",
#         str(args.circulation_switch_weight),
#         "--escape-lateral-target",
#         str(args.escape_lateral_target),
#         "--minimum-obstacle-depth",
#         str(args.minimum_obstacle_depth),
#         "--maximum-obstacle-depth",
#         str(args.maximum_obstacle_depth),
#         "--maximum-obstacle-pixels",
#         str(args.maximum_obstacle_pixels),
#     ]
#     particle_flags = {
#         "particle-anchor": args.particle_anchor,
#         "particle-energy-reweighting": args.particle_energy_reweighting,
#         "particle-collision-mask": args.particle_collision_mask,
#         "particle-noise-schedule": args.particle_noise_schedule,
#         "progressive-guidance": args.progressive_guidance,
#     }
#     for name, enabled in particle_flags.items():
#         command.append(f"--{name}" if enabled else f"--no-{name}")
#     command.append("--remove-critic" if args.remove_critic else "--no-remove-critic")
#     print("[server]", " ".join(command), flush=True)
#     process = subprocess.Popen(command, cwd=str(server_dir))
#     wait_for_server(process, args.server_host, args.server_port, args.server_timeout)
#     return process


# def bilinear_grid(grid: np.ndarray, px: float, py: float) -> float:
#     height, width = grid.shape
#     x0 = int(np.floor(px))
#     y0 = int(np.floor(py))
#     x1 = min(x0 + 1, width - 1)
#     y1 = min(y0 + 1, height - 1)
#     tx = px - x0
#     ty = py - y0
#     top = float(grid[y0, x0]) * (1.0 - tx) + float(grid[y0, x1]) * tx
#     bottom = float(grid[y1, x0]) * (1.0 - tx) + float(grid[y1, x1]) * tx
#     return top * (1.0 - ty) + bottom * ty


# class TerrainHeight:
#     def __init__(
#         self,
#         *,
#         mode: str,
#         heightmap: Optional[Path],
#         obj: Optional[Path],
#         flat_y: float,
#         size_x: float,
#         size_z: float,
#         size_y: float,
#         flip_x: bool,
#         flip_z: bool,
#         swap_xz: bool,
#     ):
#         if mode == "auto":
#             mode = "heightmap" if heightmap and heightmap.exists() else (
#                 "obj" if obj and obj.exists() else "flat"
#             )
#         self.mode = mode
#         self.flat_y = float(flat_y)
#         self.size_x = float(size_x)
#         self.size_z = float(size_z)
#         self.size_y = float(size_y)
#         self.flip_x = bool(flip_x)
#         self.flip_z = bool(flip_z)
#         self.swap_xz = bool(swap_xz)
#         self.height: Optional[np.ndarray] = None
#         self.obj_xs: Optional[np.ndarray] = None
#         self.obj_zs: Optional[np.ndarray] = None
#         self.obj_h: Optional[np.ndarray] = None

#         if mode == "heightmap":
#             if heightmap is None or not heightmap.exists():
#                 raise FileNotFoundError(f"heightmap not found: {heightmap}")
#             array = np.asarray(Image.open(heightmap))
#             if array.ndim == 3:
#                 array = array[..., 0]
#             array = array.astype(np.float32)
#             array = (array - array.min()) / max(float(array.max() - array.min()), 1e-8)
#             self.height = array * self.size_y - float(np.mean(array * self.size_y))
#         elif mode == "obj":
#             if obj is None or not obj.exists():
#                 raise FileNotFoundError(f"terrain OBJ not found: {obj}")
#             vertices = []
#             with obj.open("r", encoding="utf-8", errors="ignore") as file:
#                 for line in file:
#                     if line.startswith("v "):
#                         parts = line.split()
#                         if len(parts) >= 4:
#                             vertices.append(tuple(float(value) for value in parts[1:4]))
#             if not vertices:
#                 raise RuntimeError(f"no vertices found in {obj}")
#             array = np.asarray(vertices, dtype=np.float32)
#             xs = np.unique(array[:, 0])
#             zs = np.unique(array[:, 1])
#             grid = np.full((len(zs), len(xs)), np.nan, dtype=np.float32)
#             x_index = {float(value): index for index, value in enumerate(xs.tolist())}
#             z_index = {float(value): index for index, value in enumerate(zs.tolist())}
#             for x, z, height in array:
#                 grid[z_index[float(z)], x_index[float(x)]] = height
#             self.obj_xs = xs
#             self.obj_zs = zs
#             self.obj_h = np.nan_to_num(grid, nan=float(np.nanmean(grid)))
#         elif mode != "flat":
#             raise ValueError(f"unknown terrain mode: {mode}")

#     def _map(self, x: float, z: float) -> tuple[float, float]:
#         if self.swap_xz:
#             x, z = z, x
#         u = (x + self.size_x / 2.0) / self.size_x
#         v = (z + self.size_z / 2.0) / self.size_z
#         if self.flip_x:
#             u = 1.0 - u
#         if self.flip_z:
#             v = 1.0 - v
#         return float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))

#     def __call__(self, x: float, z: float) -> float:
#         if self.mode == "flat":
#             return self.flat_y
#         if self.mode == "heightmap":
#             assert self.height is not None
#             u, v = self._map(x, z)
#             return bilinear_grid(
#                 self.height, u * (self.height.shape[1] - 1), v * (self.height.shape[0] - 1)
#             )
#         assert self.obj_xs is not None and self.obj_zs is not None and self.obj_h is not None
#         xx = float(np.clip(x, self.obj_xs[0], self.obj_xs[-1]))
#         zz = float(np.clip(z, self.obj_zs[0], self.obj_zs[-1]))
#         column = int(np.clip(np.searchsorted(self.obj_xs, xx) - 1, 0, len(self.obj_xs) - 2))
#         row = int(np.clip(np.searchsorted(self.obj_zs, zz) - 1, 0, len(self.obj_zs) - 2))
#         x0, x1 = float(self.obj_xs[column]), float(self.obj_xs[column + 1])
#         z0, z1 = float(self.obj_zs[row]), float(self.obj_zs[row + 1])
#         tx = 0.0 if abs(x1 - x0) < 1e-8 else (xx - x0) / (x1 - x0)
#         tz = 0.0 if abs(z1 - z0) < 1e-8 else (zz - z0) / (z1 - z0)
#         top = float(self.obj_h[row, column]) * (1.0 - tx) + float(self.obj_h[row, column + 1]) * tx
#         bottom = float(self.obj_h[row + 1, column]) * (1.0 - tx) + float(self.obj_h[row + 1, column + 1]) * tx
#         return top * (1.0 - tz) + bottom * tz

#     def local_height_max(self, x: float, z: float, radius: float, samples: int = 5) -> float:
#         if radius <= 1e-6:
#             return float(self(x, z))
#         values = [
#             float(self(x + dx, z + dz))
#             for dx in np.linspace(-radius, radius, samples)
#             for dz in np.linspace(-radius, radius, samples)
#             if dx * dx + dz * dz <= radius * radius + 1e-8
#         ]
#         return max(values) if values else float(self(x, z))


# def make_sensor(
#     uuid: str, sensor_type: Any, height: int, width: int, hfov_deg: float
# ) -> habitat_sim.CameraSensorSpec:
#     specification = habitat_sim.CameraSensorSpec()
#     specification.uuid = uuid
#     specification.sensor_type = sensor_type
#     specification.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
#     specification.resolution = [int(height), int(width)]
#     specification.position = [0.0, 0.0, 0.0]
#     specification.hfov = float(hfov_deg)
#     return specification


# def make_simulator(
#     scene: Path,
#     height: int,
#     width: int,
#     hfov_deg: float,
#     *,
#     with_semantic: bool,
# ):
#     simulator_configuration = habitat_sim.SimulatorConfiguration()
#     simulator_configuration.scene_id = str(scene.expanduser().resolve())
#     simulator_configuration.enable_physics = False
#     sensors = [
#         make_sensor("rgb", habitat_sim.SensorType.COLOR, height, width, hfov_deg),
#         make_sensor("depth", habitat_sim.SensorType.DEPTH, height, width, hfov_deg),
#     ]
#     if with_semantic:
#         sensors.append(
#             make_sensor(
#                 "semantic", habitat_sim.SensorType.SEMANTIC, height, width, hfov_deg
#             )
#         )
#     agent_configuration = AgentConfiguration()
#     agent_configuration.sensor_specifications = sensors
#     return habitat_sim.Simulator(
#         habitat_sim.Configuration(simulator_configuration, [agent_configuration])
#     )

# def set_agent_pose(agent: Any, position: np.ndarray, yaw: float) -> None:
#     state = agent.get_state()
#     state.position = np.asarray(position, dtype=np.float32)
#     state.rotation = quaternion.from_rotation_vector(np.asarray([0.0, yaw, 0.0]))
#     agent.set_state(state)


# def rgb_depth(observation: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
#     rgb = np.asarray(observation["rgb"])
#     if rgb.ndim == 3 and rgb.shape[-1] == 4:
#         rgb = rgb[..., :3]
#     depth = np.asarray(observation["depth"], dtype=np.float32)
#     if depth.ndim == 3:
#         depth = depth[..., 0]
#     return rgb.astype(np.uint8), depth.astype(np.float32)


# def semantic_from_observation(observation: dict[str, np.ndarray]) -> np.ndarray:
#     semantic = np.asarray(observation["semantic"])
#     if semantic.ndim == 3:
#         semantic = semantic[..., 0]
#     return semantic.astype(np.int32)


# def pixel_to_world(
#     u: float,
#     v: float,
#     depth: float,
#     position: np.ndarray,
#     yaw: float,
#     intrinsic: np.ndarray,
# ) -> np.ndarray:
#     right = (u - float(intrinsic[0, 2])) * depth / float(intrinsic[0, 0])
#     up = -(v - float(intrinsic[1, 2])) * depth / float(intrinsic[1, 1])
#     forward_vector = np.asarray([-math.sin(yaw), 0.0, -math.cos(yaw)])
#     right_vector = np.asarray([math.cos(yaw), 0.0, -math.sin(yaw)])
#     return (
#         np.asarray(position, dtype=np.float64)
#         + depth * forward_vector
#         + right * right_vector
#         + up * np.asarray([0.0, 1.0, 0.0])
#     )


# def depth_patch_mesh(
#     u_center: float,
#     v_center: float,
#     half_size: int,
#     stride: int,
#     depth: np.ndarray,
#     position: np.ndarray,
#     yaw: float,
#     intrinsic: np.ndarray,
#     *,
#     lift: float,
#     maximum_depth_jump: float = 0.4,
# ) -> tuple[np.ndarray, np.ndarray]:
#     height, width = depth.shape
#     columns = list(
#         range(
#             max(0, int(u_center - half_size)),
#             min(width, int(u_center + half_size) + 1),
#             max(int(stride), 1),
#         )
#     )
#     rows = list(
#         range(
#             max(0, int(v_center - half_size)),
#             min(height, int(v_center + half_size) + 1),
#             max(int(stride), 1),
#         )
#     )
#     indices = -np.ones((len(rows), len(columns)), dtype=np.int64)
#     depths = np.full((len(rows), len(columns)), np.nan, dtype=np.float32)
#     vertices: list[tuple[float, float, float]] = []
#     for row_index, v in enumerate(rows):
#         for column_index, u in enumerate(columns):
#             metric_depth = float(depth[v, u])
#             if not np.isfinite(metric_depth) or metric_depth <= 0.1:
#                 continue
#             indices[row_index, column_index] = len(vertices)
#             depths[row_index, column_index] = metric_depth
#             point = pixel_to_world(
#                 u, v, metric_depth, position, yaw, intrinsic
#             ) + float(lift) * np.asarray([0.0, 1.0, 0.0])
#             vertices.append(tuple(float(value) for value in point))

#     faces: list[tuple[int, int, int]] = []
#     for row_index in range(len(rows) - 1):
#         for column_index in range(len(columns) - 1):
#             a = int(indices[row_index, column_index])
#             b = int(indices[row_index, column_index + 1])
#             c = int(indices[row_index + 1, column_index])
#             d = int(indices[row_index + 1, column_index + 1])
#             if min(a, b, c, d) < 0:
#                 continue
#             cell_depths = (
#                 depths[row_index, column_index],
#                 depths[row_index, column_index + 1],
#                 depths[row_index + 1, column_index],
#                 depths[row_index + 1, column_index + 1],
#             )
#             if max(cell_depths) - min(cell_depths) > maximum_depth_jump:
#                 continue
#             faces.append((a, c, d))
#             faces.append((a, d, b))
#     return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


# def save_obj(
#     path: Path,
#     vertices: np.ndarray,
#     faces: np.ndarray,
#     *,
#     diffuse_rgb: Optional[tuple[float, float, float]] = None,
# ) -> None:
#     material_name = None
#     if diffuse_rgb is not None:
#         red, green, blue = (float(value) for value in diffuse_rgb)
#         if not all(0.0 <= value <= 1.0 for value in (red, green, blue)):
#             raise ValueError("OBJ diffuse material values must be in [0, 1]")
#         material_name = "mesh_material"
#         material_path = path.with_suffix(".mtl")
#         with material_path.open("w", encoding="utf-8") as material:
#             material.write(f"newmtl {material_name}\n")
#             material.write(f"Ka {0.25 * red:.4f} {0.25 * green:.4f} {0.25 * blue:.4f}\n")
#             material.write(f"Kd {red:.4f} {green:.4f} {blue:.4f}\n")
#             material.write("Ks 0.1000 0.1000 0.1000\n")
#             material.write("Ns 24.0000\n")
#             material.write("d 1.0000\n")
#             material.write("illum 2\n")
#     with path.open("w", encoding="utf-8") as file:
#         if material_name is not None:
#             file.write(f"mtllib {path.with_suffix('.mtl').name}\n")
#             file.write(f"usemtl {material_name}\n")
#         for x, y, z in vertices:
#             file.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
#         for a, b, c in faces:
#             file.write(f"f {a + 1} {b + 1} {c + 1}\n")


# def register_semantic_mesh(
#     simulator: Any, mesh_path: Path, semantic_id: int
# ) -> Any:
#     template_manager = simulator.get_object_template_manager()
#     object_manager = simulator.get_rigid_object_manager()
#     template = template_manager.create_new_template(str(mesh_path))
#     template.render_asset_handle = str(mesh_path)
#     template.collision_asset_handle = str(mesh_path)
#     template.is_collidable = False
#     template_id = template_manager.register_template(
#         template, f"s2diff_obstacle_{semantic_id}_{os.path.basename(mesh_path)}"
#     )
#     object_handle = template_manager.get_template_handle_by_id(template_id)
#     obstacle = object_manager.add_object_by_template_handle(object_handle)
#     obstacle.motion_type = habitat_sim.physics.MotionType.KINEMATIC
#     obstacle.collidable = False
#     obstacle.semantic_id = int(semantic_id)
#     return obstacle


# def parse_world_xz(specification: str) -> tuple[float, float]:
#     values = [float(value) for value in str(specification).split(",")]
#     if len(values) != 2 or not np.isfinite(values).all():
#         raise ValueError(
#             f"world mesh position must be finite X,Z, got {specification!r}"
#         )
#     return values[0], values[1]


# def world_box_mesh(
#     center_x: float,
#     base_y: float,
#     center_z: float,
#     half_extent: float,
#     height: float,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """Create a closed axis-aligned box whose vertices are in world coordinates."""

#     if half_extent <= 0.0 or height <= 0.0:
#         raise ValueError("box half extent and height must be positive")
#     x0, x1 = center_x - half_extent, center_x + half_extent
#     z0, z1 = center_z - half_extent, center_z + half_extent
#     y0, y1 = base_y, base_y + height
#     vertices = np.asarray(
#         [
#             [x0, y0, z0],
#             [x1, y0, z0],
#             [x1, y0, z1],
#             [x0, y0, z1],
#             [x0, y1, z0],
#             [x1, y1, z0],
#             [x1, y1, z1],
#             [x0, y1, z1],
#         ],
#         dtype=np.float64,
#     )
#     faces = np.asarray(
#         [
#             [0, 2, 1], [0, 3, 2],
#             [4, 5, 6], [4, 6, 7],
#             [0, 1, 5], [0, 5, 4],
#             [1, 2, 6], [1, 6, 5],
#             [2, 3, 7], [2, 7, 6],
#             [3, 0, 4], [3, 4, 7],
#         ],
#         dtype=np.int64,
#     )
#     return vertices, faces


# def place_world_obstacle_meshes(
#     simulator: Any,
#     terrain: Any,
#     xz_specifications: Sequence[str],
#     output_directory: Path,
#     *,
#     half_extent: float,
#     height: float,
# ) -> tuple[list[Any], list[np.ndarray], list[np.ndarray]]:
#     """Place static obstacle boxes at exact world X,Z coordinates."""

#     mesh_directory = output_directory / "meshes"
#     mesh_directory.mkdir(parents=True, exist_ok=True)
#     objects: list[Any] = []
#     centroids: list[np.ndarray] = []
#     geometries: list[np.ndarray] = []
#     for index, specification in enumerate(xz_specifications):
#         center_x, center_z = parse_world_xz(specification)
#         base_y = terrain.local_height_max(center_x, center_z, half_extent)
#         vertices, faces = world_box_mesh(
#             center_x, base_y, center_z, half_extent, height
#         )
#         mesh_path = mesh_directory / f"world_obstacle_{index}.obj"
#         save_obj(
#             mesh_path, vertices, faces, diffuse_rgb=(0.78, 0.16, 0.06)
#         )
#         semantic_id = MESH_OBSTACLE_ID + index
#         objects.append(register_semantic_mesh(simulator, mesh_path, semantic_id))
#         centroid = vertices.mean(axis=0).astype(np.float32)
#         centroids.append(centroid)
#         geometries.append(vertices[faces][:, :, [0, 2]].astype(np.float64))
#         print(
#             f"[world-mesh] obstacle={index} semantic_id={semantic_id} "
#             f"center_xz={[center_x, center_z]} half_extent={half_extent:.3f} "
#             f"height={height:.3f}",
#             flush=True,
#         )
#     return objects, centroids, geometries


# def place_world_goal_mesh(
#     simulator: Any,
#     terrain: Any,
#     goal_x: float,
#     goal_z: float,
#     output_directory: Path,
#     *,
#     half_extent: float,
#     height: float,
# ) -> Any:
#     """Place a visible, non-obstacle semantic goal marker at the exact goal."""

#     base_y = terrain.local_height_max(goal_x, goal_z, half_extent)
#     vertices, faces = world_box_mesh(
#         goal_x, base_y, goal_z, half_extent, height
#     )
#     mesh_directory = output_directory / "meshes"
#     mesh_directory.mkdir(parents=True, exist_ok=True)
#     mesh_path = mesh_directory / "goal_marker.obj"
#     save_obj(
#         mesh_path, vertices, faces, diffuse_rgb=(0.08, 0.85, 0.18)
#     )
#     goal_object = register_semantic_mesh(simulator, mesh_path, MESH_GOAL_ID)
#     print(
#         f"[world-mesh] goal semantic_id={MESH_GOAL_ID} "
#         f"center_xz={[goal_x, goal_z]}",
#         flush=True,
#     )
#     return goal_object


# def parse_uv_fraction(specification: str, width: int, height: int) -> tuple[float, float]:
#     u_fraction, v_fraction = (
#         float(value) for value in str(specification).split(",")
#     )
#     if not (0.0 <= u_fraction <= 1.0 and 0.0 <= v_fraction <= 1.0):
#         raise ValueError(
#             f"mesh pixel fraction must be in [0,1], got {specification!r}"
#         )
#     return u_fraction * width, v_fraction * height


# def place_obstacle_meshes(
#     simulator: Any,
#     depth: np.ndarray,
#     position: np.ndarray,
#     yaw: float,
#     intrinsic: np.ndarray,
#     uv_specifications: Sequence[str],
#     output_directory: Path,
#     *,
#     mesh_half_pixels: int,
#     mesh_lift: float,
# ) -> tuple[list[Any], list[np.ndarray], list[np.ndarray]]:
#     mesh_directory = output_directory / "meshes"
#     mesh_directory.mkdir(parents=True, exist_ok=True)
#     height, width = depth.shape
#     objects: list[Any] = []
#     centroids: list[np.ndarray] = []
#     geometries: list[np.ndarray] = []
#     for index, specification in enumerate(uv_specifications):
#         u, v = parse_uv_fraction(specification, width, height)
#         vertices, faces = depth_patch_mesh(
#             u,
#             v,
#             mesh_half_pixels,
#             2,
#             depth,
#             position,
#             yaw,
#             intrinsic,
#             lift=mesh_lift,
#         )
#         if len(vertices) == 0 or len(faces) == 0:
#             raise RuntimeError(
#                 f"obstacle mesh {index} at {specification!r} has no valid depth surface"
#             )
#         mesh_path = mesh_directory / f"obstacle_{index}.obj"
#         save_obj(mesh_path, vertices, faces)
#         semantic_id = MESH_OBSTACLE_ID + index
#         objects.append(register_semantic_mesh(simulator, mesh_path, semantic_id))
#         centroid = vertices.mean(axis=0).astype(np.float32)
#         centroids.append(centroid)
#         geometries.append(vertices[faces][:, :, [0, 2]].astype(np.float64))
#         print(
#             f"[mesh] obstacle={index} semantic_id={semantic_id} "
#             f"pixels={specification} vertices={len(vertices)} "
#             f"world={centroid.tolist()}",
#             flush=True,
#         )
#     return objects, centroids, geometries


# def planar_mesh_clearance(
#     point_xz: np.ndarray,
#     geometries: Sequence[np.ndarray],
# ) -> float:
#     """Minimum 2-D distance from a robot center to projected mesh triangles."""
#     point = np.asarray(point_xz, dtype=np.float64)
#     best = float("inf")
#     for triangles in geometries:
#         triangles = np.asarray(triangles, dtype=np.float64)
#         if triangles.size == 0:
#             continue
#         a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
#         v0, v1, v2 = b - a, c - a, point[None, :] - a
#         denominator = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
#         valid = np.abs(denominator) > 1.0e-12
#         safe_denominator = np.where(valid, denominator, 1.0)
#         u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / safe_denominator
#         v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / safe_denominator
#         if np.any(valid & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)):
#             return 0.0

#         starts = np.concatenate((a, b, c), axis=0)
#         ends = np.concatenate((b, c, a), axis=0)
#         segments = ends - starts
#         squared_lengths = np.einsum("ij,ij->i", segments, segments)
#         numerators = np.einsum("ij,ij->i", point[None, :] - starts, segments)
#         fractions = np.divide(
#             numerators,
#             squared_lengths,
#             out=np.zeros_like(numerators),
#             where=squared_lengths > 1.0e-12,
#         )
#         fractions = np.clip(fractions, 0.0, 1.0)
#         closest = starts + fractions[:, None] * segments
#         best = min(best, float(np.linalg.norm(point[None, :] - closest, axis=1).min()))
#     return best


# def parse_xz_velocity(specification: str) -> np.ndarray:
#     values = [float(value) for value in str(specification).split(",")]
#     if len(values) != 2 or not np.all(np.isfinite(values)):
#         raise ValueError("obstacle velocity must be finite vx,vz")
#     return np.asarray(values, dtype=np.float64)


# def expand_obstacle_velocities(
#     specifications: Sequence[str], obstacle_count: int
# ) -> np.ndarray:
#     if obstacle_count == 0:
#         return np.zeros((0, 2), dtype=np.float64)
#     if not specifications:
#         return np.zeros((obstacle_count, 2), dtype=np.float64)
#     velocities = np.stack([parse_xz_velocity(item) for item in specifications])
#     if len(velocities) == 1 and obstacle_count > 1:
#         velocities = np.repeat(velocities, obstacle_count, axis=0)
#     if len(velocities) != obstacle_count:
#         raise ValueError(
#             "provide one obstacle velocity to broadcast or one velocity per mesh"
#         )
#     return velocities


# def translated_mesh_geometry(
#     base_geometries: Sequence[np.ndarray],
#     base_centroids: Sequence[np.ndarray],
#     velocities_xz: np.ndarray,
#     elapsed_seconds: float,
# ) -> tuple[list[np.ndarray], list[np.ndarray]]:
#     geometries: list[np.ndarray] = []
#     centroids: list[np.ndarray] = []
#     for geometry, centroid, velocity in zip(
#         base_geometries, base_centroids, velocities_xz
#     ):
#         offset_xz = np.asarray(velocity, dtype=np.float64) * elapsed_seconds
#         geometries.append(np.asarray(geometry) + offset_xz[None, None, :])
#         offset_xyz = np.asarray([offset_xz[0], 0.0, offset_xz[1]])
#         centroids.append(np.asarray(centroid, dtype=np.float64) + offset_xyz)
#     return geometries, centroids


# def move_mesh_objects(
#     objects: Sequence[Any], velocities_xz: np.ndarray, elapsed_seconds: float
# ) -> None:
#     for obstacle, velocity in zip(objects, velocities_xz):
#         dx, dz = np.asarray(velocity, dtype=np.float64) * elapsed_seconds
#         vector_type = type(obstacle.translation)
#         obstacle.translation = vector_type(float(dx), 0.0, float(dz))


# def camera_coordinates(
#     point: np.ndarray, position: np.ndarray, yaw: float
# ) -> tuple[float, float, float]:
#     delta = np.asarray(point, dtype=np.float32) - np.asarray(position, dtype=np.float32)
#     forward_x, forward_z = -math.sin(yaw), -math.cos(yaw)
#     left_x, left_z = -math.cos(yaw), math.sin(yaw)
#     forward = forward_x * float(delta[0]) + forward_z * float(delta[2])
#     left = left_x * float(delta[0]) + left_z * float(delta[2])
#     return -left, float(delta[1]), forward


# def camera_intrinsic(height: int, width: int, hfov_deg: float) -> np.ndarray:
#     hfov = math.radians(float(hfov_deg))
#     focal = (width * 0.5) / max(math.tan(hfov * 0.5), 1e-6)
#     return np.asarray(
#         [
#             [focal, 0.0, (width - 1) * 0.5],
#             [0.0, focal, (height - 1) * 0.5],
#             [0.0, 0.0, 1.0],
#         ],
#         dtype=np.float32,
#     )


# def world_goal_to_pixel(
#     point: np.ndarray,
#     position: np.ndarray,
#     yaw: float,
#     intrinsic: np.ndarray,
#     height: int,
#     width: int,
# ) -> np.ndarray:
#     """Project a world goal to a valid PixelGoal, clamping off-screen bearings."""

#     right, up, forward = camera_coordinates(point, position, yaw)
#     fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
#     cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
#     margin = 11
#     bearing = math.atan2(right, forward)
#     maximum_bearing = math.atan2(max(cx - margin, 1.0), fx)
#     bearing = float(np.clip(bearing, -maximum_bearing, maximum_bearing))
#     u = cx + fx * math.tan(bearing)
#     v = cy - fy * up / forward if forward > 0.05 else 0.62 * height
#     return np.asarray(
#         [
#             int(np.clip(round(u), margin, width - margin - 1)),
#             int(np.clip(round(v), margin, height - margin - 1)),
#         ],
#         dtype=np.int32,
#     )


# def circle_mask(height: int, width: int, u: float, v: float, radius: int) -> np.ndarray:
#     yy, xx = np.ogrid[:height, :width]
#     return (((xx - u) ** 2 + (yy - v) ** 2) <= radius**2).astype(np.uint8)


# def project_world_mask(
#     point: np.ndarray,
#     position: np.ndarray,
#     yaw: float,
#     intrinsic: np.ndarray,
#     height: int,
#     width: int,
#     radius: int,
# ) -> tuple[np.ndarray, float]:
#     right, up, forward = camera_coordinates(point, position, yaw)
#     if forward <= 0.05:
#         return np.zeros((height, width), dtype=np.uint8), forward
#     u = float(intrinsic[0, 2] + intrinsic[0, 0] * right / forward)
#     v = float(intrinsic[1, 2] - intrinsic[1, 1] * up / forward)
#     if not (radius <= u < width - radius and radius <= v < height - radius):
#         return np.zeros((height, width), dtype=np.uint8), forward
#     return circle_mask(height, width, u, v, radius), forward


# def depth_obstacle_mask(
#     depth: np.ndarray, threshold: float, minimum_y_fraction: float
# ) -> np.ndarray:
#     mask = np.isfinite(depth) & (depth > 0.05) & (depth < float(threshold))
#     mask[: int(depth.shape[0] * minimum_y_fraction)] = False
#     return mask.astype(np.uint8)


# def pixels_from_mask(mask: np.ndarray, maximum: int) -> np.ndarray:
#     v, u = np.nonzero(np.asarray(mask) > 0)
#     if u.size == 0:
#         return np.zeros((0, 2), dtype=np.int32)
#     pixels = np.stack((u, v), axis=-1).astype(np.int32)
#     if maximum > 0 and len(pixels) > maximum:
#         indices = np.linspace(0, len(pixels) - 1, maximum).astype(np.int64)
#         pixels = pixels[indices]
#     return pixels


# def waypoint_action(
#     trajectory: np.ndarray,
#     *,
#     lookahead_index: int,
#     maximum_forward_speed: float,
#     maximum_yaw_rate: float,
#     yaw_gain: float,
# ) -> np.ndarray:
#     trajectory = np.asarray(trajectory, dtype=np.float32)
#     if trajectory.ndim != 2 or trajectory.shape[0] == 0 or trajectory.shape[1] < 2:
#         return np.zeros(3, dtype=np.float32)
#     if np.max(np.linalg.norm(trajectory[:, :2], axis=-1)) < 1e-5:
#         return np.zeros(3, dtype=np.float32)
#     index = int(np.clip(lookahead_index, 0, trajectory.shape[0] - 1))
#     forward, left = float(trajectory[index, 0]), float(trajectory[index, 1])
#     bearing = math.atan2(left, max(forward, 1e-4))
#     velocity = maximum_forward_speed * max(0.0, math.cos(bearing))
#     yaw_rate = float(np.clip(yaw_gain * bearing, -maximum_yaw_rate, maximum_yaw_rate))
#     return np.asarray([velocity, 0.0, yaw_rate], dtype=np.float32)


# def integrate_mars(
#     position: np.ndarray, yaw: float, action: np.ndarray, dt: float
# ) -> tuple[np.ndarray, float]:
#     forward_velocity, lateral_velocity, yaw_rate = [float(value) for value in action]
#     forward_x, forward_z = -math.sin(yaw), -math.cos(yaw)
#     left_x, left_z = -math.cos(yaw), math.sin(yaw)
#     output = np.asarray(position, dtype=np.float32).copy()
#     output[0] += (forward_x * forward_velocity + left_x * lateral_velocity) * dt
#     output[2] += (forward_z * forward_velocity + left_z * lateral_velocity) * dt
#     return output, yaw + yaw_rate * dt


# def wrap_angle(angle: float) -> float:
#     return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


# def overlay_frame(
#     rgb: np.ndarray,
#     goal_mask: np.ndarray,
#     obstacle_mask: np.ndarray,
#     text: str,
#     *,
#     show_masks: bool,
#     detection_box: Optional[np.ndarray] = None,
#     detection_label: Optional[str] = None,
# ) -> Image.Image:
#     output = np.asarray(rgb, dtype=np.uint8).copy()
#     if show_masks:
#         output[goal_mask > 0] = (
#             0.35 * output[goal_mask > 0] + 0.65 * np.asarray([0, 255, 0])
#         ).astype(np.uint8)
#         output[obstacle_mask > 0] = (
#             0.35 * output[obstacle_mask > 0] + 0.65 * np.asarray([255, 0, 0])
#         ).astype(np.uint8)
#     image = Image.fromarray(output)
#     draw = ImageDraw.Draw(image)
#     if detection_box is not None:
#         x1, y1, x2, y2 = [float(value) for value in detection_box]
#         draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 0), width=3)
#         if detection_label:
#             draw.text((x1 + 2, max(y1 - 14, 2)), detection_label, fill=(255, 255, 0))
#     draw.rectangle((5, 5, min(image.width - 5, 12 + len(text) * 7), 28), fill=(0, 0, 0))
#     draw.text((10, 9), text, fill=(255, 255, 255))
#     return image


# def save_video(frames: Sequence[Image.Image], path: Path, fps: float) -> None:
#     import imageio.v2 as imageio

#     with imageio.get_writer(path, fps=float(fps)) as writer:
#         for frame in frames:
#             writer.append_data(np.asarray(frame.convert("RGB")))


# def parser() -> argparse.ArgumentParser:
#     argument_parser = argparse.ArgumentParser(
#         description="One-file released NavDP + in-denoising S2Diff Mars rollout"
#     )
#     argument_parser.add_argument("--navdp-root", required=True)
#     argument_parser.add_argument("--navdp-checkpoint", required=True)
#     argument_parser.add_argument("--navdp-python", default=sys.executable)
#     argument_parser.add_argument("--navdp-device", default="cuda:0")
#     argument_parser.add_argument(
#         "--planner-mode", choices=["pure-navdp", "s2diff", "gradient"], default="s2diff"
#     )
#     argument_parser.add_argument(
#         "--goal-mode", choices=["point", "pixel"], default="point"
#     )
#     argument_parser.add_argument(
#         "--belief-pixel-goal",
#         action=argparse.BooleanOptionalAction,
#         default=False,
#         help=(
#             "Use the live semantic goal mask to correct a body-frame Gaussian "
#             "belief and its projected mean as NavDP's PixelGoal while occluded."
#         ),
#     )
#     argument_parser.add_argument("--belief-minimum-goal-pixels", type=int, default=10)
#     argument_parser.add_argument("--belief-measurement-std", type=float, default=0.05)
#     argument_parser.add_argument(
#         "--belief-translation-process-std", type=float, default=0.03
#     )
#     argument_parser.add_argument(
#         "--belief-yaw-process-std-deg", type=float, default=1.0
#     )
#     argument_parser.add_argument(
#         "--belief-bootstrap-world-goal",
#         action=argparse.BooleanOptionalAction,
#         default=False,
#         help=(
#             "Simulation-only bootstrap when the goal is initially invisible. "
#             "Disable for a strict detector-only evaluation."
#         ),
#     )
#     argument_parser.add_argument("--belief-bootstrap-std", type=float, default=0.50)
#     argument_parser.add_argument("--belief-ghost-base-radius", type=int, default=10)
#     argument_parser.add_argument("--belief-ghost-covariance-scale", type=float, default=2.0)
#     argument_parser.add_argument("--belief-ghost-maximum-radius", type=int, default=80)
#     argument_parser.add_argument(
#         "--belief-heading-recovery",
#         action=argparse.BooleanOptionalAction,
#         default=True,
#     )
#     argument_parser.add_argument("--belief-recovery-bearing-deg", type=float, default=35.0)
#     argument_parser.add_argument("--belief-recovery-yaw-gain", type=float, default=1.5)
#     argument_parser.add_argument(
#         "--belief-recovery-maximum-yaw-rate", type=float, default=0.70
#     )
#     argument_parser.add_argument(
#         "--belief-recovery-maximum-forward-speed", type=float, default=0.12
#     )
#     argument_parser.add_argument(
#         "--qwen-model-id", default="Qwen/Qwen2.5-VL-3B-Instruct"
#     )
#     argument_parser.add_argument("--qwen-device", default="auto")
#     argument_parser.add_argument("--qwen-homotopy-python", default=sys.executable)
#     argument_parser.add_argument("--qwen-homotopy-host", default="127.0.0.1")
#     argument_parser.add_argument("--qwen-homotopy-port", type=int, default=8890)
#     argument_parser.add_argument("--qwen-homotopy-timeout", type=float, default=600.0)
#     argument_parser.add_argument(
#         "--start-qwen-homotopy-server",
#         action=argparse.BooleanOptionalAction,
#         default=True,
#     )
#     argument_parser.add_argument(
#         "--qwen-homotopy", action=argparse.BooleanOptionalAction, default=False,
#         help=(
#             "When a metric obstacle becomes relevant, Qwen chooses the single "
#             "LEFT/RIGHT circulation sign used by every trajectory candidate."
#         ),
#     )
#     argument_parser.add_argument("--homotopy-minimum-obstacle-pixels", type=int, default=30)
#     argument_parser.add_argument("--homotopy-release-clear-frames", type=int, default=8)
#     argument_parser.add_argument(
#         "--homotopy-consistency-repeats", type=int, default=5,
#         help="Repeat Qwen on the identical obstacle frame and use majority vote.",
#     )
#     argument_parser.add_argument(
#         "--remove-critic", action=argparse.BooleanOptionalAction, default=True
#     )
#     argument_parser.add_argument("--seed", type=int, default=7)
#     argument_parser.add_argument(
#         "--start-server", action=argparse.BooleanOptionalAction, default=True
#     )
#     argument_parser.add_argument("--server-host", default="127.0.0.1")
#     argument_parser.add_argument("--server-port", type=int, default=8888)
#     argument_parser.add_argument("--server-timeout", type=float, default=180.0)
#     argument_parser.add_argument("--candidates", type=int, default=16)
#     argument_parser.add_argument("--particles", type=int, default=8)
#     argument_parser.add_argument("--particle-std", type=float, default=0.22)
#     argument_parser.add_argument("--gradient-steps", type=int, default=3)
#     argument_parser.add_argument("--gradient-step-size", type=float, default=0.04)
#     argument_parser.add_argument("--guidance-strength", type=float, default=0.85)
#     argument_parser.add_argument("--temperature", type=float, default=0.35)
#     argument_parser.add_argument(
#         "--particle-anchor", action=argparse.BooleanOptionalAction, default=True
#     )
#     argument_parser.add_argument(
#         "--particle-energy-reweighting",
#         action=argparse.BooleanOptionalAction,
#         default=True,
#     )
#     argument_parser.add_argument(
#         "--particle-collision-mask", action=argparse.BooleanOptionalAction, default=True
#     )
#     argument_parser.add_argument(
#         "--particle-noise-schedule", action=argparse.BooleanOptionalAction, default=True
#     )
#     argument_parser.add_argument(
#         "--progressive-guidance", action=argparse.BooleanOptionalAction, default=True
#     )
#     argument_parser.add_argument("--safe-distance", type=float, default=0.42)
#     argument_parser.add_argument("--hard-collision-distance", type=float, default=0.24)
#     argument_parser.add_argument("--safety-weight", type=float, default=35.0)
#     argument_parser.add_argument("--barrier-weight", type=float, default=25.0)
#     argument_parser.add_argument("--barrier-rate", type=float, default=0.15)
#     argument_parser.add_argument("--circulation-weight", type=float, default=18.0)
#     argument_parser.add_argument(
#         "--circulation-activation-distance", type=float, default=1.50
#     )
#     argument_parser.add_argument(
#         "--circulation-activation-sharpness", type=float, default=0.20
#     )
#     argument_parser.add_argument(
#         "--minimum-circulation-progress", type=float, default=0.025
#     )
#     argument_parser.add_argument(
#         "--blocking-alignment-threshold", type=float, default=0.25
#     )
#     argument_parser.add_argument("--circulation-switch-weight", type=float, default=2.0)
#     argument_parser.add_argument("--escape-lateral-target", type=float, default=0.35)
#     argument_parser.add_argument("--minimum-obstacle-depth", type=float, default=0.10)
#     argument_parser.add_argument("--maximum-obstacle-depth", type=float, default=5.0)
#     argument_parser.add_argument("--maximum-obstacle-pixels", type=int, default=1536)

#     argument_parser.add_argument("--scene", required=True)
#     argument_parser.add_argument("--terrain-obj", default=None)
#     argument_parser.add_argument("--heightmap", default=None)
#     argument_parser.add_argument(
#         "--terrain-height-mode",
#         choices=["auto", "heightmap", "obj", "flat"],
#         default="auto",
#     )
#     argument_parser.add_argument("--flat-y", type=float, default=0.0)
#     argument_parser.add_argument("--size-x", type=float, default=SIZE_X)
#     argument_parser.add_argument("--size-z", type=float, default=SIZE_Z)
#     argument_parser.add_argument("--size-y", type=float, default=SIZE_Y)
#     argument_parser.add_argument("--flip-heightmap-x", action="store_true")
#     argument_parser.add_argument(
#         "--flip-heightmap-z", action=argparse.BooleanOptionalAction, default=True
#     )
#     argument_parser.add_argument("--swap-heightmap-xz", action="store_true")
#     argument_parser.add_argument("--clearance", type=float, default=1.4)
#     argument_parser.add_argument("--pose-terrain-radius", type=float, default=0.8)
#     argument_parser.add_argument(
#         "--robot-radius",
#         type=float,
#         default=0.24,
#         help="Planar rover footprint radius used by both guidance and evaluation.",
#     )
#     argument_parser.add_argument(
#         "--evaluation-layout",
#         default="default",
#         help="Stable layout identifier stored in the rollout archive.",
#     )

#     argument_parser.add_argument("--height", type=int, default=720)
#     argument_parser.add_argument("--width", type=int, default=720)
#     argument_parser.add_argument("--hfov-deg", type=float, default=90.0)
#     argument_parser.add_argument("--hz", type=float, default=10.0)
#     argument_parser.add_argument("--max-steps", type=int, default=300)
#     argument_parser.add_argument("--stop-distance", type=float, default=1.0)
#     argument_parser.add_argument("--start-x", type=float, default=0.0)
#     argument_parser.add_argument("--start-z", type=float, default=8.0)
#     argument_parser.add_argument("--start-yaw-deg", type=float, default=0.0)
#     argument_parser.add_argument("--goal-x", type=float, default=None)
#     argument_parser.add_argument("--goal-z", type=float, default=None)
#     argument_parser.add_argument("--goal-y", type=float, default=None)
#     argument_parser.add_argument("--goal-height", type=float, default=1.2)
#     argument_parser.add_argument("--goal-radius", type=int, default=18)
#     argument_parser.add_argument(
#         "--goal-mesh", action=argparse.BooleanOptionalAction, default=False
#     )
#     argument_parser.add_argument("--goal-mesh-half-extent", type=float, default=0.25)
#     argument_parser.add_argument("--goal-mesh-height", type=float, default=1.50)

#     argument_parser.add_argument(
#         "--obstacle-mode", choices=["none", "depth", "mesh", "ghost"], default="none"
#     )
#     argument_parser.add_argument("--obstacle-depth-threshold", type=float, default=1.4)
#     argument_parser.add_argument("--obstacle-min-y-fraction", type=float, default=0.45)
#     argument_parser.add_argument("--ghost-obstacle-x", type=float, default=None)
#     argument_parser.add_argument("--ghost-obstacle-z", type=float, default=None)
#     argument_parser.add_argument("--ghost-obstacle-y", type=float, default=None)
#     argument_parser.add_argument("--ghost-obstacle-height", type=float, default=0.45)
#     argument_parser.add_argument("--ghost-obstacle-radius", type=int, default=24)
#     argument_parser.add_argument(
#         "--obstacle-mesh-uv",
#         nargs="+",
#         default=[],
#         help=(
#             "Actual rendered obstacle mesh locations as image fractions u,v. "
#             "Example: --obstacle-mesh-uv 0.50,0.72 0.30,0.68"
#         ),
#     )
#     argument_parser.add_argument(
#         "--obstacle-world-xz",
#         nargs="*",
#         default=[],
#         metavar="X,Z",
#         help=(
#             "Static rendered obstacle-box centers in world X,Z coordinates. "
#             "Example: --obstacle-world-xz 0,0. Do not combine with "
#             "--obstacle-mesh-uv."
#         ),
#     )
#     argument_parser.add_argument(
#         "--obstacle-world-xz-item",
#         action="append",
#         default=[],
#         metavar="X,Z",
#         help=(
#             "Repeatable form that safely accepts negative coordinates, e.g. "
#             "--obstacle-world-xz-item=-3,0."
#         ),
#     )
#     argument_parser.add_argument(
#         "--world-obstacle-half-extent", type=float, default=0.75
#     )
#     argument_parser.add_argument(
#         "--world-obstacle-height", type=float, default=1.40
#     )
#     argument_parser.add_argument("--mesh-half-pixels", type=int, default=26)
#     argument_parser.add_argument("--mesh-obstacle-lift", type=float, default=0.50)
#     argument_parser.add_argument(
#         "--obstacle-velocity-xz",
#         nargs="*",
#         default=[],
#         metavar="VX,VZ",
#         help=(
#             "World-frame mesh velocities in m/s. Supply one value to broadcast "
#             "or one value per obstacle. Example: --obstacle-velocity-xz 0.30,0.0"
#         ),
#     )

#     argument_parser.add_argument("--lookahead-index", type=int, default=4)
#     argument_parser.add_argument("--maximum-forward-speed", type=float, default=0.5)
#     argument_parser.add_argument("--maximum-yaw-rate", type=float, default=0.5)
#     argument_parser.add_argument("--yaw-gain", type=float, default=1.5)
#     argument_parser.add_argument("--output", default="runs/navdp_s2diff_mars")
#     argument_parser.add_argument("--save-every", type=int, default=1)
#     argument_parser.add_argument(
#         "--save-frames", action=argparse.BooleanOptionalAction, default=True
#     )
#     argument_parser.add_argument(
#         "--save-video", action=argparse.BooleanOptionalAction, default=True
#     )
#     argument_parser.add_argument(
#         "--archive-observations",
#         action=argparse.BooleanOptionalAction,
#         default=True,
#         help="Store RGB/depth/masks in rollout.npz; disable for large evaluations.",
#     )
#     argument_parser.add_argument(
#         "--overlay-masks", action=argparse.BooleanOptionalAction, default=True
#     )
#     return argument_parser


# def main() -> None:
#     args = parser().parse_args()
#     if args.obstacle_world_xz_item:
#         args.obstacle_world_xz.extend(args.obstacle_world_xz_item)
#     np.random.seed(args.seed)
#     if args.goal_x is None or args.goal_z is None:
#         raise ValueError("fixed PointGoal requires --goal-x and --goal-z")
#     if args.belief_pixel_goal and args.goal_mode != "pixel":
#         raise ValueError("--belief-pixel-goal requires --goal-mode pixel")
#     if args.belief_pixel_goal and not args.goal_mesh:
#         raise ValueError(
#             "simulation belief tracking requires --goal-mesh so a live semantic "
#             "goal observation exists"
#         )
#     if args.belief_minimum_goal_pixels < 1:
#         raise ValueError("belief-minimum-goal-pixels must be positive")
#     if min(
#         args.belief_measurement_std,
#         args.belief_translation_process_std,
#         args.belief_yaw_process_std_deg,
#         args.belief_bootstrap_std,
#     ) < 0.0:
#         raise ValueError("belief uncertainty parameters must be non-negative")
#     if args.robot_radius < 0.0:
#         raise ValueError("robot-radius must be non-negative")
#     if args.obstacle_velocity_xz and args.obstacle_mode != "mesh":
#         raise ValueError("moving obstacle velocities require --obstacle-mode mesh")
#     if args.obstacle_mode == "ghost" and (
#         args.ghost_obstacle_x is None or args.ghost_obstacle_z is None
#     ):
#         raise ValueError("ghost mode requires --ghost-obstacle-x and --ghost-obstacle-z")
#     if args.obstacle_mesh_uv and args.obstacle_world_xz:
#         raise ValueError(
#             "choose either --obstacle-mesh-uv or --obstacle-world-xz, not both"
#         )
#     if args.obstacle_mode == "mesh" and not (
#         args.obstacle_mesh_uv or args.obstacle_world_xz
#     ):
#         raise ValueError(
#             "mesh mode requires --obstacle-world-xz X,Z [X,Z ...] or "
#             "--obstacle-mesh-uv u,v [u,v ...]"
#         )
#     if args.world_obstacle_half_extent <= 0.0 or args.world_obstacle_height <= 0.0:
#         raise ValueError("world obstacle dimensions must be positive")
#     if args.goal_mesh_half_extent <= 0.0 or args.goal_mesh_height <= 0.0:
#         raise ValueError("goal mesh dimensions must be positive")


#     if args.qwen_homotopy and args.planner_mode == "pure-navdp":
#         raise ValueError("Qwen homotopy conditioning requires s2diff or gradient mode")

#     qwen_process: Optional[subprocess.Popen[Any]] = None
#     server_process: Optional[subprocess.Popen[Any]] = None
#     simulator = None
#     try:
#         qwen_process = start_qwen_homotopy_server(args)
#         homotopy_selector = None
#         if args.qwen_homotopy:
#             homotopy_selector = QwenHomotopyClient(
#                 f"http://{args.qwen_homotopy_host}:{args.qwen_homotopy_port}",
#                 timeout=args.qwen_homotopy_timeout,
#             )
#             homotopy_selector.reset()
#         server_process = start_server(args)
#         server_url = f"http://{args.server_host}:{args.server_port}"
#         client = NavDPS2DiffClient(server_url)
#         algorithm = client.reset(
#             camera_intrinsic(args.height, args.width, args.hfov_deg),
#             batch_size=1,
#             stop_threshold=-3.0,
#         )
#         supported_algorithms = {
#             "navdp-s2diff-pixels",
#             "navdp-hlc-s2diff",
#             "navdp-hlc-s2diff-no-critic",
#             "navdp-hlc-gradient",
#             "navdp-hlc-gradient-no-critic",
#             "navdp-pure-critic",
#         }
#         if algorithm not in supported_algorithms:
#             raise RuntimeError(f"unexpected planner response: {algorithm!r}")

#         terrain = TerrainHeight(
#             mode=args.terrain_height_mode,
#             heightmap=Path(args.heightmap).expanduser().resolve() if args.heightmap else None,
#             obj=Path(args.terrain_obj).expanduser().resolve() if args.terrain_obj else None,
#             flat_y=args.flat_y,
#             size_x=args.size_x,
#             size_z=args.size_z,
#             size_y=args.size_y,
#             flip_x=args.flip_heightmap_x,
#             flip_z=args.flip_heightmap_z,
#             swap_xz=args.swap_heightmap_xz,
#         )
#         output_directory = Path(args.output).expanduser().resolve()
#         frame_directory = output_directory / "frames"
#         frame_directory.mkdir(parents=True, exist_ok=True)

#         simulator = make_simulator(
#             Path(args.scene),
#             args.height,
#             args.width,
#             args.hfov_deg,
#             with_semantic=args.obstacle_mode == "mesh" or args.goal_mesh,
#         )
#         agent = simulator.initialize_agent(0)
#         intrinsic = camera_intrinsic(args.height, args.width, args.hfov_deg)
#         goal_belief = (
#             GaussianGoalBelief(
#                 intrinsic,
#                 (args.height, args.width),
#                 minimum_visible_pixels=args.belief_minimum_goal_pixels,
#                 measurement_std=args.belief_measurement_std,
#                 translation_process_std=args.belief_translation_process_std,
#                 yaw_process_std=math.radians(args.belief_yaw_process_std_deg),
#             )
#             if args.belief_pixel_goal
#             else None
#         )
#         previous_executed_action = np.zeros(3, dtype=np.float32)
#         x, z = float(args.start_x), float(args.start_z)
#         yaw = math.radians(float(args.start_yaw_deg))
#         dt = 1.0 / float(args.hz)

#         goal_y = args.goal_y
#         if goal_y is None:
#             goal_y = terrain.local_height_max(args.goal_x, args.goal_z, 0.8) + args.goal_height
#         goal = np.asarray([args.goal_x, goal_y, args.goal_z], dtype=np.float32)
#         start_position_xz = np.asarray([x, z], dtype=np.float64)
#         initial_goal_distance = float(
#             np.linalg.norm(goal[[0, 2]].astype(np.float64) - start_position_xz)
#         )
#         goal_mesh_object = None
#         if args.goal_mesh:
#             goal_mesh_object = place_world_goal_mesh(
#                 simulator,
#                 terrain,
#                 args.goal_x,
#                 args.goal_z,
#                 output_directory,
#                 half_extent=args.goal_mesh_half_extent,
#                 height=args.goal_mesh_height,
#             )

#         ghost = None
#         if args.obstacle_mode == "ghost":
#             ghost_y = args.ghost_obstacle_y
#             if ghost_y is None:
#                 ghost_y = terrain.local_height_max(
#                     args.ghost_obstacle_x, args.ghost_obstacle_z, args.pose_terrain_radius
#                 ) + args.ghost_obstacle_height
#             ghost = np.asarray(
#                 [args.ghost_obstacle_x, ghost_y, args.ghost_obstacle_z], dtype=np.float32
#             )

#         mesh_objects: list[Any] = []
#         mesh_centroids: list[np.ndarray] = []
#         mesh_current_centroids: list[np.ndarray] = []
#         mesh_base_geometries: list[np.ndarray] = []
#         mesh_geometries: list[np.ndarray] = []
#         mesh_velocities = np.zeros((0, 2), dtype=np.float64)
#         mesh_placed = False
#         if args.obstacle_mode == "mesh" and args.obstacle_world_xz:
#             mesh_objects, mesh_centroids, mesh_base_geometries = (
#                 place_world_obstacle_meshes(
#                     simulator,
#                     terrain,
#                     args.obstacle_world_xz,
#                     output_directory,
#                     half_extent=args.world_obstacle_half_extent,
#                     height=args.world_obstacle_height,
#                 )
#             )
#             mesh_velocities = expand_obstacle_velocities(
#                 args.obstacle_velocity_xz, len(mesh_objects)
#             )
#             mesh_geometries, mesh_current_centroids = translated_mesh_geometry(
#                 mesh_base_geometries, mesh_centroids, mesh_velocities, 0.0
#             )
#             mesh_placed = True

#         row_keys = [
#                 "pose",
#                 "action_3d",
#                 "point_goal",
#                 "belief_goal_mu",
#                 "belief_goal_covariance",
#                 "belief_goal_pixel",
#                 "belief_goal_visible",
#                 "belief_goal_source",
#                 "belief_goal_time_since_seen",
#                 "belief_goal_bearing_rad",
#                 "belief_goal_pixel_sigma",
#                 "belief_heading_recovery_active",
#                 "selected_trajectory",
#                 "all_trajectories",
#                 "all_values",
#                 "selected_index",
#                 "fallback_stop",
#                 "escape_turn",
#                 "valid_obstacle_points",
#                 "selected_circulation_sign",
#                 "candidate_circulation_signs",
#                 "selected_barrier_energy",
#                 "selected_circulation_energy",
#                 "planning_time_seconds",
#                 "selected_minimum_clearance",
#                 "mean_guidance_noise_correction",
#                 "final_guidance_noise_correction",
#                 "maximum_guidance_noise_correction",
#                 "mean_final_effective_sample_size",
#                 "goal_distance",
#                 "executed_center_clearance",
#                 "executed_surface_clearance",
#                 "geometric_collision",
#                 "obstacle_positions_world",

#                 "qwen_homotopy_sign",
#                 "qwen_homotopy_side",
#                 "qwen_homotopy_confidence",
#                 "qwen_homotopy_queried",
#         ]
#         if args.archive_observations:
#             row_keys.extend(
#                 (
#                     "rgb",
#                     "depth",
#                     "goal_mask",
#                     "live_goal_mask",
#                     "ghost_goal_mask",
#                     "obstacle_mask",
#                 )
#             )
#         rows: dict[str, list[Any]] = {key: [] for key in row_keys}
#         video_frames: list[Image.Image] = []
#         success = False
#         homotopy_events: list[dict[str, Any]] = []

#         for step in range(int(args.max_steps)):
#             y = terrain.local_height_max(x, z, args.pose_terrain_radius) + args.clearance
#             position = np.asarray([x, y, z], dtype=np.float32)
#             if goal_belief is not None and step > 0:
#                 goal_belief.predict(previous_executed_action, dt)
#             set_agent_pose(agent, position, yaw)
#             if mesh_placed:
#                 elapsed_seconds = step * dt
#                 move_mesh_objects(mesh_objects, mesh_velocities, elapsed_seconds)
#                 mesh_geometries, mesh_current_centroids = translated_mesh_geometry(
#                     mesh_base_geometries,
#                     mesh_centroids,
#                     mesh_velocities,
#                     elapsed_seconds,
#                 )
#             observation = simulator.get_sensor_observations()
#             rgb, depth = rgb_depth(observation)

#             if args.obstacle_mode == "mesh" and not mesh_placed:
#                 mesh_objects, mesh_centroids, mesh_base_geometries = place_obstacle_meshes(
#                     simulator,
#                     depth,
#                     position,
#                     yaw,
#                     intrinsic,
#                     args.obstacle_mesh_uv,
#                     output_directory,
#                     mesh_half_pixels=args.mesh_half_pixels,
#                     mesh_lift=args.mesh_obstacle_lift,
#                 )
#                 mesh_velocities = expand_obstacle_velocities(
#                     args.obstacle_velocity_xz, len(mesh_objects)
#                 )
#                 mesh_geometries, mesh_current_centroids = translated_mesh_geometry(
#                     mesh_base_geometries, mesh_centroids, mesh_velocities, 0.0
#                 )
#                 mesh_placed = True
#                 observation = simulator.get_sensor_observations()
#                 rgb, depth = rgb_depth(observation)

#             semantic = (
#                 semantic_from_observation(observation)
#                 if args.obstacle_mode == "mesh" or args.goal_mesh
#                 else None
#             )
#             goal_right, _goal_up, goal_forward = camera_coordinates(
#                 goal, position, yaw
#             )
#             point_goal = np.asarray(
#                 [max(goal_forward, 0.0), -goal_right], dtype=np.float32
#             )
#             live_goal_mask = np.zeros(depth.shape, dtype=np.uint8)
#             ghost_goal_mask = np.zeros(depth.shape, dtype=np.uint8)
#             belief_goal_visible = False
#             belief_goal_source = "DISABLED"
#             belief_goal_mu = np.full(2, np.nan, dtype=np.float32)
#             belief_goal_covariance = np.full((2, 2), np.nan, dtype=np.float32)
#             belief_goal_pixel = np.full(2, -1, dtype=np.int32)
#             belief_goal_time_since_seen = float("nan")
#             belief_goal_bearing = float("nan")
#             belief_goal_pixel_sigma = float("nan")

#             if goal_belief is not None:
#                 assert semantic is not None
#                 live_goal_mask = (semantic == MESH_GOAL_ID).astype(np.uint8)
#                 belief_goal_visible = goal_belief.observe(live_goal_mask, depth)
#                 bootstrapped = False
#                 if not goal_belief.initialized:
#                     if not args.belief_bootstrap_world_goal:
#                         raise RuntimeError(
#                             "goal belief is uninitialized because the live goal mask "
#                             "has not been observed; start with the goal visible or pass "
#                             "--belief-bootstrap-world-goal for simulation"
#                         )
#                     goal_belief.initialize(
#                         np.asarray([goal_forward, -goal_right], dtype=np.float32),
#                         args.belief_bootstrap_std,
#                     )
#                     bootstrapped = True
#                 belief_projection = goal_belief.project(
#                     base_radius=args.belief_ghost_base_radius,
#                     covariance_scale=args.belief_ghost_covariance_scale,
#                     maximum_radius=args.belief_ghost_maximum_radius,
#                 )
#                 planner_goal = belief_projection.pixel_uv
#                 ghost_goal_mask = belief_projection.mask
#                 goal_mask = live_goal_mask if belief_goal_visible else ghost_goal_mask
#                 belief_goal_source = (
#                     "LIVE"
#                     if belief_goal_visible
#                     else ("WORLD_BOOTSTRAP" if bootstrapped else "GHOST")
#                 )
#                 assert goal_belief.mu is not None and goal_belief.Sigma is not None
#                 belief_goal_mu = goal_belief.mu.copy()
#                 belief_goal_covariance = goal_belief.Sigma.copy()
#                 belief_goal_pixel = belief_projection.pixel_uv.copy()
#                 belief_goal_time_since_seen = goal_belief.time_since_seen
#                 belief_goal_bearing = belief_projection.bearing_rad
#                 belief_goal_pixel_sigma = belief_projection.pixel_sigma
#             else:
#                 if args.goal_mesh:
#                     assert semantic is not None
#                     live_goal_mask = (semantic == MESH_GOAL_ID).astype(np.uint8)
#                     goal_mask = live_goal_mask
#                     if not np.any(goal_mask):
#                         goal_mask, _ = project_world_mask(
#                             goal,
#                             position,
#                             yaw,
#                             intrinsic,
#                             args.height,
#                             args.width,
#                             args.goal_radius,
#                         )
#                 else:
#                     goal_mask, _ = project_world_mask(
#                         goal,
#                         position,
#                         yaw,
#                         intrinsic,
#                         args.height,
#                         args.width,
#                         args.goal_radius,
#                     )
#                 planner_goal = point_goal
#             if args.goal_mode == "pixel" and goal_belief is None:
#                 planner_goal = world_goal_to_pixel(
#                     goal, position, yaw, intrinsic, args.height, args.width
#                 )
#                 goal_mask = circle_mask(
#                     args.height,
#                     args.width,
#                     planner_goal[0],
#                     planner_goal[1],
#                     args.goal_radius,
#                 )
#             guidance_depth = depth.copy()
#             if args.obstacle_mode == "depth":
#                 obstacle_mask = depth_obstacle_mask(
#                     depth, args.obstacle_depth_threshold, args.obstacle_min_y_fraction
#                 )
#             elif args.obstacle_mode == "mesh":
#                 assert semantic is not None
#                 semantic_ids = list(
#                     range(
#                         MESH_OBSTACLE_ID,
#                         MESH_OBSTACLE_ID + len(mesh_objects),
#                     )
#                 )
#                 obstacle_mask = np.isin(semantic, semantic_ids).astype(np.uint8)
#                 # The depth image was re-rendered after mesh placement, so
#                 # guidance_depth already contains the real obstacle depth.
#             elif args.obstacle_mode == "ghost":
#                 assert ghost is not None
#                 obstacle_mask, obstacle_forward = project_world_mask(
#                     ghost,
#                     position,
#                     yaw,
#                     intrinsic,
#                     args.height,
#                     args.width,
#                     args.ghost_obstacle_radius,
#                 )
#                 if obstacle_forward > 0.05:
#                     guidance_depth[obstacle_mask > 0] = obstacle_forward
#             else:
#                 obstacle_mask = np.zeros(depth.shape, dtype=np.uint8)

#             # Replace this mask-to-pixels line with your own detector's [u,v]
#             # array if obstacle pixels already come directly from your system.
#             obstacle_pixels = pixels_from_mask(
#                 obstacle_mask, args.maximum_obstacle_pixels
#             )
#             homotopy_decision = None
#             forced_circulation_sign = 0.0
#             obstacle_relevant_for_homotopy = False
#             if homotopy_selector is not None:
#                 homotopy_obstacle_mask = (
#                     (obstacle_mask > 0)
#                     & np.isfinite(guidance_depth)
#                     & (guidance_depth >= args.minimum_obstacle_depth)
#                     & (guidance_depth <= args.maximum_obstacle_depth)
#                 ).astype(np.uint8)
#                 qwen_overlay = overlay_frame(
#                     rgb,
#                     goal_mask,
#                     homotopy_obstacle_mask,
#                     "Qwen homotopy: choose LEFT or RIGHT",
#                     show_masks=True,
#                 )
#                 homotopy_decision = homotopy_selector.step(
#                     np.asarray(qwen_overlay.convert("RGB")), homotopy_obstacle_mask
#                 )
#                 obstacle_relevant_for_homotopy = homotopy_decision.obstacle_relevant
#                 forced_circulation_sign = homotopy_decision.circulation_sign
#                 if homotopy_decision.queried_qwen:
#                     event = {
#                         "step": step,
#                         "side": homotopy_decision.side,
#                         "circulation_sign": forced_circulation_sign,
#                         "confidence": homotopy_decision.confidence,
#                         "repeat_sides": list(homotopy_decision.repeated_sides),
#                         "repeat_confidences": list(
#                             homotopy_decision.repeated_confidences
#                         ),
#                         "consistency_rate": homotopy_decision.consistency_rate,
#                         "used_fallback": homotopy_decision.used_fallback,
#                         "raw_response": homotopy_decision.raw_response,
#                     }
#                     homotopy_events.append(event)
#                     query_directory = output_directory / "qwen_homotopy_queries"
#                     query_directory.mkdir(parents=True, exist_ok=True)
#                     qwen_overlay.save(query_directory / f"query_step_{step:04d}.png")
#                     print(
#                         f"[qwen-homotopy] side={homotopy_decision.side} "
#                         f"sign={forced_circulation_sign:+.0f} "
#                         f"confidence={homotopy_decision.confidence:.2f} "
#                         f"consistency={homotopy_decision.consistency_rate:.2%} "
#                         f"repeats={list(homotopy_decision.repeated_sides)} "
#                         f"fallback={homotopy_decision.used_fallback}",
#                         flush=True,
#                     )
#             planning_start = time.perf_counter()
#             result = client.plan(
#                 goal_xy=planner_goal,
#                 rgb=rgb,
#                 depth=guidance_depth,
#                 obstacle_pixels=obstacle_pixels,
#                 goal_mode=args.goal_mode,
#                 forced_circulation_sign=forced_circulation_sign,
#             )
#             planning_time = time.perf_counter() - planning_start
#             action = (
#                 np.zeros(3, dtype=np.float32)
#                 if result.fallback_stop
#                 else waypoint_action(
#                     result.trajectory,
#                     lookahead_index=args.lookahead_index,
#                     maximum_forward_speed=args.maximum_forward_speed,
#                     maximum_yaw_rate=args.maximum_yaw_rate,
#                     yaw_gain=args.yaw_gain,
#                 )
#             )
#             action, belief_recovery_active = belief_heading_recovery_action(
#                 action,
#                 belief_bearing=belief_goal_bearing,
#                 obstacle_relevant=obstacle_relevant_for_homotopy,
#                 enabled=args.belief_pixel_goal and args.belief_heading_recovery,
#                 activation_bearing=math.radians(args.belief_recovery_bearing_deg),
#                 yaw_gain=args.belief_recovery_yaw_gain,
#                 maximum_yaw_rate=args.belief_recovery_maximum_yaw_rate,
#                 maximum_forward_speed=args.belief_recovery_maximum_forward_speed,
#             )

#             next_position, next_yaw = integrate_mars(position, yaw, action, dt)
#             previous_executed_action = action.copy()
#             x = float(np.clip(next_position[0], -args.size_x / 2.0 + 0.5, args.size_x / 2.0 - 0.5))
#             z = float(np.clip(next_position[2], -args.size_z / 2.0 + 0.5, args.size_z / 2.0 - 0.5))
#             yaw = wrap_angle(next_yaw)
#             goal_distance = float(np.linalg.norm(goal[[0, 2]] - np.asarray([x, z])))
#             center_clearance = planar_mesh_clearance(
#                 np.asarray([x, z], dtype=np.float64), mesh_geometries
#             )
#             if np.isfinite(center_clearance):
#                 surface_clearance = max(center_clearance - float(args.robot_radius), 0.0)
#                 geometric_collision = center_clearance <= float(args.robot_radius)
#             else:
#                 surface_clearance = float("nan")
#                 geometric_collision = False
#             rotation = quaternion.from_rotation_vector(np.asarray([0.0, yaw, 0.0]))
#             pose = np.asarray(
#                 [x, y, z, rotation.x, rotation.y, rotation.z, rotation.w], dtype=np.float32
#             )

#             if args.archive_observations:
#                 rows["rgb"].append(rgb)
#                 rows["depth"].append(depth)
#                 rows["goal_mask"].append(goal_mask)
#                 rows["live_goal_mask"].append(live_goal_mask)
#                 rows["ghost_goal_mask"].append(ghost_goal_mask)
#                 rows["obstacle_mask"].append(obstacle_mask)
#             rows["pose"].append(pose)
#             rows["action_3d"].append(action)
#             rows["point_goal"].append(planner_goal)
#             rows["belief_goal_mu"].append(belief_goal_mu)
#             rows["belief_goal_covariance"].append(belief_goal_covariance)
#             rows["belief_goal_pixel"].append(belief_goal_pixel)
#             rows["belief_goal_visible"].append(belief_goal_visible)
#             rows["belief_goal_source"].append(belief_goal_source)
#             rows["belief_goal_time_since_seen"].append(belief_goal_time_since_seen)
#             rows["belief_goal_bearing_rad"].append(belief_goal_bearing)
#             rows["belief_goal_pixel_sigma"].append(belief_goal_pixel_sigma)
#             rows["belief_heading_recovery_active"].append(belief_recovery_active)
#             rows["selected_trajectory"].append(result.trajectory)
#             rows["all_trajectories"].append(result.all_trajectories)
#             rows["all_values"].append(result.all_values)
#             rows["selected_index"].append(result.selected_index)
#             rows["fallback_stop"].append(result.fallback_stop)
#             rows["escape_turn"].append(result.escape_turn)
#             rows["valid_obstacle_points"].append(result.valid_obstacle_points)
#             rows["selected_circulation_sign"].append(result.selected_circulation_sign)
#             rows["candidate_circulation_signs"].append(
#                 result.candidate_circulation_signs
#             )
#             rows["selected_barrier_energy"].append(result.selected_barrier_energy)
#             rows["selected_circulation_energy"].append(
#                 result.selected_circulation_energy
#             )
#             rows["planning_time_seconds"].append(planning_time)
#             rows["selected_minimum_clearance"].append(result.selected_minimum_clearance)
#             rows["mean_guidance_noise_correction"].append(
#                 result.mean_guidance_noise_correction
#             )
#             rows["final_guidance_noise_correction"].append(
#                 result.final_guidance_noise_correction
#             )
#             rows["maximum_guidance_noise_correction"].append(
#                 result.maximum_guidance_noise_correction
#             )
#             rows["mean_final_effective_sample_size"].append(
#                 result.mean_final_effective_sample_size
#             )
#             rows["goal_distance"].append(goal_distance)
#             rows["executed_center_clearance"].append(center_clearance)
#             rows["executed_surface_clearance"].append(surface_clearance)
#             rows["geometric_collision"].append(geometric_collision)
#             rows["obstacle_positions_world"].append(
#                 np.stack(mesh_current_centroids)
#                 if mesh_current_centroids
#                 else np.zeros((0, 3), dtype=np.float64)
#             )

#             rows["qwen_homotopy_sign"].append(forced_circulation_sign)
#             rows["qwen_homotopy_side"].append(
#                 homotopy_decision.side if homotopy_decision is not None else "AUTO"
#             )
#             rows["qwen_homotopy_confidence"].append(
#                 homotopy_decision.confidence if homotopy_decision is not None else 0.0
#             )
#             rows["qwen_homotopy_queried"].append(
#                 homotopy_decision.queried_qwen if homotopy_decision is not None else False
#             )

#             if args.save_frames and step % max(int(args.save_every), 1) == 0:

#                 side_label = (
#                     homotopy_decision.side
#                     if homotopy_decision is not None
#                     else "AUTO"
#                 )
#                 label = (
#                     f"t={step} goal={goal_distance:.2f}m qwen_side={side_label} pixels={len(obstacle_pixels)} "
#                     f"goal_src={belief_goal_source} "
#                     f"goal_sigma_px={belief_goal_pixel_sigma:.1f} "
#                     f"recover={int(belief_recovery_active)} "
#                     f"pred={result.selected_minimum_clearance:.2f}m "
#                     f"actual={surface_clearance:.2f}m "
#                     f"mode={result.selected_circulation_sign:+.0f} "
#                     f"escape={int(result.escape_turn)} "
#                     f"guide_rms={result.mean_guidance_noise_correction:.4f} "
#                     f"v={action[0]:.2f} w={action[2]:.2f}"
#                 )
#                 frame = overlay_frame(
#                     rgb,
#                     goal_mask,
#                     obstacle_mask,
#                     label,
#                     show_masks=args.overlay_masks,
#                 )
#                 frame.save(frame_directory / f"frame_{step:04d}.png")
#                 video_frames.append(frame)

#             print(
#                 f"step={step:04d} goal={goal_distance:.2f}m "
#                 f"qwen_side={homotopy_decision.side if homotopy_decision else 'AUTO'} "
#                 f"goal_src={belief_goal_source} "
#                 f"goal_sigma_px={belief_goal_pixel_sigma:.1f} "
#                 f"recover={int(belief_recovery_active)} "
#                 f"pixels={len(obstacle_pixels)} valid={result.valid_obstacle_points} "
#                 f"selected={result.selected_index} fallback={result.fallback_stop} "
#                 f"escape={result.escape_turn} mode={result.selected_circulation_sign:+.0f} "
#                 f"pred_clear={result.selected_minimum_clearance:.3f}m "
#                 f"actual_clear={surface_clearance:.3f}m "
#                 f"collision={geometric_collision} "
#                 f"barrier={result.selected_barrier_energy:.5f} "
#                 f"circ={result.selected_circulation_energy:.5f} "
#                 f"latency={planning_time * 1000.0:.1f}ms "
#                 f"guide_rms={result.mean_guidance_noise_correction:.6f} "
#                 f"ess={result.mean_final_effective_sample_size:.2f} "
#                 f"action={action.tolist()}",
#                 flush=True,
#             )
#             if goal_distance <= args.stop_distance:
#                 success = True
#                 break

#         if not rows["goal_distance"]:
#             raise RuntimeError("rollout produced no steps")
#         rollout_path = output_directory / "rollout.npz"
#         np.savez_compressed(
#             rollout_path,
#             **{
#                 key: np.stack(values)
#                 if isinstance(values[0], np.ndarray)
#                 else np.asarray(values)
#                 for key, values in rows.items()
#             },
#             goal_position=goal,
#             obstacle_position=(
#                 mesh_centroids[0]
#                 if mesh_centroids
#                 else (
#                     ghost
#                     if ghost is not None
#                     else np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)
#                 )
#             ),
#             obstacle_positions=(
#                 np.stack(mesh_centroids)
#                 if mesh_centroids
#                 else np.zeros((0, 3), dtype=np.float32)
#             ),
#             obstacle_velocity_xz=mesh_velocities,
#             success=np.asarray(success),
#             hz=np.asarray(args.hz, dtype=np.float32),
#             start_position_xz=start_position_xz,
#             initial_goal_distance=np.asarray(initial_goal_distance, dtype=np.float64),
#             stop_distance=np.asarray(args.stop_distance, dtype=np.float64),
#             robot_radius=np.asarray(args.robot_radius, dtype=np.float64),
#             evaluation_layout=np.asarray(args.evaluation_layout),
#             seed=np.asarray(args.seed, dtype=np.int64),
#             goal_mode=np.asarray(args.goal_mode),
#             belief_pixel_goal=np.asarray(args.belief_pixel_goal),
#         )
#         with (output_directory / "manifest.json").open("w", encoding="utf-8") as file:
#             json.dump(
#                 {
#                     "success": success,
#                     "steps": len(rows["goal_distance"]),
#                     "archived_observations": args.archive_observations,
#                     "final_goal_distance": float(rows["goal_distance"][-1]),
#                     "planner": "released_navdp_s2diff_pixels",
#                     "controller": "direct_waypoint_no_optimizer",
#                     "qwen_role": "obstacle_homotopy_only",
#                     "qwen_process_isolated_from_habitat": True,
#                     "qwen_creates_goal_or_action": False,
#                     "qwen_homotopy": args.qwen_homotopy,
#                     "qwen_homotopy_events": homotopy_events,
#                     "qwen_homotopy_forces_all_candidates": args.qwen_homotopy,
#                     "homotopy_sign_convention": {"LEFT": -1.0, "RIGHT": 1.0},
#                     "homotopy_minimum_obstacle_pixels": args.homotopy_minimum_obstacle_pixels,
#                     "homotopy_release_clear_frames": args.homotopy_release_clear_frames,
#                     "homotopy_consistency_repeats": args.homotopy_consistency_repeats,
#                     "uses_velocity_chunk": False,
#                     "obstacle_mode": args.obstacle_mode,
#                     "obstacle_world_xz": args.obstacle_world_xz,
#                     "goal_mesh": args.goal_mesh,
#                     "particle_anchor": args.particle_anchor,
#                     "particle_energy_reweighting": args.particle_energy_reweighting,
#                     "particle_collision_mask": args.particle_collision_mask,
#                     "goal_mode": args.goal_mode,
#                     "belief_pixel_goal": args.belief_pixel_goal,
#                     "belief_source": "semantic_goal_mask_plus_odometry",
#                     "belief_bootstrap_world_goal": args.belief_bootstrap_world_goal,
#                     "belief_measurement_std": args.belief_measurement_std,
#                     "belief_translation_process_std": args.belief_translation_process_std,
#                     "belief_yaw_process_std_deg": args.belief_yaw_process_std_deg,
#                     "belief_covariance_controls_navdp_mask_size": False,
#                     "belief_heading_recovery": args.belief_heading_recovery,
#                     "belief_recovery_obstacle_gated": True,
#                     "belief_recovery_bearing_deg": args.belief_recovery_bearing_deg,
#                     "belief_recovery_maximum_yaw_rate": args.belief_recovery_maximum_yaw_rate,
#                     "belief_recovery_maximum_forward_speed": args.belief_recovery_maximum_forward_speed,
#                     "particle_noise_schedule": args.particle_noise_schedule,
#                     "progressive_guidance": args.progressive_guidance,
#                     "mesh_obstacle_count": len(mesh_centroids),
#                     "moving_obstacles": bool(np.any(np.abs(mesh_velocities) > 0.0)),
#                     "obstacle_velocity_xz": mesh_velocities.tolist(),
#                     "evaluation_layout": args.evaluation_layout,
#                     "seed": args.seed,
#                     "robot_radius": args.robot_radius,
#                     "minimum_executed_surface_clearance": (
#                         float(np.nanmin(rows["executed_surface_clearance"]))
#                         if np.any(np.isfinite(rows["executed_surface_clearance"]))
#                         else None
#                     ),
#                     "geometric_collision": bool(
#                         np.any(rows["geometric_collision"])
#                     ),
#                     "rollout": str(rollout_path),
#                 },
#                 file,
#                 indent=2,
#             )
#         if args.save_video and video_frames:
#             save_video(
#                 video_frames,
#                 output_directory / "rollout.mp4",
#                 fps=max(args.hz / max(args.save_every, 1), 1.0),
#             )
#         print(f"Saved rollout: {rollout_path}", flush=True)
#         print(f"Success: {success}", flush=True)
#     finally:
#         if simulator is not None:
#             simulator.close()
#         stop_server(server_process)
#         stop_server(qwen_process)


# if __name__ == "__main__":
#     main()


from __future__ import annotations

import argparse
import io
import json
import math
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import habitat_sim
import numpy as np
import quaternion
import requests
from habitat_sim.agent import AgentConfiguration
from PIL import Image, ImageDraw, ImageFilter

from belief_heading_recovery import belief_heading_recovery_action
from belief_pixel_goal import GaussianGoalBelief

HERE = Path(__file__).resolve().parent
SIZE_X = 50.0
SIZE_Z = 50.0
SIZE_Y = 4.820803273566
MESH_GOAL_ID = 10000
MESH_OBSTACLE_ID = 2


@dataclass(frozen=True)
class NavDPS2DiffOutput:
    trajectory: np.ndarray
    all_trajectories: np.ndarray
    all_values: np.ndarray
    selected_index: int
    fallback_stop: bool
    escape_turn: bool
    valid_obstacle_points: int
    selected_circulation_sign: float
    candidate_circulation_signs: np.ndarray
    selected_barrier_energy: float
    selected_circulation_energy: float
    minimum_clearance: np.ndarray
    selected_minimum_clearance: float
    mean_guidance_noise_correction: float
    final_guidance_noise_correction: float
    maximum_guidance_noise_correction: float
    mean_final_effective_sample_size: float


class NavDPS2DiffClient:
    def __init__(self, server_url: str, timeout: float = 180.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = float(timeout)

    def reset(
        self,
        intrinsic: np.ndarray,
        *,
        stop_threshold: float = -3.0,
        batch_size: int = 1,
    ) -> str:
        intrinsic = np.asarray(intrinsic, dtype=np.float32)
        if intrinsic.shape != (3, 3):
            raise ValueError(f"intrinsic must have shape [3,3], got {intrinsic.shape}")
        response = requests.post(
            f"{self.server_url}/navigator_reset",
            json={
                "intrinsic": intrinsic.tolist(),
                "stop_threshold": float(stop_threshold),
                "batch_size": int(batch_size),
            },
            timeout=self.timeout,
        )
        self._raise_for_error(response)
        return str(response.json().get("algo", ""))

    def plan(
        self,
        *,
        goal_xy: np.ndarray,
        rgb: np.ndarray,
        depth: np.ndarray,
        obstacle_pixels: np.ndarray,
        goal_mode: str = "point",
        forced_circulation_sign: float = 0.0,
    ) -> NavDPS2DiffOutput:
        goal_xy = np.asarray(goal_xy, dtype=np.float32).reshape(-1)
        if goal_xy.shape != (2,):
            raise ValueError(f"goal_xy must have shape [2], got {goal_xy.shape}")
        if goal_mode not in {"point", "pixel"}:
            raise ValueError("goal_mode must be point or pixel")
        forced_circulation_sign = float(forced_circulation_sign)
        if forced_circulation_sign not in {-1.0, 0.0, 1.0}:
            raise ValueError("forced_circulation_sign must be -1, 0, or +1")

        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[-1] < 3:
            raise ValueError(f"rgb must have shape [H,W,3], got {rgb.shape}")
        rgb = rgb[..., :3]

        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.shape != rgb.shape[:2]:
            raise ValueError(
                f"depth/rgb shape mismatch: {depth.shape} vs {rgb.shape[:2]}"
            )

        if goal_mode == "pixel":
            if not np.all(np.isfinite(goal_xy)) or not np.allclose(
                goal_xy, np.round(goal_xy)
            ):
                raise ValueError("PixelGoal must be integer [u,v]")
            goal_xy = np.round(goal_xy).astype(np.int64)
            if not (0 <= goal_xy[0] < rgb.shape[1] and 0 <= goal_xy[1] < rgb.shape[0]):
                raise ValueError("PixelGoal lies outside the RGB image")

        pixels = np.asarray(obstacle_pixels)
        if pixels.size == 0:
            pixels = np.zeros((0, 2), dtype=np.int32)
        else:
            pixels = pixels.reshape(-1, 2)
            if not np.all(np.isfinite(pixels)):
                raise ValueError("obstacle pixels must be finite")
            if not np.allclose(pixels, np.round(pixels)):
                raise ValueError("obstacle pixels must be integer [u,v] coordinates")
            pixels = np.round(pixels).astype(np.int32)

        rgb_bytes = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(rgb_bytes, format="JPEG", quality=95)
        depth_u16 = np.clip(depth * 10000.0, 0.0, 65535.0).astype(np.uint16)
        depth_bytes = io.BytesIO()
        Image.fromarray(depth_u16).save(depth_bytes, format="PNG")

        endpoint = "pixelgoal_step" if goal_mode == "pixel" else "pointgoal_step"
        response = requests.post(
            f"{self.server_url}/{endpoint}",
            files={
                "image": ("image.jpg", rgb_bytes.getvalue(), "image/jpeg"),
                "depth": ("depth.png", depth_bytes.getvalue(), "image/png"),
            },
            data={
                "goal_data": json.dumps(
                    {
                        "goal_x": [float(goal_xy[0])],
                        "goal_y": [float(goal_xy[1])],
                        "obstacle_pixels": [pixels.tolist()],
                        "forced_circulation_signs": [forced_circulation_sign],
                    }
                )
            },
            timeout=self.timeout,
        )
        self._raise_for_error(response)
        payload = response.json()
        diagnostics = payload["s2diff"]
        trajectory = np.asarray(payload["trajectory"], dtype=np.float32)
        all_trajectories = np.asarray(payload["all_trajectory"], dtype=np.float32)
        all_values = np.asarray(payload["all_values"], dtype=np.float32)

        return NavDPS2DiffOutput(
            trajectory=trajectory[0],
            all_trajectories=all_trajectories[0],
            all_values=all_values[0],
            selected_index=int(diagnostics["selected_index"][0]),
            fallback_stop=bool(diagnostics["fallback_stop"][0]),
            escape_turn=bool(diagnostics["escape_turn"][0]),
            valid_obstacle_points=int(diagnostics["valid_obstacle_points"][0]),
            selected_circulation_sign=float(
                diagnostics["selected_circulation_sign"][0]
            ),
            candidate_circulation_signs=np.asarray(
                diagnostics["candidate_circulation_signs"][0], dtype=np.float32
            ),
            selected_barrier_energy=float(diagnostics["selected_barrier_energy"][0]),
            selected_circulation_energy=float(
                diagnostics["selected_circulation_energy"][0]
            ),
            minimum_clearance=np.asarray(
                diagnostics["minimum_clearance"][0], dtype=np.float32
            ),
            selected_minimum_clearance=float(
                diagnostics["selected_minimum_clearance"][0]
            ),
            mean_guidance_noise_correction=float(
                diagnostics["mean_guidance_noise_correction"][0]
            ),
            final_guidance_noise_correction=float(
                diagnostics["final_guidance_noise_correction"][0]
            ),
            maximum_guidance_noise_correction=float(
                diagnostics["maximum_guidance_noise_correction"][0]
            ),
            mean_final_effective_sample_size=float(
                diagnostics.get("mean_final_effective_sample_size", [0.0])[0]
            ),
        )

    @staticmethod
    def _raise_for_error(response: requests.Response) -> None:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict) and "error" in payload:
            raise RuntimeError(str(payload["error"]))
        response.raise_for_status()


@dataclass(frozen=True)
class QwenHomotopyDecision:
    side: str
    circulation_sign: float
    confidence: float
    obstacle_relevant: bool
    queried_qwen: bool
    raw_response: Optional[str]
    repeated_sides: tuple[str, ...]
    repeated_confidences: tuple[float, ...]
    consistency_rate: float
    used_fallback: bool


@dataclass(frozen=True)
class QwenCommandDecision:
    command: str
    confidence: float
    raw_response: str


@dataclass(frozen=True)
class QwenMissionPlanDecision:
    plan: tuple[str, ...]
    confidence: float
    raw_response: str


class QwenHomotopyClient:
    """HTTP client for the isolated visual-Qwen process."""

    def __init__(self, server_url: str, timeout: float = 300.0) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout = float(timeout)

    def reset(self) -> None:
        response = requests.post(f"{self.server_url}/reset", timeout=self.timeout)
        self._raise_for_error(response)

    def step(
        self, overlaid_rgb: np.ndarray, obstacle_mask: np.ndarray
    ) -> QwenHomotopyDecision:
        image_bytes = io.BytesIO()
        Image.fromarray(np.asarray(overlaid_rgb, dtype=np.uint8)).save(
            image_bytes, format="PNG"
        )
        mask_bytes = io.BytesIO()
        Image.fromarray((np.asarray(obstacle_mask) > 0).astype(np.uint8) * 255).save(
            mask_bytes, format="PNG"
        )
        response = requests.post(
            f"{self.server_url}/select",
            files={
                "image": ("overlay.png", image_bytes.getvalue(), "image/png"),
                "obstacle_mask": ("mask.png", mask_bytes.getvalue(), "image/png"),
            },
            timeout=self.timeout,
        )
        self._raise_for_error(response)
        payload = response.json()
        return QwenHomotopyDecision(
            side=str(payload["side"]),
            circulation_sign=float(payload["circulation_sign"]),
            confidence=float(payload["confidence"]),
            obstacle_relevant=bool(payload["obstacle_relevant"]),
            queried_qwen=bool(payload["queried_qwen"]),
            raw_response=payload.get("raw_response"),
            repeated_sides=tuple(payload.get("repeated_sides", [])),
            repeated_confidences=tuple(
                float(value) for value in payload.get("repeated_confidences", [])
            ),
            consistency_rate=float(payload.get("consistency_rate", 1.0)),
            used_fallback=bool(payload.get("used_fallback", False)),
        )

    def classify_command(
        self, image_rgb: np.ndarray, user_command: str
    ) -> QwenCommandDecision:
        image_bytes = io.BytesIO()
        Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).save(
            image_bytes, format="PNG"
        )
        response = requests.post(
            f"{self.server_url}/command",
            files={"image": ("frame.png", image_bytes.getvalue(), "image/png")},
            data={"command": str(user_command)},
            timeout=self.timeout,
        )
        self._raise_for_error(response)
        payload = response.json()
        return QwenCommandDecision(
            command=str(payload["command"]).upper(),
            confidence=float(payload["confidence"]),
            raw_response=str(payload["raw_response"]),
        )

    def classify_mission(
        self, image_rgb: np.ndarray, user_command: str
    ) -> QwenMissionPlanDecision:
        image_bytes = io.BytesIO()
        Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).save(
            image_bytes, format="PNG"
        )
        response = requests.post(
            f"{self.server_url}/mission",
            files={"image": ("frame.png", image_bytes.getvalue(), "image/png")},
            data={"command": str(user_command)},
            timeout=self.timeout,
        )
        self._raise_for_error(response)
        payload = response.json()
        return QwenMissionPlanDecision(
            plan=tuple(str(item).upper() for item in payload["plan"]),
            confidence=float(payload["confidence"]),
            raw_response=str(payload["raw_response"]),
        )

    @staticmethod
    def _raise_for_error(response: requests.Response) -> None:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict) and "error" in payload:
            raise RuntimeError(str(payload["error"]))
        response.raise_for_status()


def port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def wait_for_server(
    process: subprocess.Popen[Any], host: str, port: int, timeout: float
) -> None:
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"NavDP/S2Diff server exited with code {process.returncode}"
            )
        if port_is_open(host, port):
            return
        time.sleep(1.0)
    raise TimeoutError(f"NavDP server did not open port {port} within {timeout}s")


def stop_server(process: Optional[subprocess.Popen[Any]]) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def start_qwen_homotopy_server(
    args: argparse.Namespace,
) -> Optional[subprocess.Popen[Any]]:
    if not args.qwen_homotopy or not args.start_qwen_homotopy_server:
        return None
    if port_is_open(args.qwen_homotopy_host, args.qwen_homotopy_port):
        raise RuntimeError(
            f"Qwen homotopy port {args.qwen_homotopy_port} is already in use; "
            "pass --no-start-qwen-homotopy-server to use an existing service"
        )
    server_file = HERE / "qwen_homotopy_server.py"
    if not server_file.is_file():
        raise FileNotFoundError(f"Qwen homotopy server not found: {server_file}")
    command = [
        str(args.qwen_homotopy_python),
        str(server_file),
        "--host",
        str(args.qwen_homotopy_host),
        "--port",
        str(args.qwen_homotopy_port),
        "--model-id",
        str(args.qwen_model_id),
        "--device",
        str(args.qwen_device),
        "--minimum-obstacle-pixels",
        str(args.homotopy_minimum_obstacle_pixels),
        "--release-clear-frames",
        str(args.homotopy_release_clear_frames),
        "--consistency-repeats",
        str(args.homotopy_consistency_repeats),
    ]
    print("[qwen-server]", " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=str(HERE))
    wait_for_server(
        process,
        args.qwen_homotopy_host,
        args.qwen_homotopy_port,
        args.qwen_homotopy_timeout,
    )
    return process


def start_server(args: argparse.Namespace) -> Optional[subprocess.Popen[Any]]:
    if not args.start_server:
        return None
    if port_is_open(args.server_host, args.server_port):
        raise RuntimeError(
            f"port {args.server_port} is already in use; use --no-start-server "
            "to connect to an existing guided server"
        )

    navdp_root = Path(args.navdp_root).expanduser().resolve()
    checkpoint = Path(args.navdp_checkpoint).expanduser().resolve()
    server_dir = navdp_root / "baselines" / "navdp"
    server_file = server_dir / "navdp_s2diff_server.py"
    if not server_file.is_file():
        raise FileNotFoundError(f"guided server not found: {server_file}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"NavDP checkpoint not found: {checkpoint}")

    command = [
        str(args.navdp_python),
        str(server_file),
        "--checkpoint",
        str(checkpoint),
        "--device",
        str(args.navdp_device),
        "--planner-mode",
        str(args.planner_mode),
        "--seed",
        str(args.seed),
        "--port",
        str(args.server_port),
        "--candidates",
        str(args.candidates),
        "--particles",
        str(args.particles),
        "--particle-std",
        str(args.particle_std),
        "--gradient-steps",
        str(args.gradient_steps),
        "--gradient-step-size",
        str(args.gradient_step_size),
        "--guidance-strength",
        str(args.guidance_strength),
        "--temperature",
        str(args.temperature),
        "--safe-distance",
        str(args.safe_distance),
        "--hard-collision-distance",
        str(args.hard_collision_distance),
        "--robot-radius",
        str(args.robot_radius),
        "--safety-weight",
        str(args.safety_weight),
        "--barrier-weight",
        str(args.barrier_weight),
        "--barrier-rate",
        str(args.barrier_rate),
        "--circulation-weight",
        str(args.circulation_weight),
        "--circulation-activation-distance",
        str(args.circulation_activation_distance),
        "--circulation-activation-sharpness",
        str(args.circulation_activation_sharpness),
        "--minimum-circulation-progress",
        str(args.minimum_circulation_progress),
        "--blocking-alignment-threshold",
        str(args.blocking_alignment_threshold),
        "--circulation-switch-weight",
        str(args.circulation_switch_weight),
        "--escape-lateral-target",
        str(args.escape_lateral_target),
        "--minimum-obstacle-depth",
        str(args.minimum_obstacle_depth),
        "--maximum-obstacle-depth",
        str(args.maximum_obstacle_depth),
        "--maximum-obstacle-pixels",
        str(args.maximum_obstacle_pixels),
    ]
    particle_flags = {
        "particle-anchor": args.particle_anchor,
        "particle-energy-reweighting": args.particle_energy_reweighting,
        "particle-collision-mask": args.particle_collision_mask,
        "particle-noise-schedule": args.particle_noise_schedule,
        "progressive-guidance": args.progressive_guidance,
    }
    for name, enabled in particle_flags.items():
        command.append(f"--{name}" if enabled else f"--no-{name}")
    command.append("--remove-critic" if args.remove_critic else "--no-remove-critic")
    print("[server]", " ".join(command), flush=True)
    process = subprocess.Popen(command, cwd=str(server_dir))
    wait_for_server(process, args.server_host, args.server_port, args.server_timeout)
    return process


def bilinear_grid(grid: np.ndarray, px: float, py: float) -> float:
    height, width = grid.shape
    x0 = int(np.floor(px))
    y0 = int(np.floor(py))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    tx = px - x0
    ty = py - y0
    top = float(grid[y0, x0]) * (1.0 - tx) + float(grid[y0, x1]) * tx
    bottom = float(grid[y1, x0]) * (1.0 - tx) + float(grid[y1, x1]) * tx
    return top * (1.0 - ty) + bottom * ty


class TerrainHeight:
    def __init__(
        self,
        *,
        mode: str,
        heightmap: Optional[Path],
        obj: Optional[Path],
        flat_y: float,
        size_x: float,
        size_z: float,
        size_y: float,
        flip_x: bool,
        flip_z: bool,
        swap_xz: bool,
    ):
        if mode == "auto":
            mode = (
                "heightmap"
                if heightmap and heightmap.exists()
                else ("obj" if obj and obj.exists() else "flat")
            )
        self.mode = mode
        self.flat_y = float(flat_y)
        self.size_x = float(size_x)
        self.size_z = float(size_z)
        self.size_y = float(size_y)
        self.flip_x = bool(flip_x)
        self.flip_z = bool(flip_z)
        self.swap_xz = bool(swap_xz)
        self.height: Optional[np.ndarray] = None
        self.obj_xs: Optional[np.ndarray] = None
        self.obj_zs: Optional[np.ndarray] = None
        self.obj_h: Optional[np.ndarray] = None

        if mode == "heightmap":
            if heightmap is None or not heightmap.exists():
                raise FileNotFoundError(f"heightmap not found: {heightmap}")
            array = np.asarray(Image.open(heightmap))
            if array.ndim == 3:
                array = array[..., 0]
            array = array.astype(np.float32)
            array = (array - array.min()) / max(float(array.max() - array.min()), 1e-8)
            self.height = array * self.size_y - float(np.mean(array * self.size_y))
        elif mode == "obj":
            if obj is None or not obj.exists():
                raise FileNotFoundError(f"terrain OBJ not found: {obj}")
            vertices = []
            with obj.open("r", encoding="utf-8", errors="ignore") as file:
                for line in file:
                    if line.startswith("v "):
                        parts = line.split()
                        if len(parts) >= 4:
                            vertices.append(tuple(float(value) for value in parts[1:4]))
            if not vertices:
                raise RuntimeError(f"no vertices found in {obj}")
            array = np.asarray(vertices, dtype=np.float32)
            xs = np.unique(array[:, 0])
            zs = np.unique(array[:, 1])
            grid = np.full((len(zs), len(xs)), np.nan, dtype=np.float32)
            x_index = {float(value): index for index, value in enumerate(xs.tolist())}
            z_index = {float(value): index for index, value in enumerate(zs.tolist())}
            for x, z, height in array:
                grid[z_index[float(z)], x_index[float(x)]] = height
            self.obj_xs = xs
            self.obj_zs = zs
            self.obj_h = np.nan_to_num(grid, nan=float(np.nanmean(grid)))
        elif mode != "flat":
            raise ValueError(f"unknown terrain mode: {mode}")

    def _map(self, x: float, z: float) -> tuple[float, float]:
        if self.swap_xz:
            x, z = z, x
        u = (x + self.size_x / 2.0) / self.size_x
        v = (z + self.size_z / 2.0) / self.size_z
        if self.flip_x:
            u = 1.0 - u
        if self.flip_z:
            v = 1.0 - v
        return float(np.clip(u, 0.0, 1.0)), float(np.clip(v, 0.0, 1.0))

    def __call__(self, x: float, z: float) -> float:
        if self.mode == "flat":
            return self.flat_y
        if self.mode == "heightmap":
            assert self.height is not None
            u, v = self._map(x, z)
            return bilinear_grid(
                self.height,
                u * (self.height.shape[1] - 1),
                v * (self.height.shape[0] - 1),
            )
        assert (
            self.obj_xs is not None
            and self.obj_zs is not None
            and self.obj_h is not None
        )
        xx = float(np.clip(x, self.obj_xs[0], self.obj_xs[-1]))
        zz = float(np.clip(z, self.obj_zs[0], self.obj_zs[-1]))
        column = int(
            np.clip(np.searchsorted(self.obj_xs, xx) - 1, 0, len(self.obj_xs) - 2)
        )
        row = int(
            np.clip(np.searchsorted(self.obj_zs, zz) - 1, 0, len(self.obj_zs) - 2)
        )
        x0, x1 = float(self.obj_xs[column]), float(self.obj_xs[column + 1])
        z0, z1 = float(self.obj_zs[row]), float(self.obj_zs[row + 1])
        tx = 0.0 if abs(x1 - x0) < 1e-8 else (xx - x0) / (x1 - x0)
        tz = 0.0 if abs(z1 - z0) < 1e-8 else (zz - z0) / (z1 - z0)
        top = (
            float(self.obj_h[row, column]) * (1.0 - tx)
            + float(self.obj_h[row, column + 1]) * tx
        )
        bottom = (
            float(self.obj_h[row + 1, column]) * (1.0 - tx)
            + float(self.obj_h[row + 1, column + 1]) * tx
        )
        return top * (1.0 - tz) + bottom * tz

    def local_height_max(
        self, x: float, z: float, radius: float, samples: int = 5
    ) -> float:
        if radius <= 1e-6:
            return float(self(x, z))
        values = [
            float(self(x + dx, z + dz))
            for dx in np.linspace(-radius, radius, samples)
            for dz in np.linspace(-radius, radius, samples)
            if dx * dx + dz * dz <= radius * radius + 1e-8
        ]
        return max(values) if values else float(self(x, z))


def make_sensor(
    uuid: str, sensor_type: Any, height: int, width: int, hfov_deg: float
) -> habitat_sim.CameraSensorSpec:
    specification = habitat_sim.CameraSensorSpec()
    specification.uuid = uuid
    specification.sensor_type = sensor_type
    specification.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    specification.resolution = [int(height), int(width)]
    specification.position = [0.0, 0.0, 0.0]
    specification.hfov = float(hfov_deg)
    return specification


def make_simulator(
    scene: Path,
    height: int,
    width: int,
    hfov_deg: float,
    *,
    with_semantic: bool,
):
    simulator_configuration = habitat_sim.SimulatorConfiguration()
    simulator_configuration.scene_id = str(scene.expanduser().resolve())
    simulator_configuration.enable_physics = False
    sensors = [
        make_sensor("rgb", habitat_sim.SensorType.COLOR, height, width, hfov_deg),
        make_sensor("depth", habitat_sim.SensorType.DEPTH, height, width, hfov_deg),
    ]
    if with_semantic:
        sensors.append(
            make_sensor(
                "semantic", habitat_sim.SensorType.SEMANTIC, height, width, hfov_deg
            )
        )
    agent_configuration = AgentConfiguration()
    agent_configuration.sensor_specifications = sensors
    return habitat_sim.Simulator(
        habitat_sim.Configuration(simulator_configuration, [agent_configuration])
    )


def set_agent_pose(agent: Any, position: np.ndarray, yaw: float) -> None:
    state = agent.get_state()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = quaternion.from_rotation_vector(np.asarray([0.0, yaw, 0.0]))
    agent.set_state(state)


def rgb_depth(observation: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(observation["rgb"])
    if rgb.ndim == 3 and rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    depth = np.asarray(observation["depth"], dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return rgb.astype(np.uint8), depth.astype(np.float32)


def semantic_from_observation(observation: dict[str, np.ndarray]) -> np.ndarray:
    semantic = np.asarray(observation["semantic"])
    if semantic.ndim == 3:
        semantic = semantic[..., 0]
    return semantic.astype(np.int32)


def pixel_to_world(
    u: float,
    v: float,
    depth: float,
    position: np.ndarray,
    yaw: float,
    intrinsic: np.ndarray,
) -> np.ndarray:
    right = (u - float(intrinsic[0, 2])) * depth / float(intrinsic[0, 0])
    up = -(v - float(intrinsic[1, 2])) * depth / float(intrinsic[1, 1])
    forward_vector = np.asarray([-math.sin(yaw), 0.0, -math.cos(yaw)])
    right_vector = np.asarray([math.cos(yaw), 0.0, -math.sin(yaw)])
    return (
        np.asarray(position, dtype=np.float64)
        + depth * forward_vector
        + right * right_vector
        + up * np.asarray([0.0, 1.0, 0.0])
    )


def depth_patch_mesh(
    u_center: float,
    v_center: float,
    half_size: int,
    stride: int,
    depth: np.ndarray,
    position: np.ndarray,
    yaw: float,
    intrinsic: np.ndarray,
    *,
    lift: float,
    maximum_depth_jump: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    columns = list(
        range(
            max(0, int(u_center - half_size)),
            min(width, int(u_center + half_size) + 1),
            max(int(stride), 1),
        )
    )
    rows = list(
        range(
            max(0, int(v_center - half_size)),
            min(height, int(v_center + half_size) + 1),
            max(int(stride), 1),
        )
    )
    indices = -np.ones((len(rows), len(columns)), dtype=np.int64)
    depths = np.full((len(rows), len(columns)), np.nan, dtype=np.float32)
    vertices: list[tuple[float, float, float]] = []
    for row_index, v in enumerate(rows):
        for column_index, u in enumerate(columns):
            metric_depth = float(depth[v, u])
            if not np.isfinite(metric_depth) or metric_depth <= 0.1:
                continue
            indices[row_index, column_index] = len(vertices)
            depths[row_index, column_index] = metric_depth
            point = pixel_to_world(
                u, v, metric_depth, position, yaw, intrinsic
            ) + float(lift) * np.asarray([0.0, 1.0, 0.0])
            vertices.append(tuple(float(value) for value in point))

    faces: list[tuple[int, int, int]] = []
    for row_index in range(len(rows) - 1):
        for column_index in range(len(columns) - 1):
            a = int(indices[row_index, column_index])
            b = int(indices[row_index, column_index + 1])
            c = int(indices[row_index + 1, column_index])
            d = int(indices[row_index + 1, column_index + 1])
            if min(a, b, c, d) < 0:
                continue
            cell_depths = (
                depths[row_index, column_index],
                depths[row_index, column_index + 1],
                depths[row_index + 1, column_index],
                depths[row_index + 1, column_index + 1],
            )
            if max(cell_depths) - min(cell_depths) > maximum_depth_jump:
                continue
            faces.append((a, c, d))
            faces.append((a, d, b))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def save_obj(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    diffuse_rgb: Optional[tuple[float, float, float]] = None,
) -> None:
    material_name = None
    if diffuse_rgb is not None:
        red, green, blue = (float(value) for value in diffuse_rgb)
        if not all(0.0 <= value <= 1.0 for value in (red, green, blue)):
            raise ValueError("OBJ diffuse material values must be in [0, 1]")
        material_name = "mesh_material"
        material_path = path.with_suffix(".mtl")
        with material_path.open("w", encoding="utf-8") as material:
            material.write(f"newmtl {material_name}\n")
            material.write(
                f"Ka {0.25 * red:.4f} {0.25 * green:.4f} {0.25 * blue:.4f}\n"
            )
            material.write(f"Kd {red:.4f} {green:.4f} {blue:.4f}\n")
            material.write("Ks 0.1000 0.1000 0.1000\n")
            material.write("Ns 24.0000\n")
            material.write("d 1.0000\n")
            material.write("illum 2\n")
    with path.open("w", encoding="utf-8") as file:
        if material_name is not None:
            file.write(f"mtllib {path.with_suffix('.mtl').name}\n")
            file.write(f"usemtl {material_name}\n")
        for x, y, z in vertices:
            file.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            file.write(f"f {a + 1} {b + 1} {c + 1}\n")


def register_semantic_mesh(simulator: Any, mesh_path: Path, semantic_id: int) -> Any:
    template_manager = simulator.get_object_template_manager()
    object_manager = simulator.get_rigid_object_manager()
    template = template_manager.create_new_template(str(mesh_path))
    template.render_asset_handle = str(mesh_path)
    template.collision_asset_handle = str(mesh_path)
    template.is_collidable = False
    template_id = template_manager.register_template(
        template, f"s2diff_obstacle_{semantic_id}_{os.path.basename(mesh_path)}"
    )
    object_handle = template_manager.get_template_handle_by_id(template_id)
    obstacle = object_manager.add_object_by_template_handle(object_handle)
    obstacle.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    obstacle.collidable = False
    obstacle.semantic_id = int(semantic_id)
    return obstacle


def parse_world_xz(specification: str) -> tuple[float, float]:
    values = [float(value) for value in str(specification).split(",")]
    if len(values) != 2 or not np.isfinite(values).all():
        raise ValueError(
            f"world mesh position must be finite X,Z, got {specification!r}"
        )
    return values[0], values[1]


def world_box_mesh(
    center_x: float,
    base_y: float,
    center_z: float,
    half_extent: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a closed axis-aligned box whose vertices are in world coordinates."""

    if half_extent <= 0.0 or height <= 0.0:
        raise ValueError("box half extent and height must be positive")
    x0, x1 = center_x - half_extent, center_x + half_extent
    z0, z1 = center_z - half_extent, center_z + half_extent
    y0, y1 = base_y, base_y + height
    vertices = np.asarray(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y0, z1],
            [x0, y0, z1],
            [x0, y1, z0],
            [x1, y1, z0],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def place_world_obstacle_meshes(
    simulator: Any,
    terrain: Any,
    xz_specifications: Sequence[str],
    output_directory: Path,
    *,
    half_extent: float,
    height: float,
) -> tuple[list[Any], list[np.ndarray], list[np.ndarray]]:
    """Place static obstacle boxes at exact world X,Z coordinates."""

    mesh_directory = output_directory / "meshes"
    mesh_directory.mkdir(parents=True, exist_ok=True)
    objects: list[Any] = []
    centroids: list[np.ndarray] = []
    geometries: list[np.ndarray] = []
    for index, specification in enumerate(xz_specifications):
        center_x, center_z = parse_world_xz(specification)
        base_y = terrain.local_height_max(center_x, center_z, half_extent)
        vertices, faces = world_box_mesh(
            center_x, base_y, center_z, half_extent, height
        )
        mesh_path = mesh_directory / f"world_obstacle_{index}.obj"
        save_obj(mesh_path, vertices, faces, diffuse_rgb=(0.78, 0.16, 0.06))
        semantic_id = MESH_OBSTACLE_ID + index
        objects.append(register_semantic_mesh(simulator, mesh_path, semantic_id))
        centroid = vertices.mean(axis=0).astype(np.float32)
        centroids.append(centroid)
        geometries.append(vertices[faces][:, :, [0, 2]].astype(np.float64))
        print(
            f"[world-mesh] obstacle={index} semantic_id={semantic_id} "
            f"center_xz={[center_x, center_z]} half_extent={half_extent:.3f} "
            f"height={height:.3f}",
            flush=True,
        )
    return objects, centroids, geometries


def place_world_goal_mesh(
    simulator: Any,
    terrain: Any,
    goal_x: float,
    goal_z: float,
    output_directory: Path,
    *,
    half_extent: float,
    height: float,
) -> Any:
    """Place a visible, non-obstacle semantic goal marker at the exact goal."""

    base_y = terrain.local_height_max(goal_x, goal_z, half_extent)
    vertices, faces = world_box_mesh(goal_x, base_y, goal_z, half_extent, height)
    mesh_directory = output_directory / "meshes"
    mesh_directory.mkdir(parents=True, exist_ok=True)
    mesh_path = mesh_directory / "goal_marker.obj"
    save_obj(mesh_path, vertices, faces, diffuse_rgb=(0.08, 0.85, 0.18))
    goal_object = register_semantic_mesh(simulator, mesh_path, MESH_GOAL_ID)
    print(
        f"[world-mesh] goal semantic_id={MESH_GOAL_ID} "
        f"center_xz={[goal_x, goal_z]}",
        flush=True,
    )
    return goal_object


def parse_uv_fraction(
    specification: str, width: int, height: int
) -> tuple[float, float]:
    u_fraction, v_fraction = (float(value) for value in str(specification).split(","))
    if not (0.0 <= u_fraction <= 1.0 and 0.0 <= v_fraction <= 1.0):
        raise ValueError(f"mesh pixel fraction must be in [0,1], got {specification!r}")
    return u_fraction * width, v_fraction * height


def place_obstacle_meshes(
    simulator: Any,
    depth: np.ndarray,
    position: np.ndarray,
    yaw: float,
    intrinsic: np.ndarray,
    uv_specifications: Sequence[str],
    output_directory: Path,
    *,
    mesh_half_pixels: int,
    mesh_lift: float,
) -> tuple[list[Any], list[np.ndarray], list[np.ndarray]]:
    mesh_directory = output_directory / "meshes"
    mesh_directory.mkdir(parents=True, exist_ok=True)
    height, width = depth.shape
    objects: list[Any] = []
    centroids: list[np.ndarray] = []
    geometries: list[np.ndarray] = []
    for index, specification in enumerate(uv_specifications):
        u, v = parse_uv_fraction(specification, width, height)
        vertices, faces = depth_patch_mesh(
            u,
            v,
            mesh_half_pixels,
            2,
            depth,
            position,
            yaw,
            intrinsic,
            lift=mesh_lift,
        )
        if len(vertices) == 0 or len(faces) == 0:
            raise RuntimeError(
                f"obstacle mesh {index} at {specification!r} has no valid depth surface"
            )
        mesh_path = mesh_directory / f"obstacle_{index}.obj"
        save_obj(mesh_path, vertices, faces)
        semantic_id = MESH_OBSTACLE_ID + index
        objects.append(register_semantic_mesh(simulator, mesh_path, semantic_id))
        centroid = vertices.mean(axis=0).astype(np.float32)
        centroids.append(centroid)
        geometries.append(vertices[faces][:, :, [0, 2]].astype(np.float64))
        print(
            f"[mesh] obstacle={index} semantic_id={semantic_id} "
            f"pixels={specification} vertices={len(vertices)} "
            f"world={centroid.tolist()}",
            flush=True,
        )
    return objects, centroids, geometries


def planar_mesh_clearance(
    point_xz: np.ndarray,
    geometries: Sequence[np.ndarray],
) -> float:
    """Minimum 2-D distance from a robot center to projected mesh triangles."""
    point = np.asarray(point_xz, dtype=np.float64)
    best = float("inf")
    for triangles in geometries:
        triangles = np.asarray(triangles, dtype=np.float64)
        if triangles.size == 0:
            continue
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        v0, v1, v2 = b - a, c - a, point[None, :] - a
        denominator = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        valid = np.abs(denominator) > 1.0e-12
        safe_denominator = np.where(valid, denominator, 1.0)
        u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / safe_denominator
        v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / safe_denominator
        if np.any(valid & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)):
            return 0.0

        starts = np.concatenate((a, b, c), axis=0)
        ends = np.concatenate((b, c, a), axis=0)
        segments = ends - starts
        squared_lengths = np.einsum("ij,ij->i", segments, segments)
        numerators = np.einsum("ij,ij->i", point[None, :] - starts, segments)
        fractions = np.divide(
            numerators,
            squared_lengths,
            out=np.zeros_like(numerators),
            where=squared_lengths > 1.0e-12,
        )
        fractions = np.clip(fractions, 0.0, 1.0)
        closest = starts + fractions[:, None] * segments
        best = min(best, float(np.linalg.norm(point[None, :] - closest, axis=1).min()))
    return best


def parse_xz_velocity(specification: str) -> np.ndarray:
    values = [float(value) for value in str(specification).split(",")]
    if len(values) != 2 or not np.all(np.isfinite(values)):
        raise ValueError("obstacle velocity must be finite vx,vz")
    return np.asarray(values, dtype=np.float64)


def expand_obstacle_velocities(
    specifications: Sequence[str], obstacle_count: int
) -> np.ndarray:
    if obstacle_count == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if not specifications:
        return np.zeros((obstacle_count, 2), dtype=np.float64)
    velocities = np.stack([parse_xz_velocity(item) for item in specifications])
    if len(velocities) == 1 and obstacle_count > 1:
        velocities = np.repeat(velocities, obstacle_count, axis=0)
    if len(velocities) != obstacle_count:
        raise ValueError(
            "provide one obstacle velocity to broadcast or one velocity per mesh"
        )
    return velocities


def translated_mesh_geometry(
    base_geometries: Sequence[np.ndarray],
    base_centroids: Sequence[np.ndarray],
    velocities_xz: np.ndarray,
    elapsed_seconds: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    geometries: list[np.ndarray] = []
    centroids: list[np.ndarray] = []
    for geometry, centroid, velocity in zip(
        base_geometries, base_centroids, velocities_xz
    ):
        offset_xz = np.asarray(velocity, dtype=np.float64) * elapsed_seconds
        geometries.append(np.asarray(geometry) + offset_xz[None, None, :])
        offset_xyz = np.asarray([offset_xz[0], 0.0, offset_xz[1]])
        centroids.append(np.asarray(centroid, dtype=np.float64) + offset_xyz)
    return geometries, centroids


def move_mesh_objects(
    objects: Sequence[Any], velocities_xz: np.ndarray, elapsed_seconds: float
) -> None:
    for obstacle, velocity in zip(objects, velocities_xz):
        dx, dz = np.asarray(velocity, dtype=np.float64) * elapsed_seconds
        vector_type = type(obstacle.translation)
        obstacle.translation = vector_type(float(dx), 0.0, float(dz))


def camera_coordinates(
    point: np.ndarray, position: np.ndarray, yaw: float
) -> tuple[float, float, float]:
    delta = np.asarray(point, dtype=np.float32) - np.asarray(position, dtype=np.float32)
    forward_x, forward_z = -math.sin(yaw), -math.cos(yaw)
    left_x, left_z = -math.cos(yaw), math.sin(yaw)
    forward = forward_x * float(delta[0]) + forward_z * float(delta[2])
    left = left_x * float(delta[0]) + left_z * float(delta[2])
    return -left, float(delta[1]), forward


def camera_intrinsic(height: int, width: int, hfov_deg: float) -> np.ndarray:
    hfov = math.radians(float(hfov_deg))
    focal = (width * 0.5) / max(math.tan(hfov * 0.5), 1e-6)
    return np.asarray(
        [
            [focal, 0.0, (width - 1) * 0.5],
            [0.0, focal, (height - 1) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def world_goal_to_pixel(
    point: np.ndarray,
    position: np.ndarray,
    yaw: float,
    intrinsic: np.ndarray,
    height: int,
    width: int,
) -> np.ndarray:
    """Project a world goal to a valid PixelGoal, clamping off-screen bearings."""

    right, up, forward = camera_coordinates(point, position, yaw)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    margin = 11
    bearing = math.atan2(right, forward)
    maximum_bearing = math.atan2(max(cx - margin, 1.0), fx)
    bearing = float(np.clip(bearing, -maximum_bearing, maximum_bearing))
    u = cx + fx * math.tan(bearing)
    v = cy - fy * up / forward if forward > 0.05 else 0.62 * height
    return np.asarray(
        [
            int(np.clip(round(u), margin, width - margin - 1)),
            int(np.clip(round(v), margin, height - margin - 1)),
        ],
        dtype=np.int32,
    )


def circle_mask(height: int, width: int, u: float, v: float, radius: int) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    return (((xx - u) ** 2 + (yy - v) ** 2) <= radius**2).astype(np.uint8)


def project_world_mask(
    point: np.ndarray,
    position: np.ndarray,
    yaw: float,
    intrinsic: np.ndarray,
    height: int,
    width: int,
    radius: int,
) -> tuple[np.ndarray, float]:
    right, up, forward = camera_coordinates(point, position, yaw)
    if forward <= 0.05:
        return np.zeros((height, width), dtype=np.uint8), forward
    u = float(intrinsic[0, 2] + intrinsic[0, 0] * right / forward)
    v = float(intrinsic[1, 2] - intrinsic[1, 1] * up / forward)
    if not (radius <= u < width - radius and radius <= v < height - radius):
        return np.zeros((height, width), dtype=np.uint8), forward
    return circle_mask(height, width, u, v, radius), forward


def depth_obstacle_mask(
    depth: np.ndarray, threshold: float, minimum_y_fraction: float
) -> np.ndarray:
    mask = np.isfinite(depth) & (depth > 0.05) & (depth < float(threshold))
    mask[: int(depth.shape[0] * minimum_y_fraction)] = False
    return mask.astype(np.uint8)


def pixels_from_mask(mask: np.ndarray, maximum: int) -> np.ndarray:
    v, u = np.nonzero(np.asarray(mask) > 0)
    if u.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    pixels = np.stack((u, v), axis=-1).astype(np.int32)
    if maximum > 0 and len(pixels) > maximum:
        indices = np.linspace(0, len(pixels) - 1, maximum).astype(np.int64)
        pixels = pixels[indices]
    return pixels


def waypoint_action(
    trajectory: np.ndarray,
    *,
    lookahead_index: int,
    maximum_forward_speed: float,
    maximum_yaw_rate: float,
    yaw_gain: float,
) -> np.ndarray:
    trajectory = np.asarray(trajectory, dtype=np.float32)
    if trajectory.ndim != 2 or trajectory.shape[0] == 0 or trajectory.shape[1] < 2:
        return np.zeros(3, dtype=np.float32)
    if np.max(np.linalg.norm(trajectory[:, :2], axis=-1)) < 1e-5:
        return np.zeros(3, dtype=np.float32)
    index = int(np.clip(lookahead_index, 0, trajectory.shape[0] - 1))
    forward, left = float(trajectory[index, 0]), float(trajectory[index, 1])
    bearing = math.atan2(left, max(forward, 1e-4))
    velocity = maximum_forward_speed * max(0.0, math.cos(bearing))
    yaw_rate = float(np.clip(yaw_gain * bearing, -maximum_yaw_rate, maximum_yaw_rate))
    return np.asarray([velocity, 0.0, yaw_rate], dtype=np.float32)


def integrate_mars(
    position: np.ndarray, yaw: float, action: np.ndarray, dt: float
) -> tuple[np.ndarray, float]:
    forward_velocity, lateral_velocity, yaw_rate = [float(value) for value in action]
    forward_x, forward_z = -math.sin(yaw), -math.cos(yaw)
    left_x, left_z = -math.cos(yaw), math.sin(yaw)
    output = np.asarray(position, dtype=np.float32).copy()
    output[0] += (forward_x * forward_velocity + left_x * lateral_velocity) * dt
    output[2] += (forward_z * forward_velocity + left_z * lateral_velocity) * dt
    return output, yaw + yaw_rate * dt


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def overlay_frame(
    rgb: np.ndarray,
    goal_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    text: str,
    *,
    show_masks: bool,
    detection_box: Optional[np.ndarray] = None,
    detection_label: Optional[str] = None,
) -> Image.Image:
    output = np.asarray(rgb, dtype=np.uint8).copy()
    if show_masks:
        output[goal_mask > 0] = (
            0.35 * output[goal_mask > 0] + 0.65 * np.asarray([0, 255, 0])
        ).astype(np.uint8)
        output[obstacle_mask > 0] = (
            0.35 * output[obstacle_mask > 0] + 0.65 * np.asarray([255, 0, 0])
        ).astype(np.uint8)
    image = Image.fromarray(output)
    draw = ImageDraw.Draw(image)
    if detection_box is not None:
        x1, y1, x2, y2 = [float(value) for value in detection_box]
        draw.rectangle((x1, y1, x2, y2), outline=(255, 255, 0), width=3)
        if detection_label:
            draw.text((x1 + 2, max(y1 - 14, 2)), detection_label, fill=(255, 255, 0))
    draw.rectangle((5, 5, min(image.width - 5, 12 + len(text) * 7), 28), fill=(0, 0, 0))
    draw.text((10, 9), text, fill=(255, 255, 255))
    return image


def dilate_binary_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a binary image mask without adding a SciPy dependency."""

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if radius <= 0 or not np.any(binary):
        return binary
    kernel_size = 2 * int(radius) + 1
    return (
        np.asarray(
            Image.fromarray(binary * 255).filter(ImageFilter.MaxFilter(kernel_size))
        )
        > 0
    ).astype(np.uint8)


def save_video(frames: Sequence[Image.Image], path: Path, fps: float) -> None:
    import imageio.v2 as imageio

    with imageio.get_writer(path, fps=float(fps)) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB")))


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description="One-file released NavDP + in-denoising S2Diff Mars rollout"
    )
    argument_parser.add_argument("--navdp-root", required=True)
    argument_parser.add_argument("--navdp-checkpoint", required=True)
    argument_parser.add_argument("--navdp-python", default=sys.executable)
    argument_parser.add_argument("--navdp-device", default="cuda:0")
    argument_parser.add_argument(
        "--planner-mode", choices=["pure-navdp", "s2diff", "gradient"], default="s2diff"
    )
    argument_parser.add_argument(
        "--goal-mode", choices=["point", "pixel"], default="point"
    )
    argument_parser.add_argument(
        "--belief-pixel-goal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the live semantic goal mask to correct a body-frame Gaussian "
            "belief and its projected mean as NavDP's PixelGoal while occluded."
        ),
    )
    argument_parser.add_argument("--belief-minimum-goal-pixels", type=int, default=10)
    argument_parser.add_argument("--belief-measurement-std", type=float, default=0.05)
    argument_parser.add_argument(
        "--belief-translation-process-std", type=float, default=0.03
    )
    argument_parser.add_argument(
        "--belief-yaw-process-std-deg", type=float, default=1.0
    )
    argument_parser.add_argument(
        "--belief-bootstrap-world-goal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Simulation-only bootstrap when the goal is initially invisible. "
            "Disable for a strict detector-only evaluation."
        ),
    )
    argument_parser.add_argument("--belief-bootstrap-std", type=float, default=0.50)
    argument_parser.add_argument("--belief-ghost-base-radius", type=int, default=10)
    argument_parser.add_argument(
        "--belief-ghost-covariance-scale", type=float, default=2.0
    )
    argument_parser.add_argument("--belief-ghost-maximum-radius", type=int, default=80)
    argument_parser.add_argument(
        "--belief-heading-recovery",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    argument_parser.add_argument(
        "--belief-recovery-bearing-deg", type=float, default=35.0
    )
    argument_parser.add_argument("--belief-recovery-yaw-gain", type=float, default=1.5)
    argument_parser.add_argument(
        "--belief-recovery-maximum-yaw-rate", type=float, default=0.70
    )
    argument_parser.add_argument(
        "--belief-recovery-maximum-forward-speed", type=float, default=0.12
    )
    argument_parser.add_argument(
        "--interactive-return-home",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "At the outward goal, ask for a command, let Qwen classify RETURN "
            "or STOP, and use a separately propagated spawn/home PixelGoal belief."
        ),
    )
    argument_parser.add_argument(
        "--return-command",
        default=None,
        help="Optional non-interactive command text, for example 'come back'.",
    )
    argument_parser.add_argument(
        "--qwen-freeform-mission",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Ask once at startup for a free-form instruction. Qwen emits either "
            "GO_TO_GOAL or GO_TO_GOAL followed by RETURN_HOME."
        ),
    )
    argument_parser.add_argument(
        "--mission-command",
        default=None,
        help=(
            "Optional non-interactive free-form mission, for example "
            "'visit the target and report back'."
        ),
    )
    argument_parser.add_argument(
        "--return-goal-obstacle-activation-distance", type=float, default=1.35
    )
    argument_parser.add_argument(
        "--return-goal-obstacle-dilation-pixels", type=int, default=30
    )
    argument_parser.add_argument(
        "--qwen-model-id", default="Qwen/Qwen2.5-VL-3B-Instruct"
    )
    argument_parser.add_argument("--qwen-device", default="auto")
    argument_parser.add_argument("--qwen-homotopy-python", default=sys.executable)
    argument_parser.add_argument("--qwen-homotopy-host", default="127.0.0.1")
    argument_parser.add_argument("--qwen-homotopy-port", type=int, default=8890)
    argument_parser.add_argument("--qwen-homotopy-timeout", type=float, default=600.0)
    argument_parser.add_argument(
        "--start-qwen-homotopy-server",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    argument_parser.add_argument(
        "--qwen-homotopy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When a metric obstacle becomes relevant, Qwen chooses the single "
            "LEFT/RIGHT circulation sign used by every trajectory candidate."
        ),
    )
    argument_parser.add_argument(
        "--homotopy-minimum-obstacle-pixels", type=int, default=30
    )
    argument_parser.add_argument("--homotopy-release-clear-frames", type=int, default=8)
    argument_parser.add_argument(
        "--homotopy-consistency-repeats",
        type=int,
        default=5,
        help="Repeat Qwen on the identical obstacle frame and use majority vote.",
    )
    argument_parser.add_argument(
        "--remove-critic", action=argparse.BooleanOptionalAction, default=True
    )
    argument_parser.add_argument("--seed", type=int, default=7)
    argument_parser.add_argument(
        "--start-server", action=argparse.BooleanOptionalAction, default=True
    )
    argument_parser.add_argument("--server-host", default="127.0.0.1")
    argument_parser.add_argument("--server-port", type=int, default=8888)
    argument_parser.add_argument("--server-timeout", type=float, default=180.0)
    argument_parser.add_argument("--candidates", type=int, default=16)
    argument_parser.add_argument("--particles", type=int, default=8)
    argument_parser.add_argument("--particle-std", type=float, default=0.22)
    argument_parser.add_argument("--gradient-steps", type=int, default=3)
    argument_parser.add_argument("--gradient-step-size", type=float, default=0.04)
    argument_parser.add_argument("--guidance-strength", type=float, default=0.85)
    argument_parser.add_argument("--temperature", type=float, default=0.35)
    argument_parser.add_argument(
        "--particle-anchor", action=argparse.BooleanOptionalAction, default=True
    )
    argument_parser.add_argument(
        "--particle-energy-reweighting",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    argument_parser.add_argument(
        "--particle-collision-mask", action=argparse.BooleanOptionalAction, default=True
    )
    argument_parser.add_argument(
        "--particle-noise-schedule", action=argparse.BooleanOptionalAction, default=True
    )
    argument_parser.add_argument(
        "--progressive-guidance", action=argparse.BooleanOptionalAction, default=True
    )
    argument_parser.add_argument("--safe-distance", type=float, default=0.42)
    argument_parser.add_argument("--hard-collision-distance", type=float, default=0.24)
    argument_parser.add_argument("--safety-weight", type=float, default=35.0)
    argument_parser.add_argument("--barrier-weight", type=float, default=25.0)
    argument_parser.add_argument("--barrier-rate", type=float, default=0.15)
    argument_parser.add_argument("--circulation-weight", type=float, default=18.0)
    argument_parser.add_argument(
        "--circulation-activation-distance", type=float, default=1.50
    )
    argument_parser.add_argument(
        "--circulation-activation-sharpness", type=float, default=0.20
    )
    argument_parser.add_argument(
        "--minimum-circulation-progress", type=float, default=0.025
    )
    argument_parser.add_argument(
        "--blocking-alignment-threshold", type=float, default=0.25
    )
    argument_parser.add_argument("--circulation-switch-weight", type=float, default=2.0)
    argument_parser.add_argument("--escape-lateral-target", type=float, default=0.35)
    argument_parser.add_argument("--minimum-obstacle-depth", type=float, default=0.10)
    argument_parser.add_argument("--maximum-obstacle-depth", type=float, default=5.0)
    argument_parser.add_argument("--maximum-obstacle-pixels", type=int, default=1536)

    argument_parser.add_argument("--scene", required=True)
    argument_parser.add_argument("--terrain-obj", default=None)
    argument_parser.add_argument("--heightmap", default=None)
    argument_parser.add_argument(
        "--terrain-height-mode",
        choices=["auto", "heightmap", "obj", "flat"],
        default="auto",
    )
    argument_parser.add_argument("--flat-y", type=float, default=0.0)
    argument_parser.add_argument("--size-x", type=float, default=SIZE_X)
    argument_parser.add_argument("--size-z", type=float, default=SIZE_Z)
    argument_parser.add_argument("--size-y", type=float, default=SIZE_Y)
    argument_parser.add_argument("--flip-heightmap-x", action="store_true")
    argument_parser.add_argument(
        "--flip-heightmap-z", action=argparse.BooleanOptionalAction, default=True
    )
    argument_parser.add_argument("--swap-heightmap-xz", action="store_true")
    argument_parser.add_argument("--clearance", type=float, default=1.4)
    argument_parser.add_argument("--pose-terrain-radius", type=float, default=0.8)
    argument_parser.add_argument(
        "--robot-radius",
        type=float,
        default=0.24,
        help="Planar rover footprint radius used by both guidance and evaluation.",
    )
    argument_parser.add_argument(
        "--evaluation-layout",
        default="default",
        help="Stable layout identifier stored in the rollout archive.",
    )

    argument_parser.add_argument("--height", type=int, default=720)
    argument_parser.add_argument("--width", type=int, default=720)
    argument_parser.add_argument("--hfov-deg", type=float, default=90.0)
    argument_parser.add_argument("--hz", type=float, default=10.0)
    argument_parser.add_argument("--max-steps", type=int, default=300)
    argument_parser.add_argument("--stop-distance", type=float, default=1.0)
    argument_parser.add_argument("--start-x", type=float, default=0.0)
    argument_parser.add_argument("--start-z", type=float, default=8.0)
    argument_parser.add_argument("--start-yaw-deg", type=float, default=0.0)
    argument_parser.add_argument("--goal-x", type=float, default=None)
    argument_parser.add_argument("--goal-z", type=float, default=None)
    argument_parser.add_argument("--goal-y", type=float, default=None)
    argument_parser.add_argument("--goal-height", type=float, default=1.2)
    argument_parser.add_argument("--goal-radius", type=int, default=18)
    argument_parser.add_argument(
        "--goal-mesh", action=argparse.BooleanOptionalAction, default=False
    )
    argument_parser.add_argument("--goal-mesh-half-extent", type=float, default=0.25)
    argument_parser.add_argument("--goal-mesh-height", type=float, default=1.50)

    argument_parser.add_argument(
        "--obstacle-mode", choices=["none", "depth", "mesh", "ghost"], default="none"
    )
    argument_parser.add_argument("--obstacle-depth-threshold", type=float, default=1.4)
    argument_parser.add_argument("--obstacle-min-y-fraction", type=float, default=0.45)
    argument_parser.add_argument("--ghost-obstacle-x", type=float, default=None)
    argument_parser.add_argument("--ghost-obstacle-z", type=float, default=None)
    argument_parser.add_argument("--ghost-obstacle-y", type=float, default=None)
    argument_parser.add_argument("--ghost-obstacle-height", type=float, default=0.45)
    argument_parser.add_argument("--ghost-obstacle-radius", type=int, default=24)
    argument_parser.add_argument(
        "--obstacle-mesh-uv",
        nargs="+",
        default=[],
        help=(
            "Actual rendered obstacle mesh locations as image fractions u,v. "
            "Example: --obstacle-mesh-uv 0.50,0.72 0.30,0.68"
        ),
    )
    argument_parser.add_argument(
        "--obstacle-world-xz",
        nargs="*",
        default=[],
        metavar="X,Z",
        help=(
            "Static rendered obstacle-box centers in world X,Z coordinates. "
            "Example: --obstacle-world-xz 0,0. Do not combine with "
            "--obstacle-mesh-uv."
        ),
    )
    argument_parser.add_argument(
        "--obstacle-world-xz-item",
        action="append",
        default=[],
        metavar="X,Z",
        help=(
            "Repeatable form that safely accepts negative coordinates, e.g. "
            "--obstacle-world-xz-item=-3,0."
        ),
    )
    argument_parser.add_argument(
        "--world-obstacle-half-extent", type=float, default=0.75
    )
    argument_parser.add_argument("--world-obstacle-height", type=float, default=1.40)
    argument_parser.add_argument("--mesh-half-pixels", type=int, default=26)
    argument_parser.add_argument("--mesh-obstacle-lift", type=float, default=0.50)
    argument_parser.add_argument(
        "--obstacle-velocity-xz",
        nargs="*",
        default=[],
        metavar="VX,VZ",
        help=(
            "World-frame mesh velocities in m/s. Supply one value to broadcast "
            "or one value per obstacle. Example: --obstacle-velocity-xz 0.30,0.0"
        ),
    )

    argument_parser.add_argument("--lookahead-index", type=int, default=4)
    argument_parser.add_argument("--maximum-forward-speed", type=float, default=0.5)
    argument_parser.add_argument("--maximum-yaw-rate", type=float, default=0.5)
    argument_parser.add_argument("--yaw-gain", type=float, default=1.5)
    argument_parser.add_argument("--output", default="runs/navdp_s2diff_mars")
    argument_parser.add_argument("--save-every", type=int, default=1)
    argument_parser.add_argument(
        "--save-frames", action=argparse.BooleanOptionalAction, default=True
    )
    argument_parser.add_argument(
        "--save-video", action=argparse.BooleanOptionalAction, default=True
    )
    argument_parser.add_argument(
        "--archive-observations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store RGB/depth/masks in rollout.npz; disable for large evaluations.",
    )
    argument_parser.add_argument(
        "--overlay-masks", action=argparse.BooleanOptionalAction, default=True
    )
    return argument_parser


def main() -> None:
    args = parser().parse_args()
    if args.obstacle_world_xz_item:
        args.obstacle_world_xz.extend(args.obstacle_world_xz_item)
    np.random.seed(args.seed)
    if args.goal_x is None or args.goal_z is None:
        raise ValueError("fixed PointGoal requires --goal-x and --goal-z")
    if args.belief_pixel_goal and args.goal_mode != "pixel":
        raise ValueError("--belief-pixel-goal requires --goal-mode pixel")
    if args.belief_pixel_goal and not args.goal_mesh:
        raise ValueError(
            "simulation belief tracking requires --goal-mesh so a live semantic "
            "goal observation exists"
        )
    if args.belief_minimum_goal_pixels < 1:
        raise ValueError("belief-minimum-goal-pixels must be positive")
    return_home_enabled = args.interactive_return_home or args.qwen_freeform_mission
    if return_home_enabled and not args.belief_pixel_goal:
        raise ValueError("return-home modes require --belief-pixel-goal")
    if return_home_enabled and not args.qwen_homotopy:
        raise ValueError("return-home modes require --qwen-homotopy")
    if args.mission_command is not None and not args.qwen_freeform_mission:
        raise ValueError("--mission-command requires --qwen-freeform-mission")
    if args.return_goal_obstacle_activation_distance <= 0.0:
        raise ValueError("return goal obstacle activation distance must be positive")
    if args.return_goal_obstacle_dilation_pixels < 0:
        raise ValueError("return goal obstacle dilation pixels must be non-negative")
    if (
        min(
            args.belief_measurement_std,
            args.belief_translation_process_std,
            args.belief_yaw_process_std_deg,
            args.belief_bootstrap_std,
        )
        < 0.0
    ):
        raise ValueError("belief uncertainty parameters must be non-negative")
    if args.robot_radius < 0.0:
        raise ValueError("robot-radius must be non-negative")
    if args.obstacle_velocity_xz and args.obstacle_mode != "mesh":
        raise ValueError("moving obstacle velocities require --obstacle-mode mesh")
    if args.obstacle_mode == "ghost" and (
        args.ghost_obstacle_x is None or args.ghost_obstacle_z is None
    ):
        raise ValueError(
            "ghost mode requires --ghost-obstacle-x and --ghost-obstacle-z"
        )
    if args.obstacle_mesh_uv and args.obstacle_world_xz:
        raise ValueError(
            "choose either --obstacle-mesh-uv or --obstacle-world-xz, not both"
        )
    if args.obstacle_mode == "mesh" and not (
        args.obstacle_mesh_uv or args.obstacle_world_xz
    ):
        raise ValueError(
            "mesh mode requires --obstacle-world-xz X,Z [X,Z ...] or "
            "--obstacle-mesh-uv u,v [u,v ...]"
        )
    if args.world_obstacle_half_extent <= 0.0 or args.world_obstacle_height <= 0.0:
        raise ValueError("world obstacle dimensions must be positive")
    if args.goal_mesh_half_extent <= 0.0 or args.goal_mesh_height <= 0.0:
        raise ValueError("goal mesh dimensions must be positive")

    if args.qwen_homotopy and args.planner_mode == "pure-navdp":
        raise ValueError("Qwen homotopy conditioning requires s2diff or gradient mode")

    qwen_process: Optional[subprocess.Popen[Any]] = None
    server_process: Optional[subprocess.Popen[Any]] = None
    simulator = None
    try:
        qwen_process = start_qwen_homotopy_server(args)
        homotopy_selector = None
        if args.qwen_homotopy:
            homotopy_selector = QwenHomotopyClient(
                f"http://{args.qwen_homotopy_host}:{args.qwen_homotopy_port}",
                timeout=args.qwen_homotopy_timeout,
            )
            homotopy_selector.reset()
        server_process = start_server(args)
        server_url = f"http://{args.server_host}:{args.server_port}"
        client = NavDPS2DiffClient(server_url)
        algorithm = client.reset(
            camera_intrinsic(args.height, args.width, args.hfov_deg),
            batch_size=1,
            stop_threshold=-3.0,
        )
        supported_algorithms = {
            "navdp-s2diff-pixels",
            "navdp-hlc-s2diff",
            "navdp-hlc-s2diff-no-critic",
            "navdp-hlc-gradient",
            "navdp-hlc-gradient-no-critic",
            "navdp-pure-critic",
        }
        if algorithm not in supported_algorithms:
            raise RuntimeError(f"unexpected planner response: {algorithm!r}")

        terrain = TerrainHeight(
            mode=args.terrain_height_mode,
            heightmap=(
                Path(args.heightmap).expanduser().resolve() if args.heightmap else None
            ),
            obj=(
                Path(args.terrain_obj).expanduser().resolve()
                if args.terrain_obj
                else None
            ),
            flat_y=args.flat_y,
            size_x=args.size_x,
            size_z=args.size_z,
            size_y=args.size_y,
            flip_x=args.flip_heightmap_x,
            flip_z=args.flip_heightmap_z,
            swap_xz=args.swap_heightmap_xz,
        )
        output_directory = Path(args.output).expanduser().resolve()
        frame_directory = output_directory / "frames"
        frame_directory.mkdir(parents=True, exist_ok=True)

        simulator = make_simulator(
            Path(args.scene),
            args.height,
            args.width,
            args.hfov_deg,
            with_semantic=args.obstacle_mode == "mesh" or args.goal_mesh,
        )
        agent = simulator.initialize_agent(0)
        intrinsic = camera_intrinsic(args.height, args.width, args.hfov_deg)
        goal_belief = (
            GaussianGoalBelief(
                intrinsic,
                (args.height, args.width),
                minimum_visible_pixels=args.belief_minimum_goal_pixels,
                measurement_std=args.belief_measurement_std,
                translation_process_std=args.belief_translation_process_std,
                yaw_process_std=math.radians(args.belief_yaw_process_std_deg),
            )
            if args.belief_pixel_goal
            else None
        )
        previous_executed_action = np.zeros(3, dtype=np.float32)
        home_belief = (
            GaussianGoalBelief(
                intrinsic,
                (args.height, args.width),
                minimum_visible_pixels=args.belief_minimum_goal_pixels,
                measurement_std=args.belief_measurement_std,
                translation_process_std=args.belief_translation_process_std,
                yaw_process_std=math.radians(args.belief_yaw_process_std_deg),
            )
            if return_home_enabled
            else None
        )
        if home_belief is not None:
            # Home starts at the rover origin. Every executed action propagates
            # this stationary world location into the current body frame.
            home_belief.initialize(
                np.zeros(2, dtype=np.float32),
                args.belief_measurement_std,
                visible=False,
            )
        mission_phase = "OUTBOUND"
        return_goal_obstacle_active = False
        return_command_event: Optional[dict[str, Any]] = None
        mission_plan_event: Optional[dict[str, Any]] = None
        mission_plan_pending = bool(args.qwen_freeform_mission)
        automatic_return_requested = False
        roundtrip_completed = False
        x, z = float(args.start_x), float(args.start_z)
        yaw = math.radians(float(args.start_yaw_deg))
        dt = 1.0 / float(args.hz)

        goal_y = args.goal_y
        if goal_y is None:
            goal_y = (
                terrain.local_height_max(args.goal_x, args.goal_z, 0.8)
                + args.goal_height
            )
        goal = np.asarray([args.goal_x, goal_y, args.goal_z], dtype=np.float32)
        start_position_xz = np.asarray([x, z], dtype=np.float64)
        initial_goal_distance = float(
            np.linalg.norm(goal[[0, 2]].astype(np.float64) - start_position_xz)
        )
        goal_mesh_object = None
        if args.goal_mesh:
            goal_mesh_object = place_world_goal_mesh(
                simulator,
                terrain,
                args.goal_x,
                args.goal_z,
                output_directory,
                half_extent=args.goal_mesh_half_extent,
                height=args.goal_mesh_height,
            )

        ghost = None
        if args.obstacle_mode == "ghost":
            ghost_y = args.ghost_obstacle_y
            if ghost_y is None:
                ghost_y = (
                    terrain.local_height_max(
                        args.ghost_obstacle_x,
                        args.ghost_obstacle_z,
                        args.pose_terrain_radius,
                    )
                    + args.ghost_obstacle_height
                )
            ghost = np.asarray(
                [args.ghost_obstacle_x, ghost_y, args.ghost_obstacle_z],
                dtype=np.float32,
            )

        mesh_objects: list[Any] = []
        mesh_centroids: list[np.ndarray] = []
        mesh_current_centroids: list[np.ndarray] = []
        mesh_base_geometries: list[np.ndarray] = []
        mesh_geometries: list[np.ndarray] = []
        mesh_velocities = np.zeros((0, 2), dtype=np.float64)
        mesh_placed = False
        if args.obstacle_mode == "mesh" and args.obstacle_world_xz:
            mesh_objects, mesh_centroids, mesh_base_geometries = (
                place_world_obstacle_meshes(
                    simulator,
                    terrain,
                    args.obstacle_world_xz,
                    output_directory,
                    half_extent=args.world_obstacle_half_extent,
                    height=args.world_obstacle_height,
                )
            )
            mesh_velocities = expand_obstacle_velocities(
                args.obstacle_velocity_xz, len(mesh_objects)
            )
            mesh_geometries, mesh_current_centroids = translated_mesh_geometry(
                mesh_base_geometries, mesh_centroids, mesh_velocities, 0.0
            )
            mesh_placed = True

        row_keys = [
            "pose",
            "action_3d",
            "mission_phase",
            "return_goal_obstacle_active",
            "point_goal",
            "belief_goal_mu",
            "belief_goal_covariance",
            "belief_goal_pixel",
            "belief_goal_visible",
            "belief_goal_source",
            "belief_goal_time_since_seen",
            "belief_goal_bearing_rad",
            "belief_goal_pixel_sigma",
            "belief_heading_recovery_active",
            "selected_trajectory",
            "all_trajectories",
            "all_values",
            "selected_index",
            "fallback_stop",
            "escape_turn",
            "valid_obstacle_points",
            "selected_circulation_sign",
            "candidate_circulation_signs",
            "selected_barrier_energy",
            "selected_circulation_energy",
            "planning_time_seconds",
            "selected_minimum_clearance",
            "mean_guidance_noise_correction",
            "final_guidance_noise_correction",
            "maximum_guidance_noise_correction",
            "mean_final_effective_sample_size",
            "goal_distance",
            "executed_center_clearance",
            "executed_surface_clearance",
            "geometric_collision",
            "obstacle_positions_world",
            "qwen_homotopy_sign",
            "qwen_homotopy_side",
            "qwen_homotopy_confidence",
            "qwen_homotopy_queried",
        ]
        if args.archive_observations:
            row_keys.extend(
                (
                    "rgb",
                    "depth",
                    "goal_mask",
                    "live_goal_mask",
                    "ghost_goal_mask",
                    "obstacle_mask",
                )
            )
        rows: dict[str, list[Any]] = {key: [] for key in row_keys}
        video_frames: list[Image.Image] = []
        success = False
        homotopy_events: list[dict[str, Any]] = []

        for step in range(int(args.max_steps)):
            y = (
                terrain.local_height_max(x, z, args.pose_terrain_radius)
                + args.clearance
            )
            position = np.asarray([x, y, z], dtype=np.float32)
            if goal_belief is not None and step > 0:
                goal_belief.predict(previous_executed_action, dt)
            if home_belief is not None and step > 0:
                home_belief.predict(previous_executed_action, dt)
            set_agent_pose(agent, position, yaw)
            if mesh_placed:
                elapsed_seconds = step * dt
                move_mesh_objects(mesh_objects, mesh_velocities, elapsed_seconds)
                mesh_geometries, mesh_current_centroids = translated_mesh_geometry(
                    mesh_base_geometries,
                    mesh_centroids,
                    mesh_velocities,
                    elapsed_seconds,
                )
            observation = simulator.get_sensor_observations()
            rgb, depth = rgb_depth(observation)

            if args.obstacle_mode == "mesh" and not mesh_placed:
                mesh_objects, mesh_centroids, mesh_base_geometries = (
                    place_obstacle_meshes(
                        simulator,
                        depth,
                        position,
                        yaw,
                        intrinsic,
                        args.obstacle_mesh_uv,
                        output_directory,
                        mesh_half_pixels=args.mesh_half_pixels,
                        mesh_lift=args.mesh_obstacle_lift,
                    )
                )
                mesh_velocities = expand_obstacle_velocities(
                    args.obstacle_velocity_xz, len(mesh_objects)
                )
                mesh_geometries, mesh_current_centroids = translated_mesh_geometry(
                    mesh_base_geometries, mesh_centroids, mesh_velocities, 0.0
                )
                mesh_placed = True
                observation = simulator.get_sensor_observations()
                rgb, depth = rgb_depth(observation)

            if mission_plan_pending:
                mission_command = args.mission_command
                if mission_command is None:
                    print(
                        "\nWhat should the rover do? You may use a vague command, "
                        "for example: 'visit the goal and come back'.",
                        flush=True,
                    )
                    try:
                        mission_command = input("> ").strip()
                    except EOFError as error:
                        raise RuntimeError(
                            "no startup command was available; set "
                            "--mission-command for a non-interactive run"
                        ) from error
                assert homotopy_selector is not None
                mission_decision = homotopy_selector.classify_mission(
                    rgb, mission_command
                )
                automatic_return_requested = mission_decision.plan == (
                    "GO_TO_GOAL",
                    "RETURN_HOME",
                )
                mission_plan_event = {
                    "step": step,
                    "user_command": mission_command,
                    "plan": list(mission_decision.plan),
                    "confidence": mission_decision.confidence,
                    "raw_response": mission_decision.raw_response,
                }
                Image.fromarray(rgb).save(output_directory / "qwen_mission_frame.png")
                print(
                    f"[qwen-mission] text={mission_command!r} "
                    f"plan={list(mission_decision.plan)} "
                    f"confidence={mission_decision.confidence:.2f}",
                    flush=True,
                )
                mission_plan_pending = False

            semantic = (
                semantic_from_observation(observation)
                if args.obstacle_mode == "mesh" or args.goal_mesh
                else None
            )
            goal_right, _goal_up, goal_forward = camera_coordinates(goal, position, yaw)
            point_goal = np.asarray(
                [max(goal_forward, 0.0), -goal_right], dtype=np.float32
            )
            live_goal_mask = np.zeros(depth.shape, dtype=np.uint8)
            ghost_goal_mask = np.zeros(depth.shape, dtype=np.uint8)
            belief_goal_visible = False
            belief_goal_source = "DISABLED"
            belief_goal_mu = np.full(2, np.nan, dtype=np.float32)
            belief_goal_covariance = np.full((2, 2), np.nan, dtype=np.float32)
            belief_goal_pixel = np.full(2, -1, dtype=np.int32)
            belief_goal_time_since_seen = float("nan")
            belief_goal_bearing = float("nan")
            belief_goal_pixel_sigma = float("nan")

            if goal_belief is not None:
                assert semantic is not None
                live_goal_mask = (semantic == MESH_GOAL_ID).astype(np.uint8)
                bootstrapped = False
                if mission_phase == "OUTBOUND":
                    active_goal_belief = goal_belief
                    belief_goal_visible = goal_belief.observe(live_goal_mask, depth)
                    if not goal_belief.initialized:
                        if not args.belief_bootstrap_world_goal:
                            raise RuntimeError(
                                "goal belief is uninitialized because the live goal "
                                "mask has not been observed; start with the goal visible "
                                "or pass --belief-bootstrap-world-goal for simulation"
                            )
                        goal_belief.initialize(
                            np.asarray([goal_forward, -goal_right], dtype=np.float32),
                            args.belief_bootstrap_std,
                        )
                        bootstrapped = True
                else:
                    assert home_belief is not None and home_belief.initialized
                    active_goal_belief = home_belief
                    belief_goal_visible = False

                belief_projection = active_goal_belief.project(
                    base_radius=args.belief_ghost_base_radius,
                    covariance_scale=args.belief_ghost_covariance_scale,
                    maximum_radius=args.belief_ghost_maximum_radius,
                )
                planner_goal = belief_projection.pixel_uv
                ghost_goal_mask = belief_projection.mask
                if mission_phase == "OUTBOUND" and belief_goal_visible:
                    goal_mask = live_goal_mask
                    belief_goal_source = "LIVE"
                else:
                    goal_mask = ghost_goal_mask
                    belief_goal_source = (
                        "HOME_BELIEF"
                        if mission_phase == "RETURN_HOME"
                        else ("WORLD_BOOTSTRAP" if bootstrapped else "GHOST")
                    )
                assert (
                    active_goal_belief.mu is not None
                    and active_goal_belief.Sigma is not None
                )
                belief_goal_mu = active_goal_belief.mu.copy()
                belief_goal_covariance = active_goal_belief.Sigma.copy()
                belief_goal_pixel = belief_projection.pixel_uv.copy()
                belief_goal_time_since_seen = active_goal_belief.time_since_seen
                belief_goal_bearing = belief_projection.bearing_rad
                belief_goal_pixel_sigma = belief_projection.pixel_sigma
            else:
                if args.goal_mesh:
                    assert semantic is not None
                    live_goal_mask = (semantic == MESH_GOAL_ID).astype(np.uint8)
                    goal_mask = live_goal_mask
                    if not np.any(goal_mask):
                        goal_mask, _ = project_world_mask(
                            goal,
                            position,
                            yaw,
                            intrinsic,
                            args.height,
                            args.width,
                            args.goal_radius,
                        )
                else:
                    goal_mask, _ = project_world_mask(
                        goal,
                        position,
                        yaw,
                        intrinsic,
                        args.height,
                        args.width,
                        args.goal_radius,
                    )
                planner_goal = point_goal
            if args.goal_mode == "pixel" and goal_belief is None:
                planner_goal = world_goal_to_pixel(
                    goal, position, yaw, intrinsic, args.height, args.width
                )
                goal_mask = circle_mask(
                    args.height,
                    args.width,
                    planner_goal[0],
                    planner_goal[1],
                    args.goal_radius,
                )
            guidance_depth = depth.copy()
            if args.obstacle_mode == "depth":
                obstacle_mask = depth_obstacle_mask(
                    depth, args.obstacle_depth_threshold, args.obstacle_min_y_fraction
                )
            elif args.obstacle_mode == "mesh":
                assert semantic is not None
                semantic_ids = list(
                    range(
                        MESH_OBSTACLE_ID,
                        MESH_OBSTACLE_ID + len(mesh_objects),
                    )
                )
                obstacle_mask = np.isin(semantic, semantic_ids).astype(np.uint8)
                # The depth image was re-rendered after mesh placement, so
                # guidance_depth already contains the real obstacle depth.
            elif args.obstacle_mode == "ghost":
                assert ghost is not None
                obstacle_mask, obstacle_forward = project_world_mask(
                    ghost,
                    position,
                    yaw,
                    intrinsic,
                    args.height,
                    args.width,
                    args.ghost_obstacle_radius,
                )
                if obstacle_forward > 0.05:
                    guidance_depth[obstacle_mask > 0] = obstacle_forward
            else:
                obstacle_mask = np.zeros(depth.shape, dtype=np.uint8)

            # Replace this mask-to-pixels line with your own detector's [u,v]
            if mission_phase == "RETURN_HOME":
                distance_from_reached_goal = float(
                    np.linalg.norm(goal[[0, 2]] - position[[0, 2]])
                )
                if (
                    not return_goal_obstacle_active
                    and distance_from_reached_goal
                    >= args.return_goal_obstacle_activation_distance
                ):
                    return_goal_obstacle_active = True
                    print(
                        "[roundtrip] reached-goal keep-out is now active",
                        flush=True,
                    )
                if return_goal_obstacle_active and np.any(live_goal_mask):
                    reached_goal_keepout = dilate_binary_mask(
                        live_goal_mask,
                        args.return_goal_obstacle_dilation_pixels,
                    )
                    obstacle_mask = (
                        (obstacle_mask > 0) | (reached_goal_keepout > 0)
                    ).astype(np.uint8)
                    target_depths = depth[
                        (live_goal_mask > 0)
                        & np.isfinite(depth)
                        & (depth > args.minimum_obstacle_depth)
                    ]
                    if target_depths.size:
                        guidance_depth[reached_goal_keepout > 0] = float(
                            np.median(target_depths)
                        )

            # array if obstacle pixels already come directly from your system.
            obstacle_pixels = pixels_from_mask(
                obstacle_mask, args.maximum_obstacle_pixels
            )
            homotopy_decision = None
            forced_circulation_sign = 0.0
            obstacle_relevant_for_homotopy = False
            if homotopy_selector is not None:
                homotopy_obstacle_mask = (
                    (obstacle_mask > 0)
                    & np.isfinite(guidance_depth)
                    & (guidance_depth >= args.minimum_obstacle_depth)
                    & (guidance_depth <= args.maximum_obstacle_depth)
                ).astype(np.uint8)
                qwen_overlay = overlay_frame(
                    rgb,
                    goal_mask,
                    homotopy_obstacle_mask,
                    "Qwen homotopy: choose LEFT or RIGHT",
                    show_masks=True,
                )
                homotopy_decision = homotopy_selector.step(
                    np.asarray(qwen_overlay.convert("RGB")), homotopy_obstacle_mask
                )
                obstacle_relevant_for_homotopy = homotopy_decision.obstacle_relevant
                forced_circulation_sign = homotopy_decision.circulation_sign
                if homotopy_decision.queried_qwen:
                    event = {
                        "step": step,
                        "side": homotopy_decision.side,
                        "circulation_sign": forced_circulation_sign,
                        "confidence": homotopy_decision.confidence,
                        "repeat_sides": list(homotopy_decision.repeated_sides),
                        "repeat_confidences": list(
                            homotopy_decision.repeated_confidences
                        ),
                        "consistency_rate": homotopy_decision.consistency_rate,
                        "used_fallback": homotopy_decision.used_fallback,
                        "raw_response": homotopy_decision.raw_response,
                    }
                    homotopy_events.append(event)
                    query_directory = output_directory / "qwen_homotopy_queries"
                    query_directory.mkdir(parents=True, exist_ok=True)
                    qwen_overlay.save(query_directory / f"query_step_{step:04d}.png")
                    print(
                        f"[qwen-homotopy] side={homotopy_decision.side} "
                        f"sign={forced_circulation_sign:+.0f} "
                        f"confidence={homotopy_decision.confidence:.2f} "
                        f"consistency={homotopy_decision.consistency_rate:.2%} "
                        f"repeats={list(homotopy_decision.repeated_sides)} "
                        f"fallback={homotopy_decision.used_fallback}",
                        flush=True,
                    )
            planning_start = time.perf_counter()
            result = client.plan(
                goal_xy=planner_goal,
                rgb=rgb,
                depth=guidance_depth,
                obstacle_pixels=obstacle_pixels,
                goal_mode=args.goal_mode,
                forced_circulation_sign=forced_circulation_sign,
            )
            planning_time = time.perf_counter() - planning_start
            action = (
                np.zeros(3, dtype=np.float32)
                if result.fallback_stop
                else waypoint_action(
                    result.trajectory,
                    lookahead_index=args.lookahead_index,
                    maximum_forward_speed=args.maximum_forward_speed,
                    maximum_yaw_rate=args.maximum_yaw_rate,
                    yaw_gain=args.yaw_gain,
                )
            )
            action, belief_recovery_active = belief_heading_recovery_action(
                action,
                belief_bearing=belief_goal_bearing,
                obstacle_relevant=obstacle_relevant_for_homotopy,
                enabled=args.belief_pixel_goal and args.belief_heading_recovery,
                activation_bearing=math.radians(args.belief_recovery_bearing_deg),
                yaw_gain=args.belief_recovery_yaw_gain,
                maximum_yaw_rate=args.belief_recovery_maximum_yaw_rate,
                maximum_forward_speed=args.belief_recovery_maximum_forward_speed,
            )

            next_position, next_yaw = integrate_mars(position, yaw, action, dt)
            previous_executed_action = action.copy()
            x = float(
                np.clip(
                    next_position[0], -args.size_x / 2.0 + 0.5, args.size_x / 2.0 - 0.5
                )
            )
            z = float(
                np.clip(
                    next_position[2], -args.size_z / 2.0 + 0.5, args.size_z / 2.0 - 0.5
                )
            )
            rows["mission_phase"].append(mission_phase)
            rows["return_goal_obstacle_active"].append(return_goal_obstacle_active)

            yaw = wrap_angle(next_yaw)
            outbound_goal_distance = float(
                np.linalg.norm(goal[[0, 2]] - np.asarray([x, z]))
            )
            home_distance = float(
                np.linalg.norm(start_position_xz - np.asarray([x, z]))
            )
            goal_distance = (
                home_distance
                if mission_phase == "RETURN_HOME"
                else outbound_goal_distance
            )
            center_clearance = planar_mesh_clearance(
                np.asarray([x, z], dtype=np.float64), mesh_geometries
            )
            if np.isfinite(center_clearance):
                surface_clearance = max(
                    center_clearance - float(args.robot_radius), 0.0
                )
                geometric_collision = center_clearance <= float(args.robot_radius)
            else:
                surface_clearance = float("nan")
                geometric_collision = False
            rotation = quaternion.from_rotation_vector(np.asarray([0.0, yaw, 0.0]))
            pose = np.asarray(
                [x, y, z, rotation.x, rotation.y, rotation.z, rotation.w],
                dtype=np.float32,
            )

            if args.archive_observations:
                rows["rgb"].append(rgb)
                rows["depth"].append(depth)
                rows["goal_mask"].append(goal_mask)
                rows["live_goal_mask"].append(live_goal_mask)
                rows["ghost_goal_mask"].append(ghost_goal_mask)
                rows["obstacle_mask"].append(obstacle_mask)
            rows["pose"].append(pose)
            rows["action_3d"].append(action)
            rows["point_goal"].append(planner_goal)
            rows["belief_goal_mu"].append(belief_goal_mu)
            rows["belief_goal_covariance"].append(belief_goal_covariance)
            rows["belief_goal_pixel"].append(belief_goal_pixel)
            rows["belief_goal_visible"].append(belief_goal_visible)
            rows["belief_goal_source"].append(belief_goal_source)
            rows["belief_goal_time_since_seen"].append(belief_goal_time_since_seen)
            rows["belief_goal_bearing_rad"].append(belief_goal_bearing)
            rows["belief_goal_pixel_sigma"].append(belief_goal_pixel_sigma)
            rows["belief_heading_recovery_active"].append(belief_recovery_active)
            rows["selected_trajectory"].append(result.trajectory)
            rows["all_trajectories"].append(result.all_trajectories)
            rows["all_values"].append(result.all_values)
            rows["selected_index"].append(result.selected_index)
            rows["fallback_stop"].append(result.fallback_stop)
            rows["escape_turn"].append(result.escape_turn)
            rows["valid_obstacle_points"].append(result.valid_obstacle_points)
            rows["selected_circulation_sign"].append(result.selected_circulation_sign)
            rows["candidate_circulation_signs"].append(
                result.candidate_circulation_signs
            )
            rows["selected_barrier_energy"].append(result.selected_barrier_energy)
            rows["selected_circulation_energy"].append(
                result.selected_circulation_energy
            )
            rows["planning_time_seconds"].append(planning_time)
            rows["selected_minimum_clearance"].append(result.selected_minimum_clearance)
            rows["mean_guidance_noise_correction"].append(
                result.mean_guidance_noise_correction
            )
            rows["final_guidance_noise_correction"].append(
                result.final_guidance_noise_correction
            )
            rows["maximum_guidance_noise_correction"].append(
                result.maximum_guidance_noise_correction
            )
            rows["mean_final_effective_sample_size"].append(
                result.mean_final_effective_sample_size
            )
            rows["goal_distance"].append(goal_distance)
            rows["executed_center_clearance"].append(center_clearance)
            rows["executed_surface_clearance"].append(surface_clearance)
            rows["geometric_collision"].append(geometric_collision)
            rows["obstacle_positions_world"].append(
                np.stack(mesh_current_centroids)
                if mesh_current_centroids
                else np.zeros((0, 3), dtype=np.float64)
            )

            rows["qwen_homotopy_sign"].append(forced_circulation_sign)
            rows["qwen_homotopy_side"].append(
                homotopy_decision.side if homotopy_decision is not None else "AUTO"
            )
            rows["qwen_homotopy_confidence"].append(
                homotopy_decision.confidence if homotopy_decision is not None else 0.0
            )
            rows["qwen_homotopy_queried"].append(
                homotopy_decision.queried_qwen
                if homotopy_decision is not None
                else False
            )

            if args.save_frames and step % max(int(args.save_every), 1) == 0:

                side_label = (
                    homotopy_decision.side if homotopy_decision is not None else "AUTO"
                )
                label = (
                    f"t={step} phase={mission_phase} goal={goal_distance:.2f}m "
                    f"qwen_side={side_label} pixels={len(obstacle_pixels)} "
                    f"goal_src={belief_goal_source} "
                    f"goal_sigma_px={belief_goal_pixel_sigma:.1f} "
                    f"recover={int(belief_recovery_active)} "
                    f"pred={result.selected_minimum_clearance:.2f}m "
                    f"actual={surface_clearance:.2f}m "
                    f"mode={result.selected_circulation_sign:+.0f} "
                    f"escape={int(result.escape_turn)} "
                    f"guide_rms={result.mean_guidance_noise_correction:.4f} "
                    f"v={action[0]:.2f} w={action[2]:.2f}"
                )
                frame = overlay_frame(
                    rgb,
                    goal_mask,
                    obstacle_mask,
                    label,
                    show_masks=args.overlay_masks,
                )
                frame.save(frame_directory / f"frame_{step:04d}.png")
                video_frames.append(frame)

            print(
                f"step={step:04d} phase={mission_phase} goal={goal_distance:.2f}m "
                f"qwen_side={homotopy_decision.side if homotopy_decision else 'AUTO'} "
                f"goal_src={belief_goal_source} "
                f"goal_sigma_px={belief_goal_pixel_sigma:.1f} "
                f"recover={int(belief_recovery_active)} "
                f"pixels={len(obstacle_pixels)} valid={result.valid_obstacle_points} "
                f"selected={result.selected_index} fallback={result.fallback_stop} "
                f"escape={result.escape_turn} mode={result.selected_circulation_sign:+.0f} "
                f"pred_clear={result.selected_minimum_clearance:.3f}m "
                f"actual_clear={surface_clearance:.3f}m "
                f"collision={geometric_collision} "
                f"barrier={result.selected_barrier_energy:.5f} "
                f"circ={result.selected_circulation_energy:.5f} "
                f"latency={planning_time * 1000.0:.1f}ms "
                f"guide_rms={result.mean_guidance_noise_correction:.6f} "
                f"ess={result.mean_final_effective_sample_size:.2f} "
                f"action={action.tolist()}",
                flush=True,
            )
            if goal_distance <= args.stop_distance:
                if args.qwen_freeform_mission and mission_phase == "OUTBOUND":
                    if automatic_return_requested:
                        print(
                            "[mission] outward goal reached; advancing automatically "
                            "to RETURN_HOME",
                            flush=True,
                        )
                        mission_phase = "RETURN_HOME"
                        assert homotopy_selector is not None
                        homotopy_selector.reset()
                        continue
                    success = True
                    print(
                        "[mission] outward goal reached; GO_TO_GOAL plan complete",
                        flush=True,
                    )
                    break

                if args.interactive_return_home and mission_phase == "OUTBOUND":
                    user_command = args.return_command
                    if user_command is None:
                        print(
                            "\nOutward goal reached. What should the rover do? "
                            "(for example: come back / stop)",
                            flush=True,
                        )
                        try:
                            user_command = input("> ").strip()
                        except EOFError as error:
                            raise RuntimeError(
                                "no interactive command was available; set "
                                "--return-command 'come back' for a non-interactive run"
                            ) from error
                    assert homotopy_selector is not None
                    command_overlay = overlay_frame(
                        rgb,
                        goal_mask,
                        obstacle_mask,
                        "Qwen command: RETURN or STOP",
                        show_masks=True,
                    )
                    command_decision = homotopy_selector.classify_command(
                        np.asarray(command_overlay.convert("RGB")),
                        user_command,
                    )
                    return_command_event = {
                        "step": step,
                        "user_command": user_command,
                        "command": command_decision.command,
                        "confidence": command_decision.confidence,
                        "raw_response": command_decision.raw_response,
                    }
                    command_overlay.save(output_directory / "qwen_return_command.png")
                    print(
                        f"[qwen-command] text={user_command!r} "
                        f"decision={command_decision.command} "
                        f"confidence={command_decision.confidence:.2f}",
                        flush=True,
                    )
                    if command_decision.command == "RETURN":
                        mission_phase = "RETURN_HOME"
                        homotopy_selector.reset()
                        continue
                    success = True
                    break
                success = True
                if mission_phase == "RETURN_HOME":
                    roundtrip_completed = True
                break

        if not rows["goal_distance"]:
            raise RuntimeError("rollout produced no steps")
        rollout_path = output_directory / "rollout.npz"
        np.savez_compressed(
            rollout_path,
            **{
                key: (
                    np.stack(values)
                    if isinstance(values[0], np.ndarray)
                    else np.asarray(values)
                )
                for key, values in rows.items()
            },
            goal_position=goal,
            obstacle_position=(
                mesh_centroids[0]
                if mesh_centroids
                else (
                    ghost
                    if ghost is not None
                    else np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)
                )
            ),
            obstacle_positions=(
                np.stack(mesh_centroids)
                if mesh_centroids
                else np.zeros((0, 3), dtype=np.float32)
            ),
            obstacle_velocity_xz=mesh_velocities,
            success=np.asarray(success),
            hz=np.asarray(args.hz, dtype=np.float32),
            start_position_xz=start_position_xz,
            initial_goal_distance=np.asarray(initial_goal_distance, dtype=np.float64),
            stop_distance=np.asarray(args.stop_distance, dtype=np.float64),
            robot_radius=np.asarray(args.robot_radius, dtype=np.float64),
            evaluation_layout=np.asarray(args.evaluation_layout),
            seed=np.asarray(args.seed, dtype=np.int64),
            goal_mode=np.asarray(args.goal_mode),
            belief_pixel_goal=np.asarray(args.belief_pixel_goal),
            interactive_return_home=np.asarray(args.interactive_return_home),
            qwen_freeform_mission=np.asarray(args.qwen_freeform_mission),
            automatic_return_requested=np.asarray(automatic_return_requested),
            roundtrip_completed=np.asarray(roundtrip_completed),
            final_mission_phase=np.asarray(mission_phase),
        )
        with (output_directory / "manifest.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "success": success,
                    "steps": len(rows["goal_distance"]),
                    "archived_observations": args.archive_observations,
                    "final_goal_distance": float(rows["goal_distance"][-1]),
                    "planner": "released_navdp_s2diff_pixels",
                    "controller": "direct_waypoint_no_optimizer",
                    "qwen_role": "homotopy_return_command_and_mission_plan",
                    "qwen_process_isolated_from_habitat": True,
                    "qwen_creates_goal_or_action": False,
                    "qwen_homotopy": args.qwen_homotopy,
                    "qwen_homotopy_events": homotopy_events,
                    "qwen_homotopy_forces_all_candidates": args.qwen_homotopy,
                    "homotopy_sign_convention": {"LEFT": -1.0, "RIGHT": 1.0},
                    "homotopy_minimum_obstacle_pixels": args.homotopy_minimum_obstacle_pixels,
                    "homotopy_release_clear_frames": args.homotopy_release_clear_frames,
                    "homotopy_consistency_repeats": args.homotopy_consistency_repeats,
                    "uses_velocity_chunk": False,
                    "obstacle_mode": args.obstacle_mode,
                    "obstacle_world_xz": args.obstacle_world_xz,
                    "goal_mesh": args.goal_mesh,
                    "particle_anchor": args.particle_anchor,
                    "particle_energy_reweighting": args.particle_energy_reweighting,
                    "particle_collision_mask": args.particle_collision_mask,
                    "goal_mode": args.goal_mode,
                    "interactive_return_home": args.interactive_return_home,
                    "qwen_freeform_mission": args.qwen_freeform_mission,
                    "mission_plan_event": mission_plan_event,
                    "automatic_return_requested": automatic_return_requested,
                    "phase_completion_source": "metric_distance_state_machine",
                    "roundtrip_completed": roundtrip_completed,
                    "final_mission_phase": mission_phase,
                    "return_command_event": return_command_event,
                    "home_belief_source": "spawn_origin_plus_executed_odometry",
                    "reached_goal_becomes_obstacle_on_return": True,
                    "return_goal_obstacle_activation_distance": (
                        args.return_goal_obstacle_activation_distance
                    ),
                    "return_goal_obstacle_dilation_pixels": (
                        args.return_goal_obstacle_dilation_pixels
                    ),
                    "belief_pixel_goal": args.belief_pixel_goal,
                    "belief_source": "semantic_goal_mask_plus_odometry",
                    "belief_bootstrap_world_goal": args.belief_bootstrap_world_goal,
                    "belief_measurement_std": args.belief_measurement_std,
                    "belief_translation_process_std": args.belief_translation_process_std,
                    "belief_yaw_process_std_deg": args.belief_yaw_process_std_deg,
                    "belief_covariance_controls_navdp_mask_size": False,
                    "belief_heading_recovery": args.belief_heading_recovery,
                    "belief_recovery_obstacle_gated": True,
                    "belief_recovery_bearing_deg": args.belief_recovery_bearing_deg,
                    "belief_recovery_maximum_yaw_rate": args.belief_recovery_maximum_yaw_rate,
                    "belief_recovery_maximum_forward_speed": args.belief_recovery_maximum_forward_speed,
                    "particle_noise_schedule": args.particle_noise_schedule,
                    "progressive_guidance": args.progressive_guidance,
                    "mesh_obstacle_count": len(mesh_centroids),
                    "moving_obstacles": bool(np.any(np.abs(mesh_velocities) > 0.0)),
                    "obstacle_velocity_xz": mesh_velocities.tolist(),
                    "evaluation_layout": args.evaluation_layout,
                    "seed": args.seed,
                    "robot_radius": args.robot_radius,
                    "minimum_executed_surface_clearance": (
                        float(np.nanmin(rows["executed_surface_clearance"]))
                        if np.any(np.isfinite(rows["executed_surface_clearance"]))
                        else None
                    ),
                    "geometric_collision": bool(np.any(rows["geometric_collision"])),
                    "rollout": str(rollout_path),
                },
                file,
                indent=2,
            )
        if args.save_video and video_frames:
            save_video(
                video_frames,
                output_directory / "rollout.mp4",
                fps=max(args.hz / max(args.save_every, 1), 1.0),
            )
        print(f"Saved rollout: {rollout_path}", flush=True)
        print(f"Success: {success}", flush=True)
    finally:
        if simulator is not None:
            simulator.close()
        stop_server(server_process)
        stop_server(qwen_process)


if __name__ == "__main__":
    main()
