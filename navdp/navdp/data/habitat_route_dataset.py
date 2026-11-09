from __future__ import annotations

import json
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from navdp.extensions import (
    DepthObstacleMap,
    SAMDepthTargetExtractor,
    SubgoalBeliefBank,
)


@dataclass
class HabitatEpisode:
    npz_path: Path
    bbox_path: Optional[Path]
    category: str
    length: int


class HabitatRouteDataset(Dataset):
    """Frame/chunk dataset for the custom Habitat RGB-D target episodes.

    Expected layout:

    ```text
    dataset_root/
      beanbag/
        ep_0000.npz
        ep_0000_bboxes.json
      cabinet/
        ep_0000.npz
      ...
    ```

    Expected `.npz` fields from your screenshot:

    ```text
    rgb                [T,H,W,3] uint8
    depth              [T,H,W] float32
    pose               [T,7] float32
    action_waypoint    [T,3] float32
    seg_masks          [T,H,W] uint8
    goal_visible_pixels[T]
    goal_category      scalar string
    ```

    Returned sample fields are compatible with the route-belief training code:

    ```text
    spatial_semantic : [2,H,W]      mask + normalized depth
    depth            : [H,W]
    rgb              : [H,W,3]
    obstacle_map     : [G,G]
    belief_tensor    : [1,11]       single active target slot for v1
    route_index      : scalar
    active_goal_index: scalar
    expert_waypoints : [horizon, action_dim], selected by action_mode
    proprio          : [7], [3], or [1], depending on proprio_mode
    ```
    """

    def __init__(
        self,
        root: str | Path,
        horizon: int = 8,
        stride: int = 1,
        image_size: Optional[int | Tuple[int, int]] = 224,
        categories: Optional[Sequence[str]] = None,
        camera_intrinsics: Optional[Mapping[str, float]] = None,
        min_mask_area: int = 50,
        min_visible_pixels: int = 20,
        build_belief: bool = True,
        build_obstacle_map: bool = True,
        obstacle_grid_size: int = 96,
        obstacle_resolution: float = 0.05,
        depth_scale: float = 1.0,
        pose_planar_axes: Tuple[int, int] = (0, 2),
        proprio_mode: str = "pose7",
        action_mode: str = "waypoint",
        yaw_axis: str = "y",
        cache_belief: bool = True,
        cache_episodes: int = 0,
        include_rgb: bool = True,
        include_obstacle_channel: bool = False,
        goal_mask_dropout_prob: float = 0.0,
        couple_belief_dropout: bool = True,
        occlusion_mean_len: int = 12,
        occlusion_seed: int = 1234,
        occlusion_trigger: str = "markov",
        occlusion_bearing_frac: float = 0.55,
    ):
        self.root = Path(root)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.image_size = _parse_image_size(image_size)
        self.categories = {_normalize_category(c) for c in categories} if categories is not None else None
        self.min_visible_pixels = int(min_visible_pixels)
        self.build_belief = bool(build_belief)
        self.build_obstacle_map = bool(build_obstacle_map)
        self.obstacle_grid_size = int(obstacle_grid_size)
        self.depth_scale = float(depth_scale)
        self.pose_planar_axes = pose_planar_axes
        self.proprio_mode = _validate_proprio_mode(proprio_mode)
        self.action_mode = _validate_action_mode(action_mode)
        self._action_key = _action_key_for_mode(self.action_mode)
        self.yaw_axis = _validate_yaw_axis(yaw_axis)
        self.cache_belief = bool(cache_belief)
        self.cache_episodes = max(int(cache_episodes), 0)
        self.include_rgb = bool(include_rgb)
        self.include_obstacle_channel = bool(include_obstacle_channel)
        self.goal_mask_dropout_prob = float(np.clip(goal_mask_dropout_prob, 0.0, 1.0))
        self.couple_belief_dropout = bool(couple_belief_dropout)
        self.occlusion_mean_len = max(int(occlusion_mean_len), 1)
        self.occlusion_seed = int(occlusion_seed)
        if occlusion_trigger not in ("markov", "edge"):
            raise ValueError("occlusion_trigger must be 'markov' or 'edge'")
        self.occlusion_trigger = str(occlusion_trigger)
        self.occlusion_bearing_frac = float(np.clip(occlusion_bearing_frac, 0.0, 1.0))
        self._belief_cache: Dict[int, np.ndarray] = {}
        self._occlusion_cache: Dict[int, np.ndarray] = {}
        self._episode_cache: OrderedDict[int, Dict[str, object]] = OrderedDict()

        if not self.root.exists():
            raise FileNotFoundError(f"dataset root does not exist: {self.root}")

        self.episodes = self._discover_episodes()
        if not self.episodes:
            raise ValueError(f"no .npz episodes found under {self.root}")

        self.index: List[Tuple[int, int]] = []
        for ep_id, ep in enumerate(self.episodes):
            last = max(ep.length - 1, 0)
            for t in range(0, last + 1, self.stride):
                self.index.append((ep_id, t))

        h0, w0 = self._first_hw()
        intr = dict(camera_intrinsics or _default_intrinsics(h0, w0))
        self.target_extractor = SAMDepthTargetExtractor(
            intr,
            min_mask_area=min_mask_area,
            depth_scale=depth_scale,
            position_dim=2,
        )
        self.obstacle_builder = DepthObstacleMap(
            grid_size=obstacle_grid_size,
            resolution=obstacle_resolution,
            camera_intrinsics=intr,
            depth_scale=depth_scale,
        )

    def __len__(self) -> int:
        return len(self.index)

    def precompute_belief_cache(self, verbose: bool = False) -> None:
        """Build the tiny per-episode belief cache before DataLoader workers start."""
        if not self.build_belief or not self.cache_belief:
            return
        for ep_id, ep in enumerate(self.episodes):
            if ep_id in self._belief_cache:
                continue
            if verbose and (ep_id == 0 or (ep_id + 1) % 25 == 0 or ep_id + 1 == len(self.episodes)):
                print(f"precomputing belief cache {ep_id + 1}/{len(self.episodes)}", flush=True)
            with np.load(ep.npz_path, allow_pickle=True) as data:
                category = _read_category(data, fallback=ep.category)
                self._belief_cache[ep_id] = self._episode_belief_sequence(ep_id, data, category)

    def __getitem__(self, i: int) -> Dict[str, object]:
        ep_id, t = self.index[i]
        ep = self.episodes[ep_id]
        data = self._episode_arrays(ep_id)
        category = _read_category(data, fallback=ep.category)

        depth = np.asarray(data["depth"][t], dtype=np.float32)
        seg = np.asarray(data["seg_masks"][t]).astype(np.uint8)
        goal_mask = _goal_mask_from_seg(seg)
        obstacle_mask = _obstacle_mask_from_seg(seg)
        depth_for_obstacles = depth
        pose = np.asarray(data["pose"][t], dtype=np.float32) if "pose" in data else np.zeros(7, np.float32)
        proprio = _proprio_from_pose(pose, self.proprio_mode, self.pose_planar_axes, self.yaw_axis)
        true_visible_pixels = _visible_pixel_count(data, t, goal_mask)
        true_visible = true_visible_pixels >= self.min_visible_pixels and int(goal_mask.sum()) >= self.min_visible_pixels
        goal_mask_dropped = self._goal_mask_dropped(ep_id, t, true_visible)

        if self.image_size is not None:
            if self.include_rgb:
                rgb, depth, goal_mask = _resize_triplet(np.asarray(data["rgb"][t]), depth, goal_mask, self.image_size)
            else:
                depth, goal_mask = _resize_depth_mask(depth, goal_mask, self.image_size)
            _, obstacle_mask = _resize_depth_mask(depth_for_obstacles, obstacle_mask, self.image_size)
        elif self.include_rgb:
            rgb = np.asarray(data["rgb"][t])

        if goal_mask_dropped:
            observed_goal_mask = np.zeros_like(goal_mask, dtype=np.uint8)
            visible_pixels = 0
            visible = False
        else:
            observed_goal_mask = goal_mask
            visible_pixels = true_visible_pixels
            visible = true_visible

        goal_mask_float = (observed_goal_mask > 0).astype(np.float32)
        obstacle_mask_float = (obstacle_mask > 0).astype(np.float32)
        depth_norm = _normalize_depth(depth)
        spatial_parts = [goal_mask_float]
        if self.include_obstacle_channel:
            spatial_parts.append(obstacle_mask_float)
        spatial_parts.append(depth_norm)
        spatial = np.stack(spatial_parts, axis=0).astype(np.float32)

        action_targets = _action_chunk(data, t, self.horizon, self.action_mode)

        if self.build_obstacle_map:
            obstacle_map = self.obstacle_builder.build(depth_for_obstacles)
        else:
            obstacle_map = np.zeros((self.obstacle_grid_size, self.obstacle_grid_size), dtype=np.float32)

        if self.build_belief:
            belief = self._belief_tensor_for_frame(ep_id, data, category, t)
        else:
            belief = _empty_belief_tensor()

        bboxes = _load_bboxes(ep.bbox_path, t)
        if bboxes is None:
            bboxes = _bbox_from_arrays(data, t)

        sample = {
            "depth": torch.from_numpy(depth.astype(np.float32)),
            "semantic": torch.from_numpy(goal_mask_float[None]),
            "goal_mask": torch.from_numpy(goal_mask_float),
            "obstacle_mask": torch.from_numpy(obstacle_mask_float),
            "spatial_semantic": torch.from_numpy(spatial),
            "obstacle_map": torch.from_numpy(obstacle_map.astype(np.float32)),
            "belief_tensor": torch.from_numpy(belief.astype(np.float32)),
            "route_index": torch.tensor(0, dtype=torch.long),
            "active_goal_index": torch.tensor(0, dtype=torch.long),
            "active_goal_id": category,
            "goal_category": category,
            "expert_waypoints": torch.from_numpy(action_targets.astype(np.float32)),
            "proprio": torch.from_numpy(proprio.astype(np.float32)),
            "robot_pose": torch.from_numpy(pose.astype(np.float32)),
            "visible": torch.tensor(visible, dtype=torch.bool),
            "goal_visible_pixels": torch.tensor(visible_pixels, dtype=torch.long),
            "true_goal_visible_pixels": torch.tensor(true_visible_pixels, dtype=torch.long),
            "goal_mask_dropped": torch.tensor(goal_mask_dropped, dtype=torch.bool),
            "obstacle_visible_pixels": torch.tensor(_obstacle_visible_pixel_count(data, t, obstacle_mask), dtype=torch.long),
            "episode_path": str(ep.npz_path),
            "frame_index": torch.tensor(t, dtype=torch.long),
            "bboxes": bboxes,
        }
        if self.include_rgb:
            sample["rgb"] = torch.from_numpy(rgb.copy())
        return sample

    def _episode_arrays(self, ep_id: int) -> Dict[str, object]:
        if self.cache_episodes > 0 and ep_id in self._episode_cache:
            cached = self._episode_cache.pop(ep_id)
            self._episode_cache[ep_id] = cached
            return cached

        ep = self.episodes[ep_id]
        with np.load(ep.npz_path, allow_pickle=True) as data:
            out: Dict[str, object] = {
                "depth": np.asarray(data["depth"], dtype=np.float32),
                "seg_masks": np.asarray(data["seg_masks"], dtype=np.uint8),
                self._action_key: np.asarray(data[self._action_key], dtype=np.float32),
                "goal_category": _read_category(data, fallback=ep.category),
            }
            if "pose" in data:
                out["pose"] = np.asarray(data["pose"], dtype=np.float32)
            goal_visible = _read_first_available_array(data, ["goal_visible_pixels", "goal_visible_px"])
            if goal_visible is not None:
                out["goal_visible_pixels"] = goal_visible
            obstacle_visible = _read_first_available_array(data, ["obstacle_visible_pixels", "obstacle_visible_px"])
            if obstacle_visible is not None:
                out["obstacle_visible_pixels"] = obstacle_visible
            if "goal_bbox" in data:
                out["goal_bbox"] = np.asarray(data["goal_bbox"])
            if "obstacle_bbox" in data:
                out["obstacle_bbox"] = np.asarray(data["obstacle_bbox"])
            if "obstacle_category" in data:
                out["obstacle_category"] = _read_scalar(data["obstacle_category"])
            if "obstacle_position" in data:
                out["obstacle_position"] = np.asarray(data["obstacle_position"], dtype=np.float32)
            if self.include_rgb:
                out["rgb"] = np.asarray(data["rgb"], dtype=np.uint8)

        if self.cache_episodes > 0:
            self._episode_cache[ep_id] = out
            while len(self._episode_cache) > self.cache_episodes:
                self._episode_cache.popitem(last=False)
        return out

    def _goal_mask_dropped(self, ep_id: int, t: int, visible: bool) -> bool:
        if not visible:
            return False
        # Edge-triggered occlusion ignores the probability knob (it fires on geometry).
        if self.couple_belief_dropout and self.occlusion_trigger == "edge":
            occ = self._occlusion_for(ep_id, self.episodes[ep_id].length)
            return bool(occ[t]) if t < len(occ) else False
        if self.goal_mask_dropout_prob <= 0.0:
            return False
        if self.couple_belief_dropout:
            occ = self._occlusion_for(ep_id, self.episodes[ep_id].length)
            return bool(occ[t]) if t < len(occ) else False
        return bool(np.random.random() < self.goal_mask_dropout_prob)

    def _occlusion_for(self, ep_id: int, length: int) -> np.ndarray:
        """Deterministic per-episode occlusion schedule (cached for the run).

        Determinism by ep_id keeps the precomputed belief cache valid while still
        giving every episode its own occlusion pattern. The same schedule drives
        both the dropped image goal mask and the belief observation feed so they
        stay consistent: a hidden frame means the belief must coast on odometry.
        """
        length = int(length)
        cached = self._occlusion_cache.get(ep_id)
        if cached is not None and len(cached) >= length:
            return cached[:length]
        occ = self._build_occlusion(ep_id, length)
        self._occlusion_cache[ep_id] = occ
        return occ

    def _goal_xnorm_sequence(self, ep_id: int, length: int) -> np.ndarray:
        """Per-frame normalized horizontal position of the goal centroid in [-1, 1]
        (|x|~1 = at the FOV edge). NaN when the goal is not visible that frame."""
        data = self._episode_arrays(ep_id)
        masks = np.asarray(data["seg_masks"], dtype=np.uint8)
        n = min(length, masks.shape[0])
        xn = np.full(length, np.nan, dtype=np.float32)
        for t in range(n):
            gm = _goal_mask_from_seg(masks[t])
            xs = np.where(gm > 0)[1]
            if xs.size >= self.min_visible_pixels:
                w = gm.shape[1]
                xn[t] = float((xs.mean() - (w - 1) / 2.0) / ((w - 1) / 2.0))
        return xn

    def _build_occlusion_edge(self, ep_id: int, length: int) -> np.ndarray:
        """Edge-triggered occlusion: hide the goal for a run whenever it drifts to the
        FOV edge (|x_norm| >= occlusion_bearing_frac). This concentrates the mask-drop
        (and belief-coast) windows on exactly the edge-recovery moments, so the policy
        must READ the belief direction to turn back -- the Path-B training signal."""
        xn = self._goal_xnorm_sequence(ep_id, length)
        occ = np.zeros(length, dtype=bool)
        frac = self.occlusion_bearing_frac
        run = 0
        for t in range(length):
            if t == 0:
                continue  # always see the goal on frame 0
            if run > 0:
                occ[t] = True
                run -= 1
                continue
            if np.isfinite(xn[t]) and abs(xn[t]) >= frac:
                occ[t] = True
                run = self.occlusion_mean_len - 1  # hold the occlusion through the turn-back
        return occ

    def _build_occlusion(self, ep_id: int, length: int) -> np.ndarray:
        if length <= 0 or not self.couple_belief_dropout:
            return np.zeros(max(length, 0), dtype=bool)
        if self.occlusion_trigger == "edge":
            return self._build_occlusion_edge(ep_id, length)
        if self.goal_mask_dropout_prob <= 0.0:
            return np.zeros(max(length, 0), dtype=bool)
        # Two-state Markov chain: stationary occluded fraction = goal_mask_dropout_prob,
        # mean occluded run length = occlusion_mean_len, so time_since_seen accumulates
        # in correlated windows instead of i.i.d. single-frame flickers.
        p = self.goal_mask_dropout_prob
        b = 1.0 / float(self.occlusion_mean_len)  # occluded -> visible
        a = min(b * p / max(1.0 - p, 1e-6), 1.0)  # visible -> occluded
        rng = np.random.default_rng(self.occlusion_seed + int(ep_id))
        occ = np.zeros(length, dtype=bool)
        state = False  # always see the goal on frame 0 before it can be lost
        for step in range(length):
            if step == 0:
                state = False
            elif state:
                state = not (rng.random() < b)
            else:
                state = rng.random() < a
            occ[step] = state
        return occ

    def _discover_episodes(self) -> List[HabitatEpisode]:
        episodes: List[HabitatEpisode] = []
        for npz_path in sorted(self.root.glob("*.npz")):
            if npz_path.name.endswith("_bboxes.npz"):
                continue
            ep = self._episode_from_path(npz_path, fallback_category=self.root.name)
            if ep is not None:
                episodes.append(ep)
        for class_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            category = _normalize_category(class_dir.name)
            if self.categories is not None and category not in self.categories:
                continue
            for npz_path in sorted(class_dir.glob("*.npz")):
                if npz_path.name.endswith("_bboxes.npz"):
                    continue
                ep = self._episode_from_path(npz_path, fallback_category=category)
                if ep is not None:
                    episodes.append(ep)
        return episodes

    def _episode_from_path(self, npz_path: Path, fallback_category: str) -> Optional[HabitatEpisode]:
        with np.load(npz_path, allow_pickle=True) as data:
            if "rgb" not in data or "depth" not in data or "action_waypoint" not in data or "seg_masks" not in data:
                return None
            category_from_file = _read_category(data, fallback=fallback_category)
            if self.categories is not None and category_from_file not in self.categories:
                return None
            length = int(data["rgb"].shape[0])
        bbox_path = npz_path.with_name(f"{npz_path.stem}_bboxes.json")
        return HabitatEpisode(
            npz_path=npz_path,
            bbox_path=bbox_path if bbox_path.exists() else None,
            category=category_from_file,
            length=length,
        )

    def _first_hw(self) -> tuple[int, int]:
        with np.load(self.episodes[0].npz_path, allow_pickle=True) as data:
            h, w = data["depth"].shape[1:3]
        return int(h), int(w)

    def _belief_tensor_for_frame(self, ep_id: int, data, category: str, t: int) -> np.ndarray:
        if not self.cache_belief:
            return self._episode_belief_sequence(ep_id, data, category, max_step=t)[t]

        cached = self._belief_cache.get(ep_id)
        if cached is None:
            cached = self._episode_belief_sequence(ep_id, data, category)
            self._belief_cache[ep_id] = cached
        return cached[t]

    def _episode_belief_sequence(self, ep_id: int, data, category: str, max_step: Optional[int] = None) -> np.ndarray:
        bank = SubgoalBeliefBank([category], sigma_visible=0.05, odom_noise=0.02)
        prev_pose = None
        depths = np.asarray(data["depth"], dtype=np.float32)
        masks = np.asarray(data["seg_masks"], dtype=np.uint8)
        poses = np.asarray(data["pose"], dtype=np.float32) if "pose" in data else None
        full_length = int(depths.shape[0])
        occ = self._occlusion_for(ep_id, full_length)
        length = full_length
        if max_step is not None:
            length = min(length, int(max_step) + 1)
        frames: List[np.ndarray] = []
        for step in range(length):
            depth = depths[step]
            mask = _goal_mask_from_seg(masks[step])
            visible_pixels = _visible_pixel_count(data, step, mask)
            occluded = bool(occ[step]) if step < len(occ) else False
            if occluded or visible_pixels < self.min_visible_pixels or int(mask.sum()) < self.min_visible_pixels:
                obs = {category: {"visible": False, "position": None, "confidence": 0.0}}
            else:
                extracted = self.target_extractor.extract({category: mask.astype(bool)}, depth)
                obs = extracted
            pose = poses[step] if poses is not None else None
            odom = _odom_delta(prev_pose, pose, planar_axes=self.pose_planar_axes, yaw_axis=self.yaw_axis)
            bank.update(obs, odom_delta=odom, step=step)
            prev_pose = pose
            frame = bank.as_tensor([category], active_goal_id=category, route_index=0, route_length=1).cpu().numpy()
            frames.append(frame.astype(np.float32))
        return np.stack(frames, axis=0)


class HabitatEpisodeBatchSampler(Sampler[List[int]]):
    """Yield batches from one episode at a time for efficient episode caching."""

    def __init__(
        self,
        dataset: HabitatRouteDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self.episode_to_indices: Dict[int, List[int]] = {}
        for global_idx, (ep_id, _frame) in enumerate(dataset.index):
            self.episode_to_indices.setdefault(ep_id, []).append(global_idx)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        ep_ids = list(self.episode_to_indices)
        if self.shuffle:
            rng.shuffle(ep_ids)
        for ep_id in ep_ids:
            indices = list(self.episode_to_indices[ep_id])
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    yield batch

    def __len__(self) -> int:
        total = 0
        for indices in self.episode_to_indices.values():
            n = len(indices)
            total += n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)
        return total


def habitat_route_collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if all(torch.is_tensor(v) for v in vals):
            out[key] = torch.stack(vals)
        else:
            out[key] = vals
    return out


def _read_category(data, fallback: str) -> str:
    if "goal_category" not in data:
        return _normalize_category(fallback)
    return _normalize_category(_read_scalar(data["goal_category"]))


def _normalize_category(category: str) -> str:
    return str(category).replace("_", " ")


def _read_scalar(raw: object) -> str:
    if np.asarray(raw).shape == ():
        return str(np.asarray(raw).item())
    return str(raw)


def _read_first_available_array(data, keys: Sequence[str]) -> Optional[np.ndarray]:
    for key in keys:
        if key in data:
            return np.asarray(data[key])
    return None


def _goal_mask_from_seg(seg: np.ndarray) -> np.ndarray:
    arr = np.asarray(seg)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError("seg_masks frame must have shape [H,W] or [1,H,W]")
    max_val = int(arr.max()) if arr.size else 0
    if max_val <= 1:
        return (arr > 0).astype(np.uint8)
    return (arr == 1).astype(np.uint8)


def _obstacle_mask_from_seg(seg: np.ndarray) -> np.ndarray:
    arr = np.asarray(seg)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError("seg_masks frame must have shape [H,W] or [1,H,W]")
    return (arr == 2).astype(np.uint8)


def _chunk_array(arr: np.ndarray, start: int, horizon: int) -> np.ndarray:
    start = min(int(start), max(len(arr) - 1, 0))
    chunk = arr[start : start + horizon]
    if len(chunk) < horizon:
        pad = np.repeat(chunk[-1:], horizon - len(chunk), axis=0)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk


def _action_chunk(data, start: int, horizon: int, action_mode: str) -> np.ndarray:
    key = _action_key_for_mode(action_mode)
    if key not in data:
        raise KeyError(f"dataset episode is missing {key!r} required by action_mode={action_mode!r}")
    arr = np.asarray(data[key], dtype=np.float32)
    return _chunk_array(arr, start, horizon)


def _action_key_for_mode(action_mode: str) -> str:
    if action_mode == "waypoint":
        return "action_waypoint"
    if action_mode == "action3d":
        return "action_3d"
    if action_mode == "action2d":
        return "action_2d"
    raise ValueError(f"unknown action_mode: {action_mode}")


def _visible_pixel_count(data, frame: int, mask: np.ndarray) -> int:
    if "goal_visible_pixels" not in data:
        return int(np.asarray(mask).sum())
    vals = np.asarray(data["goal_visible_pixels"])
    if vals.shape == ():
        return int(vals.item())
    if frame >= len(vals):
        return int(np.asarray(mask).sum())
    return int(vals[frame])


def _obstacle_visible_pixel_count(data, frame: int, mask: np.ndarray) -> int:
    if "obstacle_visible_pixels" not in data:
        return int(np.asarray(mask).sum())
    vals = np.asarray(data["obstacle_visible_pixels"])
    if vals.shape == ():
        return int(vals.item())
    if frame >= len(vals):
        return int(np.asarray(mask).sum())
    return int(vals[frame])


def _normalize_depth(depth: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d)
    if not valid.any():
        return np.zeros_like(d, dtype=np.float32)
    lo = float(np.nanmin(d[valid]))
    hi = float(np.nanmax(d[valid]))
    return ((d - lo) / (hi - lo + eps)).astype(np.float32)


def _default_intrinsics(height: int, width: int) -> Dict[str, float]:
    # Conservative pinhole default for square Habitat renders when exact camera
    # intrinsics are unavailable. Replace via camera_intrinsics for real runs.
    f = float(max(height, width))
    return {"fx": f, "fy": f, "cx": (width - 1) * 0.5, "cy": (height - 1) * 0.5}


def _parse_image_size(size: Optional[int | Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if size is None:
        return None
    if isinstance(size, int):
        return (size, size)
    return (int(size[0]), int(size[1]))


def _validate_proprio_mode(mode: str) -> str:
    valid = {"pose7", "planar3", "zero"}
    if mode not in valid:
        raise ValueError(f"proprio_mode must be one of {sorted(valid)}, got {mode!r}")
    return mode


def _validate_action_mode(mode: str) -> str:
    valid = {"waypoint", "action3d", "action2d"}
    if mode not in valid:
        raise ValueError(f"action_mode must be one of {sorted(valid)}, got {mode!r}")
    return mode


def _validate_yaw_axis(axis: str) -> str:
    valid = {"x", "y", "z"}
    if axis not in valid:
        raise ValueError(f"yaw_axis must be one of {sorted(valid)}, got {axis!r}")
    return axis


def _proprio_from_pose(
    pose: np.ndarray,
    mode: str,
    planar_axes: Tuple[int, int],
    yaw_axis: str,
) -> np.ndarray:
    if mode == "pose7":
        return np.asarray(pose, dtype=np.float32)
    if mode == "zero":
        return np.zeros(1, dtype=np.float32)
    ax0, ax1 = planar_axes
    if len(pose) < 7:
        return np.zeros(3, dtype=np.float32)
    yaw = _yaw_from_quat_xyzw(pose[3:7], axis=yaw_axis)
    return np.asarray([pose[ax0], pose[ax1], yaw], dtype=np.float32)


def _resize_triplet(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    size: Tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import cv2

        h, w = size
        rgb_r = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        depth_r = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_r = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return rgb_r, depth_r, mask_r
    except Exception:
        return _resize_torch(rgb, depth, mask, size)


def _resize_depth_mask(
    depth: np.ndarray,
    mask: np.ndarray,
    size: Tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import cv2

        h, w = size
        depth_r = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_r = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return depth_r, mask_r
    except Exception:
        h, w = size
        depth_t = torch.from_numpy(depth)[None, None].float()
        mask_t = torch.from_numpy(mask)[None, None].float()
        depth_r = F.interpolate(depth_t, size=(h, w), mode="nearest")[0, 0]
        mask_r = F.interpolate(mask_t, size=(h, w), mode="nearest")[0, 0]
        return depth_r.numpy(), mask_r.byte().numpy()


def _resize_torch(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    size: Tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = size
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)[None].float()
    depth_t = torch.from_numpy(depth)[None, None].float()
    mask_t = torch.from_numpy(mask)[None, None].float()
    rgb_r = F.interpolate(rgb_t, size=(h, w), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    depth_r = F.interpolate(depth_t, size=(h, w), mode="nearest")[0, 0]
    mask_r = F.interpolate(mask_t, size=(h, w), mode="nearest")[0, 0]
    return rgb_r.byte().numpy(), depth_r.numpy(), mask_r.byte().numpy()


def _odom_delta(
    prev_pose: Optional[np.ndarray],
    pose: Optional[np.ndarray],
    planar_axes: Tuple[int, int] = (0, 2),
    yaw_axis: str = "y",
) -> np.ndarray:
    if prev_pose is None or pose is None or len(prev_pose) < 7 or len(pose) < 7:
        return np.zeros(3, dtype=np.float32)
    ax0, ax1 = planar_axes
    prev_xy = np.asarray([prev_pose[ax0], prev_pose[ax1]], dtype=np.float32)
    cur_xy = np.asarray([pose[ax0], pose[ax1]], dtype=np.float32)
    yaw_prev = _yaw_from_quat_xyzw(prev_pose[3:7], axis=yaw_axis)
    yaw_cur = _yaw_from_quat_xyzw(pose[3:7], axis=yaw_axis)
    delta_world = cur_xy - prev_xy
    c = math.cos(-yaw_prev)
    s = math.sin(-yaw_prev)
    dx = c * delta_world[0] - s * delta_world[1]
    dy = s * delta_world[0] + c * delta_world[1]
    dtheta = _wrap_angle(yaw_cur - yaw_prev)
    return np.asarray([dx, dy, dtheta], dtype=np.float32)


def _yaw_from_quat_xyzw(q: Sequence[float], axis: str = "y") -> float:
    x, y, z, w = [float(v) for v in q]
    if axis == "y":
        siny_cosp = 2.0 * (w * y + x * z)
        cosy_cosp = 1.0 - 2.0 * (x * x + y * y)
    elif axis == "z":
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    elif axis == "x":
        siny_cosp = 2.0 * (w * x + y * z)
        cosy_cosp = 1.0 - 2.0 * (x * x + y * y)
    else:
        raise ValueError(f"yaw axis must be x, y, or z, got {axis!r}")
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_angle(a: float) -> float:
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def _empty_belief_tensor() -> np.ndarray:
    return np.asarray([[0, 0, 1000, 0, 1000, 0, 0, 0, 0, 1, 0]], dtype=np.float32)


def _load_bboxes(path: Optional[Path], frame: int) -> object:
    if path is None:
        return None
    try:
        with path.open("r") as f:
            data = json.load(f)
        if isinstance(data, list) and frame < len(data):
            return data[frame]
        if isinstance(data, dict):
            if isinstance(data.get("bboxes"), list) and frame < len(data["bboxes"]):
                return data["bboxes"][frame]
            return data.get(str(frame), data.get(frame))
        return data
    except Exception:
        return None


def _bbox_from_arrays(data, frame: int) -> Optional[Dict[str, object]]:
    out: Dict[str, object] = {}
    if "goal_bbox" in data:
        arr = np.asarray(data["goal_bbox"])
        if arr.ndim >= 2 and frame < len(arr):
            out["goal"] = arr[frame].astype(int).tolist()
    if "obstacle_bbox" in data:
        arr = np.asarray(data["obstacle_bbox"])
        if arr.ndim >= 2 and frame < len(arr):
            out["obstacle"] = arr[frame].astype(int).tolist()
    if "obstacle_category" in data:
        out["obstacle_category"] = data["obstacle_category"]
    if "obstacle_position" in data:
        out["obstacle_position"] = np.asarray(data["obstacle_position"], dtype=np.float32).tolist()
    return out or None
