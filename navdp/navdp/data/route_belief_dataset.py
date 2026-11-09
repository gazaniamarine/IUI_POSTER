from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class RouteBeliefDataset(Dataset):
    """Load route-belief training samples from .npz or .pkl files.

    Expected per-sample fields:
        rgb, depth, sam_masks, obstacle_map, belief_tensor, route_index,
        active_goal_id, expert_waypoints, odom_delta, robot_pose

    Numeric arrays are converted to tensors. String/dict fields are kept as-is.
    Files may contain one sample dict or batched arrays with a shared first dim.
    """

    def __init__(
        self,
        paths: Sequence[str | Path] | str | Path,
        required_fields: Optional[Iterable[str]] = None,
    ):
        root_paths = _expand_paths(paths)
        if not root_paths:
            raise ValueError("no route-belief files found")
        self.files = root_paths
        self.required_fields = tuple(
            required_fields
            or (
                "rgb",
                "depth",
                "obstacle_map",
                "belief_tensor",
                "route_index",
                "active_goal_id",
                "expert_waypoints",
            )
        )
        self.index: List[Tuple[int, Optional[int]]] = []
        self._lengths: List[Optional[int]] = []
        for file_id, path in enumerate(self.files):
            sample = _load_file(path)
            n = _batched_length(sample)
            self._lengths.append(n)
            if n is None:
                self.index.append((file_id, None))
            else:
                self.index.extend((file_id, i) for i in range(n))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, object]:
        file_id, row = self.index[i]
        sample = _load_file(self.files[file_id])
        if "samples" in sample and isinstance(sample["samples"], list):
            if row is None:
                raise IndexError("list-backed sample requires a row index")
            sample = sample["samples"][row]
            if not isinstance(sample, dict):
                raise TypeError("items in a samples list must be dictionaries")
            row = None
        if row is not None:
            n = self._lengths[file_id]
            sample = {
                k: (v[row] if _is_batched_value(v, n) else v)
                for k, v in sample.items()
            }
        missing = [k for k in self.required_fields if k not in sample]
        if missing:
            raise KeyError(f"{self.files[file_id]} is missing fields: {missing}")
        return {k: _to_tensor_if_numeric(v) for k, v in sample.items()}


def route_belief_collate(batch: List[Mapping[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if all(torch.is_tensor(v) for v in vals):
            out[key] = torch.stack(vals)
        else:
            out[key] = vals
    return out


def _expand_paths(paths: Sequence[str | Path] | str | Path) -> List[Path]:
    if isinstance(paths, (str, Path)):
        p = Path(paths)
        if p.is_dir():
            return sorted([*p.glob("*.npz"), *p.glob("*.pkl"), *p.glob("*.pickle")])
        return [p]
    out: List[Path] = []
    for item in paths:
        p = Path(item)
        if p.is_dir():
            out.extend(sorted([*p.glob("*.npz"), *p.glob("*.pkl"), *p.glob("*.pickle")]))
        else:
            out.append(p)
    return out


def _load_file(path: Path) -> Dict[str, object]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {k: _unwrap_np_object(data[k]) for k in data.files}
    if path.suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, list):
            return {"samples": obj}
        if not isinstance(obj, dict):
            raise TypeError(f"{path} must contain a dict or list of dicts")
        return obj
    raise ValueError(f"unsupported route-belief file extension: {path.suffix}")


def _batched_length(sample: Mapping[str, object]) -> Optional[int]:
    if "samples" in sample and isinstance(sample["samples"], list):
        return len(sample["samples"])
    lengths = []
    for v in sample.values():
        if isinstance(v, np.ndarray) and v.ndim > 0 and v.dtype != object:
            lengths.append(v.shape[0])
        elif torch.is_tensor(v) and v.ndim > 0:
            lengths.append(v.shape[0])
    if not lengths:
        return None
    n = lengths[0]
    return n if lengths.count(n) == len(lengths) else None


def _is_batched_value(value: object, n: Optional[int]) -> bool:
    if n is None:
        return False
    if isinstance(value, np.ndarray) and value.ndim > 0 and value.dtype != object:
        return value.shape[0] == n
    if torch.is_tensor(value) and value.ndim > 0:
        return value.shape[0] == n
    return False


def _to_tensor_if_numeric(value: object) -> object:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"f", "i", "u", "b"}:
            return torch.from_numpy(value)
        return value.tolist()
    if isinstance(value, (float, int, bool, np.number)):
        return torch.tensor(value)
    return value


def _unwrap_np_object(value: np.ndarray) -> object:
    if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
        return value.item()
    return value
