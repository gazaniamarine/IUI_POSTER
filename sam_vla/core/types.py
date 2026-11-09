from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Pose:
    x: float
    y: float
    z: float
    yaw: float


@dataclass
class Observation:
    rgb: np.ndarray
    depth: Optional[np.ndarray]
    pose: Pose
    frame_idx: int
    semantic: Optional[np.ndarray] = None


@dataclass
class Detection:
    class_name: str
    bbox_norm: tuple[float, float, float, float]
    confidence: float

    def validate(self) -> None:
        _validate_bbox_norm(self.bbox_norm, "Detection.bbox_norm")


@dataclass
class GoalSpec:
    goal_bbox_norm: tuple[float, float, float, float]
    obstacle_bboxes_norm: list[tuple[float, float, float, float]]
    instruction_text: str

    def validate(self) -> None:
        _validate_bbox_norm(self.goal_bbox_norm, "GoalSpec.goal_bbox_norm")
        for i, bbox in enumerate(self.obstacle_bboxes_norm):
            _validate_bbox_norm(bbox, f"GoalSpec.obstacle_bboxes_norm[{i}]")


@dataclass
class Action:
    v_fwd: float
    v_lat: float
    yaw_rate: float


@dataclass
class TrackedGoal:
    """One goal discovered by the multi-goal (SAM3 + CLIP) path.

    clip_embedding is the crop's CLIP image embedding at goal-minting time,
    kept around so later re-segmentation passes can re-identify the same
    physical object via clip_goal_classifier.match_or_mint instead of
    minting a duplicate goal_id.
    """

    goal_id: str
    category: str
    clip_embedding: np.ndarray
    clip_score: float
    first_seen_step: int
    bbox_norm: tuple[float, float, float, float]

    def validate(self) -> None:
        _validate_bbox_norm(self.bbox_norm, "TrackedGoal.bbox_norm")


@dataclass
class MultiGoalSpec:
    goals: dict[str, TrackedGoal]
    route: list[str]
    obstacle_bboxes_norm: list[tuple[float, float, float, float]]
    instruction_text: str
    goal_vocabulary: list[str]

    def validate(self) -> None:
        for goal_id, goal in self.goals.items():
            if goal.goal_id != goal_id:
                raise ValueError(
                    f"MultiGoalSpec.goals key {goal_id!r} does not match "
                    f"TrackedGoal.goal_id {goal.goal_id!r}"
                )
            goal.validate()
        for route_goal_id in self.route:
            if route_goal_id not in self.goals:
                raise ValueError(
                    f"MultiGoalSpec.route references unknown goal_id {route_goal_id!r}"
                )
        for i, bbox in enumerate(self.obstacle_bboxes_norm):
            _validate_bbox_norm(bbox, f"MultiGoalSpec.obstacle_bboxes_norm[{i}]")


def _validate_bbox_norm(bbox: tuple[float, float, float, float], name: str) -> None:
    x0, y0, x1, y1 = bbox
    for coord_name, coord in (("x0", x0), ("y0", y0), ("x1", x1), ("y1", y1)):
        if not (0.0 <= coord <= 1.0):
            raise ValueError(f"{name}: {coord_name}={coord!r} is out of range [0, 1]")
    if x0 >= x1:
        raise ValueError(f"{name}: x0 ({x0!r}) must be < x1 ({x1!r})")
    if y0 >= y1:
        raise ValueError(f"{name}: y0 ({y0!r}) must be < y1 ({y1!r})")
