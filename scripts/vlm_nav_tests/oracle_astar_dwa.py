"""Privileged-geometry A* + dynamic-window controller for static box obstacles.

This module intentionally contains no learned perception.  It is an oracle-map
reference: obstacle centers and dimensions are supplied directly in world X,Z.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BoxObstacle:
    center_x: float
    center_z: float
    half_extent: float


@dataclass(frozen=True)
class AStarConfig:
    resolution: float = 0.10
    padding: float = 4.0
    planning_clearance: float = 0.18


@dataclass(frozen=True)
class DWAConfig:
    maximum_forward_speed: float = 0.50
    maximum_yaw_rate: float = 0.80
    maximum_forward_acceleration: float = 2.0
    maximum_yaw_acceleration: float = 4.0
    forward_samples: int = 6
    yaw_samples: int = 17
    prediction_horizon: float = 2.0
    prediction_dt: float = 0.10
    path_lookahead: float = 1.20
    desired_surface_clearance: float = 0.18
    goal_weight: float = 3.0
    path_weight: float = 1.2
    heading_weight: float = 0.8
    clearance_weight: float = 5.0
    speed_weight: float = 0.35


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def heading_to(start_xz: np.ndarray, target_xz: np.ndarray) -> float:
    """Yaw matching the Mars convention: yaw=0 faces world -Z."""

    delta = np.asarray(target_xz, dtype=np.float64) - np.asarray(
        start_xz, dtype=np.float64
    )
    return math.atan2(-float(delta[0]), -float(delta[1]))


def integrate_unicycle(
    state: np.ndarray, forward_speed: float, yaw_rate: float, dt: float
) -> np.ndarray:
    x, z, yaw = [float(value) for value in state]
    x += -math.sin(yaw) * float(forward_speed) * float(dt)
    z += -math.cos(yaw) * float(forward_speed) * float(dt)
    yaw = wrap_angle(yaw + float(yaw_rate) * float(dt))
    return np.asarray([x, z, yaw], dtype=np.float64)


def center_clearance_to_boxes(
    point_xz: np.ndarray, obstacles: Sequence[BoxObstacle]
) -> float:
    """Exact planar distance from a point to the union of axis-aligned boxes."""

    if not obstacles:
        return float("inf")
    x, z = [float(value) for value in np.asarray(point_xz).reshape(2)]
    best = float("inf")
    for obstacle in obstacles:
        dx = max(abs(x - obstacle.center_x) - obstacle.half_extent, 0.0)
        dz = max(abs(z - obstacle.center_z) - obstacle.half_extent, 0.0)
        best = min(best, math.hypot(dx, dz))
    return best


def surface_clearance_to_boxes(
    point_xz: np.ndarray,
    obstacles: Sequence[BoxObstacle],
    robot_radius: float,
) -> float:
    return max(
        center_clearance_to_boxes(point_xz, obstacles) - float(robot_radius), 0.0
    )


def collides(
    point_xz: np.ndarray,
    obstacles: Sequence[BoxObstacle],
    inflation: float,
) -> bool:
    return center_clearance_to_boxes(point_xz, obstacles) <= float(inflation)


def _segment_is_free(
    start: np.ndarray,
    end: np.ndarray,
    obstacles: Sequence[BoxObstacle],
    inflation: float,
    spacing: float,
) -> bool:
    distance = float(np.linalg.norm(end - start))
    count = max(int(math.ceil(distance / max(spacing, 1.0e-4))), 1)
    for fraction in np.linspace(0.0, 1.0, count + 1):
        if collides(start * (1.0 - fraction) + end * fraction, obstacles, inflation):
            return False
    return True


def _smooth_path(
    path: np.ndarray,
    obstacles: Sequence[BoxObstacle],
    inflation: float,
    spacing: float,
) -> np.ndarray:
    if len(path) <= 2:
        return path
    output = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        target = len(path) - 1
        while target > anchor + 1 and not _segment_is_free(
            path[anchor], path[target], obstacles, inflation, spacing
        ):
            target -= 1
        output.append(path[target])
        anchor = target
    return np.asarray(output, dtype=np.float64)


def astar_path(
    start_xz: np.ndarray,
    goal_xz: np.ndarray,
    obstacles: Sequence[BoxObstacle],
    robot_radius: float,
    config: AStarConfig = AStarConfig(),
) -> np.ndarray:
    """Plan an 8-connected global path using exact obstacle coordinates."""

    start = np.asarray(start_xz, dtype=np.float64).reshape(2)
    goal = np.asarray(goal_xz, dtype=np.float64).reshape(2)
    inflation = float(robot_radius) + float(config.planning_clearance)
    points = [start, goal]
    for obstacle in obstacles:
        points.extend(
            [
                np.asarray(
                    [
                        obstacle.center_x - obstacle.half_extent - inflation,
                        obstacle.center_z - obstacle.half_extent - inflation,
                    ]
                ),
                np.asarray(
                    [
                        obstacle.center_x + obstacle.half_extent + inflation,
                        obstacle.center_z + obstacle.half_extent + inflation,
                    ]
                ),
            ]
        )
    stacked = np.stack(points)
    lower = stacked.min(axis=0) - float(config.padding)
    upper = stacked.max(axis=0) + float(config.padding)
    resolution = float(config.resolution)
    shape = np.ceil((upper - lower) / resolution).astype(np.int64) + 1

    def world(index: tuple[int, int]) -> np.ndarray:
        return lower + np.asarray(index, dtype=np.float64) * resolution

    def grid(point: np.ndarray) -> tuple[int, int]:
        index = np.rint((point - lower) / resolution).astype(np.int64)
        index = np.clip(index, 0, shape - 1)
        return int(index[0]), int(index[1])

    start_index, goal_index = grid(start), grid(goal)
    if collides(start, obstacles, inflation):
        raise ValueError("A* start lies inside the inflated obstacle map")
    if collides(goal, obstacles, inflation):
        raise ValueError("A* goal lies inside the inflated obstacle map")

    neighbors = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    )
    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (float(np.linalg.norm(goal - start)), 0.0, start_index))
    cost = {start_index: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal_index:
            break
        closed.add(current)
        current_world = world(current)
        for dx, dz in neighbors:
            neighbor = (current[0] + dx, current[1] + dz)
            if not (0 <= neighbor[0] < shape[0] and 0 <= neighbor[1] < shape[1]):
                continue
            neighbor_world = world(neighbor)
            if collides(neighbor_world, obstacles, inflation):
                continue
            if dx != 0 and dz != 0:
                # Prevent a diagonal edge from cutting through an inflated corner.
                side_a = world((current[0] + dx, current[1]))
                side_b = world((current[0], current[1] + dz))
                if collides(side_a, obstacles, inflation) or collides(
                    side_b, obstacles, inflation
                ):
                    continue
            step_cost = float(np.linalg.norm(neighbor_world - current_world))
            proposed = current_cost + step_cost
            if proposed + 1.0e-12 >= cost.get(neighbor, float("inf")):
                continue
            cost[neighbor] = proposed
            parent[neighbor] = current
            heuristic = float(np.linalg.norm(goal - neighbor_world))
            heapq.heappush(open_heap, (proposed + heuristic, proposed, neighbor))

    if goal_index != start_index and goal_index not in parent:
        raise RuntimeError("A* found no route through the privileged obstacle map")

    indices = [goal_index]
    while indices[-1] != start_index:
        indices.append(parent[indices[-1]])
    indices.reverse()
    path = np.stack([world(index) for index in indices])
    path[0] = start
    path[-1] = goal
    return _smooth_path(
        path,
        obstacles,
        inflation,
        spacing=max(resolution * 0.5, 0.025),
    )


def _polyline_progress(path: np.ndarray) -> np.ndarray:
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate((np.zeros(1), np.cumsum(segment_lengths)))


def _lookahead_point(path: np.ndarray, point: np.ndarray, distance: float) -> np.ndarray:
    if len(path) == 1:
        return path[0]
    progress = _polyline_progress(path)
    starts, ends = path[:-1], path[1:]
    vectors = ends - starts
    squared = np.einsum("ij,ij->i", vectors, vectors)
    fractions = np.divide(
        np.einsum("ij,ij->i", point[None, :] - starts, vectors),
        squared,
        out=np.zeros(len(vectors), dtype=np.float64),
        where=squared > 1.0e-12,
    )
    fractions = np.clip(fractions, 0.0, 1.0)
    projections = starts + fractions[:, None] * vectors
    nearest_segment = int(
        np.argmin(np.linalg.norm(projections - point[None, :], axis=1))
    )
    nearest_progress = progress[nearest_segment] + fractions[nearest_segment] * math.sqrt(
        squared[nearest_segment]
    )
    target_progress = min(nearest_progress + float(distance), progress[-1])
    segment = int(np.searchsorted(progress, target_progress, side="right") - 1)
    segment = int(np.clip(segment, 0, len(path) - 2))
    span = max(progress[segment + 1] - progress[segment], 1.0e-8)
    fraction = (target_progress - progress[segment]) / span
    return path[segment] * (1.0 - fraction) + path[segment + 1] * fraction


def _distance_to_polyline(point: np.ndarray, path: np.ndarray) -> float:
    if len(path) == 1:
        return float(np.linalg.norm(point - path[0]))
    starts, ends = path[:-1], path[1:]
    vectors = ends - starts
    squared = np.einsum("ij,ij->i", vectors, vectors)
    fractions = np.divide(
        np.einsum("ij,ij->i", point[None, :] - starts, vectors),
        squared,
        out=np.zeros(len(vectors), dtype=np.float64),
        where=squared > 1.0e-12,
    )
    closest = starts + np.clip(fractions, 0.0, 1.0)[:, None] * vectors
    return float(np.linalg.norm(closest - point[None, :], axis=1).min())


def dwa_action(
    state: np.ndarray,
    previous_action: np.ndarray,
    global_path: np.ndarray,
    goal_xz: np.ndarray,
    obstacles: Sequence[BoxObstacle],
    robot_radius: float,
    control_dt: float,
    config: DWAConfig = DWAConfig(),
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``[forward, 0, yaw_rate]``, best rollout, and its score."""

    state = np.asarray(state, dtype=np.float64).reshape(3)
    previous_action = np.asarray(previous_action, dtype=np.float64).reshape(3)
    path = np.asarray(global_path, dtype=np.float64)
    goal = np.asarray(goal_xz, dtype=np.float64).reshape(2)
    maximum_dv = float(config.maximum_forward_acceleration) * float(control_dt)
    maximum_dw = float(config.maximum_yaw_acceleration) * float(control_dt)
    v_low = max(0.0, float(previous_action[0]) - maximum_dv)
    v_high = min(
        float(config.maximum_forward_speed), float(previous_action[0]) + maximum_dv
    )
    w_low = max(
        -float(config.maximum_yaw_rate), float(previous_action[2]) - maximum_dw
    )
    w_high = min(
        float(config.maximum_yaw_rate), float(previous_action[2]) + maximum_dw
    )
    velocities = np.unique(
        np.linspace(v_low, v_high, max(int(config.forward_samples), 2))
    )
    yaw_rates = np.unique(
        np.concatenate(
            (
                np.linspace(w_low, w_high, max(int(config.yaw_samples), 3)),
                np.asarray([np.clip(0.0, w_low, w_high)]),
            )
        )
    )
    lookahead = _lookahead_point(path, state[:2], config.path_lookahead)
    rollout_steps = max(
        int(math.ceil(config.prediction_horizon / config.prediction_dt)), 1
    )
    best_score = float("inf")
    best_action: np.ndarray | None = None
    best_rollout: np.ndarray | None = None

    for velocity in velocities:
        for yaw_rate in yaw_rates:
            predicted = state.copy()
            rollout = []
            minimum_surface_clearance = float("inf")
            feasible = True
            for _ in range(rollout_steps):
                predicted = integrate_unicycle(
                    predicted, velocity, yaw_rate, config.prediction_dt
                )
                rollout.append(predicted.copy())
                center_clearance = center_clearance_to_boxes(
                    predicted[:2], obstacles
                )
                if center_clearance <= float(robot_radius):
                    feasible = False
                    break
                minimum_surface_clearance = min(
                    minimum_surface_clearance,
                    center_clearance - float(robot_radius),
                )
            if not feasible:
                continue
            endpoint = rollout[-1]
            heading_error = abs(
                wrap_angle(heading_to(endpoint[:2], lookahead) - endpoint[2])
            )
            clearance_deficit = max(
                float(config.desired_surface_clearance)
                - minimum_surface_clearance,
                0.0,
            )
            score = (
                float(config.goal_weight) * np.linalg.norm(endpoint[:2] - lookahead)
                + float(config.path_weight)
                * _distance_to_polyline(endpoint[:2], path)
                + float(config.heading_weight) * heading_error
                + float(config.clearance_weight) * clearance_deficit**2
                - float(config.speed_weight) * float(velocity)
                + 0.05 * np.linalg.norm(endpoint[:2] - goal)
            )
            if score < best_score:
                best_score = float(score)
                best_action = np.asarray([velocity, 0.0, yaw_rate], dtype=np.float32)
                best_rollout = np.asarray(rollout, dtype=np.float32)

    if best_action is None or best_rollout is None:
        desired = heading_to(state[:2], lookahead)
        yaw_error = wrap_angle(desired - state[2])
        yaw_rate = float(
            np.clip(2.0 * yaw_error, -config.maximum_yaw_rate, config.maximum_yaw_rate)
        )
        best_action = np.asarray([0.0, 0.0, yaw_rate], dtype=np.float32)
        predicted = state.copy()
        fallback_rollout = []
        for _ in range(rollout_steps):
            predicted = integrate_unicycle(
                predicted, 0.0, yaw_rate, config.prediction_dt
            )
            fallback_rollout.append(predicted.copy())
        best_rollout = np.asarray(fallback_rollout, dtype=np.float32)
        best_score = float("inf")
    return best_action, best_rollout, best_score
