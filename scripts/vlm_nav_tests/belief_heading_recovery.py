# """Obstacle-gated heading recovery for an off-screen goal belief."""

# from __future__ import annotations

# import math

# import numpy as np


# def belief_heading_recovery_action(
#     navdp_action: np.ndarray,
#     *,
#     belief_bearing: float,
#     obstacle_relevant: bool,
#     enabled: bool,
#     activation_bearing: float,
#     yaw_gain: float,
#     maximum_yaw_rate: float,
#     maximum_forward_speed: float,
# ) -> tuple[np.ndarray, bool]:
#     """Strengthen goal reacquisition only after the obstacle gate is clear."""

#     action = np.asarray(navdp_action, dtype=np.float32).copy()
#     if (
#         not enabled
#         or obstacle_relevant
#         or not np.isfinite(belief_bearing)
#         or abs(float(belief_bearing)) <= float(activation_bearing)
#     ):
#         return action, False
#     action[2] = float(
#         np.clip(
#             float(yaw_gain) * float(belief_bearing),
#             -float(maximum_yaw_rate),
#             float(maximum_yaw_rate),
#         )
#     )
#     action[0] = (
#         0.0
#         if abs(float(belief_bearing)) >= math.pi / 2.0
#         else min(max(float(action[0]), 0.0), float(maximum_forward_speed))
#     )
#     action[1] = 0.0
#     return action, True

"""Obstacle-gated heading recovery for an off-screen goal belief."""

from __future__ import annotations

import math

import numpy as np


def belief_heading_recovery_action(
    navdp_action: np.ndarray,
    *,
    belief_bearing: float,
    obstacle_relevant: bool,
    enabled: bool,
    activation_bearing: float,
    yaw_gain: float,
    maximum_yaw_rate: float,
    maximum_forward_speed: float,
) -> tuple[np.ndarray, bool]:
    """Strengthen goal reacquisition only after the obstacle gate is clear."""

    action = np.asarray(navdp_action, dtype=np.float32).copy()
    if (
        not enabled
        or obstacle_relevant
        or not np.isfinite(belief_bearing)
        or abs(float(belief_bearing)) <= float(activation_bearing)
    ):
        return action, False
    action[2] = float(
        np.clip(
            float(yaw_gain) * float(belief_bearing),
            -float(maximum_yaw_rate),
            float(maximum_yaw_rate),
        )
    )
    action[0] = (
        0.0
        if abs(float(belief_bearing)) >= math.pi / 2.0
        else min(max(float(action[0]), 0.0), float(maximum_forward_speed))
    )
    action[1] = 0.0
    return action, True
