"""Integration project (next.md) Phase 2: exercises navdp_upstream_client's
pure/mockable pieces -- no real navdp_server needed. encode_depth_png16's
scale/clip and pointgoal_step's request/response shape were read directly
from InternRobotics/NavDP's baselines/navdp/navdp_server.py +
policy_agent.py + policy_network.py (see navdp_upstream_client.py's
docstring), so these tests pin down that contract."""

import json
from unittest.mock import patch

import numpy as np
from PIL import Image

from sam_vla.core.types import Action
from sam_vla.policy.navdp_upstream_client import (
    encode_depth_png16,
    encode_rgb_jpeg,
    pointgoal_step,
    select_action_from_trajectory,
)


def test_encode_depth_png16_round_trips_meters_to_units():
    depth_m = np.array([[0.0, 1.0], [3.5, 6.5535]], dtype=np.float32)
    png_bytes = encode_depth_png16(depth_m)
    decoded = np.asarray(Image.open(__import__("io").BytesIO(png_bytes)).convert("I"))
    expected_units = (depth_m * 10000.0).astype(np.uint16)
    np.testing.assert_array_equal(decoded, expected_units.astype(np.int32))


def test_encode_depth_png16_clips_out_of_range():
    depth_m = np.array([[-1.0, 100.0]], dtype=np.float32)
    png_bytes = encode_depth_png16(depth_m)
    decoded = np.asarray(Image.open(__import__("io").BytesIO(png_bytes)).convert("I"))
    assert decoded[0, 0] == 0
    assert decoded[0, 1] == 65535


def test_encode_rgb_jpeg_produces_decodable_image():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    jpeg_bytes = encode_rgb_jpeg(rgb)
    decoded = Image.open(__import__("io").BytesIO(jpeg_bytes))
    assert decoded.size == (8, 8)
    assert decoded.mode == "RGB"


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_pointgoal_step_clips_goal_and_parses_trajectory():
    predict_size = 4
    fake_trajectory = np.arange(predict_size * 3, dtype=np.float32).reshape(
        1, predict_size, 3
    )
    fake_values = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    captured = {}

    def fake_post(url, files=None, data=None, timeout=None):
        captured["url"] = url
        captured["files"] = files
        captured["data"] = data
        return _FakeResponse(
            {
                "trajectory": fake_trajectory.tolist(),
                "all_trajectory": fake_trajectory.tolist(),
                "all_values": fake_values.tolist(),
            }
        )

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    depth = np.ones((4, 4), dtype=np.float32)
    with patch("sam_vla.policy.navdp_upstream_client.requests.post", fake_post):
        trajectory_xy, all_values = pointgoal_step(
            "http://127.0.0.1:8766", rgb, depth, goal_forward=99.0, goal_left=-99.0
        )

    assert captured["url"] == "http://127.0.0.1:8766/pointgoal_step"
    sent_goal = json.loads(captured["data"]["goal_data"])
    assert sent_goal == {
        "goal_x": [10.0],
        "goal_y": [-10.0],  # clamped per policy_agent.py
        "obstacle_pixels": [[]],  # unset by default -- see pointgoal_step's docstring
    }
    assert set(captured["files"].keys()) == {"image", "depth"}

    assert trajectory_xy.shape == (predict_size, 2)
    np.testing.assert_allclose(trajectory_xy, fake_trajectory[0, :, :2])
    np.testing.assert_allclose(all_values, fake_values.reshape(-1))


def test_select_action_from_trajectory_steers_toward_waypoint():
    # Straight ahead: zero yaw_rate, v_fwd == distance (under the speed cap).
    trajectory = np.array([[2.0, 0.0]], dtype=np.float32)
    action = select_action_from_trajectory(
        trajectory, waypoint_index=0, max_forward_speed=5.0, turn_kp=1.0, max_yaw_rate=1.0
    )
    assert isinstance(action, Action)
    assert action.yaw_rate == 0.0
    assert action.v_fwd == 2.0
    assert action.v_lat == 0.0

    # Waypoint to the left (positive y): positive yaw_rate (CCW), per
    # pose_integrator's convention and yaw_rate_toward_heading's sign.
    trajectory_left = np.array([[1.0, 1.0]], dtype=np.float32)
    action_left = select_action_from_trajectory(
        trajectory_left, waypoint_index=0, turn_kp=1.0, max_yaw_rate=10.0
    )
    assert action_left.yaw_rate > 0.0


def test_select_action_from_trajectory_clamps_speed_and_index():
    trajectory = np.array([[10.0, 0.0], [0.5, 0.0]], dtype=np.float32)
    action = select_action_from_trajectory(
        trajectory, waypoint_index=0, max_forward_speed=1.0
    )
    assert action.v_fwd == 1.0  # clamped, not the raw 10.0m distance

    # Out-of-range index clamps to the last waypoint instead of raising.
    action_clamped = select_action_from_trajectory(
        trajectory, waypoint_index=99, max_forward_speed=5.0
    )
    assert action_clamped.v_fwd == 0.5
