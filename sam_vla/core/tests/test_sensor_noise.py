import numpy as np
import pytest

from sam_vla.core.sensor_noise import apply_odom_noise
from sam_vla.core.types import Action


def test_zero_noise_returns_action_unchanged():
    action = Action(v_fwd=1.0, v_lat=0.5, yaw_rate=0.2)
    rng = np.random.default_rng(0)
    out = apply_odom_noise(action, 0.0, rng)
    assert out is action


def test_negative_noise_returns_action_unchanged():
    action = Action(v_fwd=1.0, v_lat=0.5, yaw_rate=0.2)
    rng = np.random.default_rng(0)
    out = apply_odom_noise(action, -0.1, rng)
    assert out is action


def test_zero_noise_consumes_no_rng_draws():
    action = Action(v_fwd=1.0, v_lat=0.5, yaw_rate=0.2)
    rng = np.random.default_rng(0)
    apply_odom_noise(action, 0.0, rng)
    # rng untouched -- next draw matches a fresh rng with the same seed
    fresh = np.random.default_rng(0)
    assert rng.normal() == fresh.normal()


def test_positive_noise_perturbs_all_three_fields():
    action = Action(v_fwd=1.0, v_lat=0.5, yaw_rate=0.2)
    rng = np.random.default_rng(42)
    out = apply_odom_noise(action, 0.1, rng)
    assert out.v_fwd != action.v_fwd
    assert out.v_lat != action.v_lat
    assert out.yaw_rate != action.yaw_rate


def test_deterministic_given_seeded_rng():
    action = Action(v_fwd=1.0, v_lat=0.5, yaw_rate=0.2)
    out1 = apply_odom_noise(action, 0.1, np.random.default_rng(7))
    out2 = apply_odom_noise(action, 0.1, np.random.default_rng(7))
    assert out1.v_fwd == out2.v_fwd
    assert out1.v_lat == out2.v_lat
    assert out1.yaw_rate == out2.yaw_rate


def test_yaw_rate_noise_is_half_scale_of_translational_noise():
    # Draw many samples of the *noise itself* (out - in) and confirm the
    # empirical std ratio matches the 1.0 : 0.5 split scenario.py uses.
    action = Action(v_fwd=0.0, v_lat=0.0, yaw_rate=0.0)
    rng = np.random.default_rng(123)
    odom_noise_std = 0.2
    n = 20_000
    v_fwd_samples = np.empty(n)
    yaw_samples = np.empty(n)
    for i in range(n):
        out = apply_odom_noise(action, odom_noise_std, rng)
        v_fwd_samples[i] = out.v_fwd
        yaw_samples[i] = out.yaw_rate

    std_fwd = v_fwd_samples.std()
    std_yaw = yaw_samples.std()
    assert std_fwd == pytest.approx(odom_noise_std, rel=0.05)
    assert std_yaw == pytest.approx(odom_noise_std * 0.5, rel=0.05)


def test_action_type_preserved():
    action = Action(v_fwd=1.0, v_lat=0.5, yaw_rate=0.2)
    rng = np.random.default_rng(0)
    out = apply_odom_noise(action, 0.1, rng)
    assert isinstance(out, Action)
