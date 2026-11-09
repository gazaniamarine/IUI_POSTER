import numpy as np

from sam_vla.core.belief_tracking import DEFAULT_SIGMA_VISIBLE, BeliefGoalTracker
from sam_vla.core.types import Action

_STAND_STILL = Action(v_fwd=0.0, v_lat=0.0, yaw_rate=0.0)


def _visible_mask(h=64, w=64):
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[28:36, 28:36] = 1
    depth = np.full((h, w), 5.0, dtype=np.float32)
    return mask, depth


def test_uncertainty_starts_at_floor():
    tracker = BeliefGoalTracker(hfov_deg=90.0, odom_noise=0.1)
    assert tracker.uncertainty_value() == DEFAULT_SIGMA_VISIBLE


def test_uncertainty_grows_while_unseen():
    tracker = BeliefGoalTracker(hfov_deg=90.0, odom_noise=0.1, seed=0)
    mask, depth = _visible_mask()
    assert tracker.observe(mask, depth) is True

    before = tracker.uncertainty_value()
    tracker.propagate(_STAND_STILL, dt=1.0)
    after = tracker.uncertainty_value()
    assert after > before


def test_uncertainty_growth_accelerates_with_odom_noise_growth_rate():
    tracker = BeliefGoalTracker(
        hfov_deg=90.0, odom_noise=0.1, odom_noise_growth_rate=1.0, seed=0
    )
    mask, depth = _visible_mask()
    tracker.observe(mask, depth)

    tracker.propagate(_STAND_STILL, dt=1.0)
    first_increment = tracker.uncertainty_value()
    tracker.propagate(_STAND_STILL, dt=1.0)
    second_increment = tracker.uncertainty_value() - first_increment

    assert second_increment > first_increment - DEFAULT_SIGMA_VISIBLE


def test_uncertainty_resets_on_fresh_sighting():
    tracker = BeliefGoalTracker(hfov_deg=90.0, odom_noise=0.5, seed=0)
    mask, depth = _visible_mask()
    tracker.observe(mask, depth)

    for _ in range(5):
        tracker.propagate(_STAND_STILL, dt=1.0)
    assert tracker.uncertainty_value() > DEFAULT_SIGMA_VISIBLE

    tracker.observe(mask, depth)
    assert tracker.uncertainty_value() == DEFAULT_SIGMA_VISIBLE


def test_uncertainty_does_not_grow_before_first_sighting():
    tracker = BeliefGoalTracker(hfov_deg=90.0, odom_noise=0.5, seed=0)
    tracker.propagate(_STAND_STILL, dt=1.0)
    assert tracker.uncertainty_value() == DEFAULT_SIGMA_VISIBLE
