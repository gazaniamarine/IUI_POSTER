import math

import numpy as np
import pytest

from sam_vla.core.ghost_mask import (
    belief_to_bearing_range_uncertainty,
    draw_ghost_ellipse,
    draw_ghost_mask,
    project_body_point_to_pixel,
    project_or_clamp_body_point_to_pixel,
    uncertainty_to_radius_px,
)


@pytest.mark.parametrize(
    "uncertainty,min_px,max_px,scale,expected",
    [
        (0.0, 3.0, 260.0, 100.0, 3.0),  # below floor -> clamped to min_px
        (1.0, 3.0, 260.0, 100.0, 100.0),  # within range -> uncertainty * scale
        (10.0, 3.0, 260.0, 100.0, 260.0),  # above ceiling -> clamped to max_px
    ],
)
def test_uncertainty_to_radius_px_clamps(uncertainty, min_px, max_px, scale, expected):
    assert uncertainty_to_radius_px(uncertainty, min_px, max_px, scale) == pytest.approx(
        expected
    )


def test_project_straight_ahead_lands_at_frame_center():
    h, w = 480, 640
    result = project_body_point_to_pixel(forward=5.0, left=0.0, hfov_deg=90.0, h=h, w=w)
    assert result is not None
    u, v = result
    assert u == pytest.approx((w - 1) * 0.5)
    assert v == pytest.approx((h - 1) * 0.5)


def test_project_behind_camera_returns_none():
    result = project_body_point_to_pixel(forward=-1.0, left=0.0, hfov_deg=90.0, h=480, w=640)
    assert result is None


def test_project_left_of_body_lands_left_of_center():
    h, w = 480, 640
    u, _ = project_body_point_to_pixel(forward=5.0, left=2.0, hfov_deg=90.0, h=h, w=w)
    assert u < (w - 1) * 0.5


def test_project_right_of_body_lands_right_of_center():
    h, w = 480, 640
    u, _ = project_body_point_to_pixel(forward=5.0, left=-2.0, hfov_deg=90.0, h=h, w=w)
    assert u > (w - 1) * 0.5


def test_project_far_off_axis_falls_outside_frame_returns_none():
    result = project_body_point_to_pixel(forward=1.0, left=50.0, hfov_deg=90.0, h=480, w=640)
    assert result is None


def test_clamp_matches_direct_projection_when_in_view():
    h, w = 480, 640
    direct = project_body_point_to_pixel(forward=5.0, left=1.0, hfov_deg=90.0, h=h, w=w)
    clamped = project_or_clamp_body_point_to_pixel(
        forward=5.0, left=1.0, hfov_deg=90.0, h=h, w=w
    )
    assert direct == pytest.approx(clamped)


def test_clamp_off_axis_ahead_lands_on_near_side_edge():
    h, w = 480, 640
    u, v = project_or_clamp_body_point_to_pixel(
        forward=1.0, left=50.0, hfov_deg=90.0, h=h, w=w
    )
    assert u == pytest.approx(0.0)  # left-positive -> clamps to left edge
    assert v == pytest.approx((h - 1) * 0.5)

    u, v = project_or_clamp_body_point_to_pixel(
        forward=1.0, left=-50.0, hfov_deg=90.0, h=h, w=w
    )
    assert u == pytest.approx(float(w - 1))  # right-positive left -> right edge


def test_clamp_behind_but_off_to_a_side_lands_on_that_side_edge():
    h, w = 480, 640
    u, v = project_or_clamp_body_point_to_pixel(
        forward=-2.0, left=2.0, hfov_deg=90.0, h=h, w=w
    )
    assert u == pytest.approx(0.0)  # behind-left -> left edge
    assert v == pytest.approx((h - 1) * 0.5)

    u, v = project_or_clamp_body_point_to_pixel(
        forward=-2.0, left=-2.0, hfov_deg=90.0, h=h, w=w
    )
    assert u == pytest.approx(float(w - 1))  # behind-right -> right edge


def test_clamp_directly_behind_lands_on_bottom_edge():
    h, w = 480, 640
    u, v = project_or_clamp_body_point_to_pixel(
        forward=-2.0, left=0.0, hfov_deg=90.0, h=h, w=w
    )
    assert u == pytest.approx((w - 1) * 0.5)
    assert v == pytest.approx(float(h - 1))


def test_clamp_at_origin_lands_on_bottom_edge():
    h, w = 480, 640
    u, v = project_or_clamp_body_point_to_pixel(
        forward=0.0, left=0.0, hfov_deg=90.0, h=h, w=w
    )
    assert u == pytest.approx((w - 1) * 0.5)
    assert v == pytest.approx(float(h - 1))


def test_draw_ghost_mask_is_non_destructive_copy():
    rgb = np.full((20, 20, 3), 128, dtype=np.uint8)
    original = rgb.copy()
    annotated = draw_ghost_mask(rgb, u=10, v=10, radius_px=3)
    assert np.array_equal(rgb, original)  # input untouched
    assert not np.array_equal(annotated, rgb)  # something was actually drawn


def test_draw_ghost_mask_blends_green_at_center_and_leaves_far_pixels_alone():
    rgb = np.full((40, 40, 3), 100, dtype=np.uint8)
    annotated = draw_ghost_mask(rgb, u=20, v=20, radius_px=5, alpha=0.5)

    center = annotated[20, 20].astype(np.float32)
    expected_center = 100 * 0.5 + np.array([0, 255, 0], dtype=np.float32) * 0.5
    assert center == pytest.approx(expected_center, abs=1.0)

    corner = annotated[0, 0]
    assert tuple(corner) == (100, 100, 100)


def test_draw_ghost_mask_translucent_not_opaque():
    rgb = np.full((20, 20, 3), 100, dtype=np.uint8)
    annotated = draw_ghost_mask(rgb, u=10, v=10, radius_px=5, alpha=0.45)
    center = annotated[10, 10]
    assert center[1] > 100  # green channel pulled up toward the overlay color
    assert center[0] < 100  # but not fully replaced -- background still shows through


def test_draw_ghost_ellipse_matches_circle_when_radii_equal():
    rgb = np.full((40, 40, 3), 100, dtype=np.uint8)
    circle = draw_ghost_mask(rgb, u=20, v=20, radius_px=6, alpha=0.5)
    ellipse = draw_ghost_ellipse(
        rgb, u=20, v=20, radius_u_px=6, radius_v_px=6, alpha=0.5
    )
    assert np.array_equal(circle, ellipse)


def test_draw_ghost_ellipse_is_non_destructive_copy():
    rgb = np.full((20, 20, 3), 128, dtype=np.uint8)
    original = rgb.copy()
    annotated = draw_ghost_ellipse(rgb, u=10, v=10, radius_u_px=3, radius_v_px=8)
    assert np.array_equal(rgb, original)
    assert not np.array_equal(annotated, rgb)


def test_draw_ghost_ellipse_stretches_along_larger_axis():
    rgb = np.full((60, 60, 3), 100, dtype=np.uint8)
    annotated = draw_ghost_ellipse(
        rgb, u=30, v=30, radius_u_px=20, radius_v_px=4, alpha=1.0
    )
    # far along the wide (u) axis should be painted, far along the narrow (v) axis should not
    assert tuple(annotated[30, 45]) == (0, 255, 0)
    assert tuple(annotated[45, 30]) == (100, 100, 100)


def test_belief_to_bearing_range_uncertainty_straight_ahead_isotropic():
    mean = np.array([5.0, 0.0])
    cov = np.eye(2) * (0.5**2)
    bearing_deg, distance_m, bearing_unc_deg, distance_unc_m = (
        belief_to_bearing_range_uncertainty(mean, cov)
    )
    assert bearing_deg == pytest.approx(0.0)
    assert distance_m == pytest.approx(5.0)
    # isotropic covariance -> radial and tangential sigma both 0.5, distance
    # uncertainty is the radial component directly
    assert distance_unc_m == pytest.approx(0.5)
    # tangential sigma (0.5) at range 5 -> arctan-small-angle ~= 5.73 deg
    assert bearing_unc_deg == pytest.approx(math.degrees(0.5 / 5.0))


def test_belief_to_bearing_range_uncertainty_left_positive():
    mean = np.array([5.0, 3.0])  # forward, left
    cov = np.eye(2) * 0.01
    bearing_deg, _, _, _ = belief_to_bearing_range_uncertainty(mean, cov)
    assert bearing_deg > 0.0  # left is positive bearing, matches BeliefGoalTracker


def test_belief_to_bearing_range_uncertainty_anisotropic_along_bearing():
    # all variance along the forward axis (radial, since bearing is 0) -> should
    # show up entirely as distance uncertainty, none as bearing uncertainty
    mean = np.array([5.0, 0.0])
    cov = np.array([[4.0, 0.0], [0.0, 0.0]])
    bearing_deg, distance_m, bearing_unc_deg, distance_unc_m = (
        belief_to_bearing_range_uncertainty(mean, cov)
    )
    assert distance_unc_m == pytest.approx(2.0)
    assert bearing_unc_deg == pytest.approx(0.0)


def test_belief_to_bearing_range_uncertainty_at_origin_falls_back_isotropic():
    mean = np.array([0.0, 0.0])
    cov = np.eye(2) * (1.0**2)
    bearing_deg, distance_m, bearing_unc_deg, distance_unc_m = (
        belief_to_bearing_range_uncertainty(mean, cov)
    )
    assert distance_m == pytest.approx(0.0)
    assert bearing_unc_deg == pytest.approx(180.0)
    assert distance_unc_m == pytest.approx(1.0)
