"""Standalone test for goal_pixel_ratio() (imported from
scripts/vlm_nav_tests/rollout_navdp.py): runs it against a mask image loaded
from disk and prints goal_px/rest_px/frame_fraction/goal_to_rest_ratio for
eyeballing against the mask (e.g. mars_belief_demo1/frames/mask_0006.png).

Usage:
    python scripts/habitat_tests/test_goal_pixel_ratio.py [path/to/mask.png]
"""

import sys

import numpy as np
from PIL import Image

from rollout_navdp import goal_pixel_ratio

DEFAULT_MASK_PATH = "navdp_rollout20260713_121315/frames/mask_0146.png"


def load_goal_mask(path: str) -> np.ndarray:
    """Load a mask image and collapse it to a single-channel uint8 array (nonzero = goal pixel)."""
    im = Image.open(path).convert("L")
    return np.array(im, dtype=np.uint8)


def main():
    mask_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MASK_PATH

    goal_mask = load_goal_mask(mask_path)
    result = goal_pixel_ratio(goal_mask)

    total_px = goal_mask.shape[0] * goal_mask.shape[1]
    assert (
        result["goal_px"] + result["rest_px"] == total_px
    ), "goal_px + rest_px must equal total pixel count"
    assert 0.0 <= result["frame_fraction"] <= 1.0, "frame_fraction must be in [0, 1]"
    assert result["goal_px"] == int(
        np.count_nonzero(goal_mask)
    ), "goal_px must match nonzero count in mask"

    print(f"[test] mask: {mask_path} ({goal_mask.shape[1]}x{goal_mask.shape[0]})")
    print(f"[test] goal_px={result['goal_px']} rest_px={result['rest_px']}")
    print(f"[test] frame_fraction={result['frame_fraction']:.4f}")
    print(f"[test] goal_to_rest_ratio={result['goal_to_rest_ratio']:.4f}")
    print(f"[test] goal percentage={result['goal_to_rest_ratio'] * 100:.2f}%")
    print("[test] PASSED")


if __name__ == "__main__":
    main()
