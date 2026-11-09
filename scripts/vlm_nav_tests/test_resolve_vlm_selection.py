"""Standalone test for resolve_vlm_selection() (imported directly, no Tk/
labelme/InteractiveCapture): runs it against an already-captured frame +
annotation pair and prints the goal/obstacle selection + resolved 3D
positions, to diff against a full interactive-flow run on the same frame
(e.g. vlm_nav_out/rgb_0000_mission.json).

Usage:
    python scripts/vlm_nav_tests/test_resolve_vlm_selection.py [frame_idx]
"""

import json
import sys

from vlm_nav_interactive import OUT_DIR, ANNOTATIONS_DIR, resolve_vlm_selection


def main():
    frame_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    rgb_path = f"{OUT_DIR}/rgb_{frame_idx:04d}.png"
    overlay_path = f"{OUT_DIR}/rgb_{frame_idx:04d}_at.png"
    annotation_path = f"{ANNOTATIONS_DIR}/rgb_{frame_idx:04d}.json"

    success, result, status_msg = resolve_vlm_selection(
        rgb_path, overlay_path, annotation_path, frame_idx
    )
    if not success:
        print(f"[test] FAILED: {status_msg}")
        sys.exit(1)

    response, goal_mesh, obstacle_meshes = result
    print("[test] resolve_vlm_selection() succeeded")
    print(
        json.dumps(
            {"vlm_response": response, "goal": goal_mesh, "obstacles": obstacle_meshes},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
