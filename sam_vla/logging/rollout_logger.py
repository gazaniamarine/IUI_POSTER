import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from sam_vla.core.goal_geometry import GoalPosition
from sam_vla.core.types import Action, Detection, GoalSpec, Observation, Pose
from sam_vla.env.sim_utils import distance_to_goal


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RolloutLogger:
    def __init__(self):
        self._rgb_frames = []
        self._vis_frames = []
        self._poses = []
        self._actions = []
        self._frame_indices = []
        self._timestamps = []
        self._distances_to_goal = []
        self._manifest = {
            "start_time": _now_iso(),
            "steps": [],
            "goal_resolution": None,
        }
        self._goal_position: Optional[GoalPosition] = None

    def log_goal_resolution(
        self,
        goal_spec: GoalSpec,
        vlm_result: dict,
        goal_position: Optional[GoalPosition] = None,
    ) -> None:
        """Record the one-shot first-frame VLM goal-selection call (raw result + resolved spec)."""
        self._goal_position = goal_position
        self._manifest["goal_resolution"] = {
            "vlm_result": vlm_result,
            "goal_bbox_norm": list(goal_spec.goal_bbox_norm),
            "instruction_text": goal_spec.instruction_text,
            "goal_position": list(goal_position) if goal_position is not None else None,
            "timestamp": _now_iso(),
        }

    def log_step(
        self,
        obs: Observation,
        action: Action,
        pose: Pose,
        vla_result: Optional[dict] = None,
        vis_rgb: Optional[np.ndarray] = None,
        uncertainty_event: Optional[dict] = None,
    ) -> None:
        """vis_rgb, if given, is the goal/obstacle-overlaid version of obs.rgb
        (see perception.semantic_overlay.overlay_semantic_masks) for the same
        step -- used in place of the raw frame by save_frames/save_video so the
        dumped video shows what the rover's mask-conditioned policy is seeing.

        uncertainty_event, if given, is a Study 1 (next.md) uncertainty_trigger
        payload -- one-off per rising-edge crossing of --uncertainty-threshold,
        not per step. None on every other step."""
        rgb = np.asarray(obs.rgb, dtype=np.uint8)
        pose_tuple = (pose.x, pose.y, pose.z, pose.yaw)
        action_tuple = (action.v_fwd, action.v_lat, action.yaw_rate)
        timestamp = _now_iso()

        dist = (
            distance_to_goal(pose, self._goal_position)
            if self._goal_position is not None
            else None
        )

        self._rgb_frames.append(rgb)
        if vis_rgb is not None:
            self._vis_frames.append(np.asarray(vis_rgb, dtype=np.uint8))
        self._poses.append(pose_tuple)
        self._actions.append(action_tuple)
        self._frame_indices.append(obs.frame_idx)
        self._timestamps.append(timestamp)
        self._distances_to_goal.append(dist if dist is not None else float("nan"))

        self._manifest["steps"].append(
            {
                "frame_idx": obs.frame_idx,
                "pose": list(pose_tuple),
                "action": list(action_tuple),
                "vla_result": vla_result,
                "goal_position": (
                    list(self._goal_position)
                    if self._goal_position is not None
                    else None
                ),
                "distance_to_goal": dist,
                "timestamp": timestamp,
                "uncertainty_event": uncertainty_event,
            }
        )

    def flush(self, out_dir: str) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        npz_path = out_path / "rollout.npz"
        manifest_path = out_path / "manifest.json"

        np.savez(
            npz_path,
            rgb_frames=np.stack(self._rgb_frames),
            poses=np.array(self._poses, dtype=np.float64),
            actions=np.array(self._actions, dtype=np.float64),
            frame_indices=np.array(self._frame_indices, dtype=np.int64),
            timestamps=np.array(self._timestamps, dtype=str),
            distances_to_goal=np.array(self._distances_to_goal, dtype=np.float64),
        )

        with open(manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

        print(
            f"RolloutLogger: flushed {len(self._frame_indices)} steps -> "
            f"{npz_path}, {manifest_path}"
        )

    def save_sam_first_frame(
        self,
        rgb: np.ndarray,
        detections: list[Detection],
        goal_spec: GoalSpec,
        out_dir: str,
    ) -> None:
        """Save the raw first frame and a SAM-annotated copy (all detection boxes
        in red, the VLM-chosen goal box in green) to out_dir."""
        import cv2
        import imageio.v3 as iio

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        rgb = np.asarray(rgb, dtype=np.uint8)
        iio.imwrite(out_path / "first_frame.png", rgb)

        height, width = rgb.shape[:2]
        annotated = rgb.copy()
        for det in detections:
            x0, y0, x1, y1 = det.bbox_norm
            pt0 = (int(x0 * width), int(y0 * height))
            pt1 = (int(x1 * width), int(y1 * height))
            cv2.rectangle(annotated, pt0, pt1, (255, 0, 0), 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (pt0[0], max(pt0[1] - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

        gx0, gy0, gx1, gy1 = goal_spec.goal_bbox_norm
        gpt0 = (int(gx0 * width), int(gy0 * height))
        gpt1 = (int(gx1 * width), int(gy1 * height))
        cv2.rectangle(annotated, gpt0, gpt1, (0, 255, 0), 3)
        cv2.putText(
            annotated,
            "GOAL",
            (gpt0[0], max(gpt0[1] - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        annotated_path = out_path / "first_frame_annotated.png"
        iio.imwrite(annotated_path, annotated)
        print(
            f"RolloutLogger: saved first frame + SAM-annotated frame "
            f"({len(detections)} detections) -> {out_path / 'first_frame.png'}, {annotated_path}"
        )

    def _output_frames(self) -> list:
        """Goal/obstacle-overlaid frames if every step logged one, else the raw
        RGB frames (e.g. for rollouts that never pass vis_rgb to log_step)."""
        if len(self._vis_frames) == len(self._rgb_frames):
            return self._vis_frames
        return self._rgb_frames

    def save_frames(self, out_dir: str) -> None:
        """Dump each logged frame (goal/obstacle-overlaid if available) as a
        PNG (frame_<frame_idx>.png)."""
        import imageio.v3 as iio

        frames_dir = Path(out_dir) / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        frames = self._output_frames()
        for frame_idx, rgb in zip(self._frame_indices, frames):
            iio.imwrite(frames_dir / f"frame_{frame_idx:06d}.png", rgb)

        print(f"RolloutLogger: saved {len(frames)} frames -> {frames_dir}")

    def save_video(self, out_dir: str, fps: int = 10) -> None:
        """Encode the logged frames (goal/obstacle-overlaid if available) into
        rollout.mp4, muxing via PyAV directly rather than imageio.v3's pyav
        plugin: PyAV >=15 leaves stream.codec_context.time_base unset until
        the first frame is muxed, which that plugin doesn't account for and
        crashes on (`AttributeError: 'NoneType' object has no attribute
        'numerator'` from av.frame.Frame.time_base.__set__) -- setting it
        explicitly here sidesteps the incompatibility."""
        import av
        from fractions import Fraction

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        video_path = out_path / "rollout.mp4"

        frames = self._output_frames()
        height, width = np.asarray(frames[0]).shape[:2]

        container = av.open(str(video_path), mode="w")
        stream = container.add_stream("libx264", rate=fps)
        stream.height = height
        stream.width = width
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, fps)

        for rgb in frames:
            frame = av.VideoFrame.from_ndarray(
                np.asarray(rgb, dtype=np.uint8), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()

        print(
            f"RolloutLogger: saved video ({len(frames)} frames @ {fps}fps) -> {video_path}"
        )


if __name__ == "__main__":
    import tempfile

    from sam_vla.core.types import GoalSpec

    logger = RolloutLogger()
    logger.log_goal_resolution(
        goal_spec=GoalSpec(
            goal_bbox_norm=(0.4, 0.4, 0.6, 0.6),
            obstacle_bboxes_norm=[],
            instruction_text="Navigate to the rock target.",
        ),
        vlm_result={"goal_index": 1, "reasoning": "closest large rock"},
        goal_position=(10.0, 0.0, 0.0),
    )

    for i in range(5):
        obs = Observation(
            rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            depth=None,
            pose=Pose(x=float(i), y=0.0, z=0.0, yaw=0.0),
            frame_idx=i,
        )
        action = Action(v_fwd=0.5, v_lat=0.0, yaw_rate=0.1 * i)
        pose = Pose(x=float(i), y=0.0, z=0.0, yaw=0.0)
        vla_result = {"v_fwd": 0.5, "v_lat": 0.0, "yaw_rate": 0.1 * i}
        logger.log_step(obs, action, pose, vla_result=vla_result)

    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.flush(tmp_dir)

        npz_data = np.load(Path(tmp_dir) / "rollout.npz")
        with open(Path(tmp_dir) / "manifest.json") as f:
            manifest = json.load(f)

        print("rgb_frames shape:", npz_data["rgb_frames"].shape)
        print("poses shape:", npz_data["poses"].shape)
        print("actions shape:", npz_data["actions"].shape)
        print("frame_indices shape:", npz_data["frame_indices"].shape)
        print("timestamps shape:", npz_data["timestamps"].shape)
        print("distances_to_goal:", npz_data["distances_to_goal"])
        print("manifest steps:", len(manifest["steps"]))
        print("manifest goal_resolution:", manifest["goal_resolution"])

        assert npz_data["rgb_frames"].shape[0] == 5
        assert npz_data["poses"].shape == (5, 4)
        assert npz_data["actions"].shape == (5, 3)
        assert npz_data["distances_to_goal"].shape == (5,)
        assert npz_data["distances_to_goal"][0] == 10.0
        assert manifest["goal_resolution"]["vlm_result"]["goal_index"] == 1
        assert manifest["steps"][2]["vla_result"]["yaw_rate"] == 0.2
        assert npz_data["frame_indices"].shape == (5,)
        assert npz_data["timestamps"].shape == (5,)
        assert len(manifest["steps"]) == 5
        print("Round-trip check passed: step counts match.")
