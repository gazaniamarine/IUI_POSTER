"""
Periodic SAM3 segmentation for the multi-goal path (see next.md and the
approved plan at /home/gpu/.claude/plans/formulate-the-plan-next-md-stateless-thimble.md).

Not a live single-frame-append tracker: `Sam3BasePredictor.start_session`
loads a fixed list of already-written frame files at session-start
(`load_video_frames`, packages/sam3/sam3/model/io_utils.py:118) and there is
no public API to append one live frame to an already-open session. So
`Sam3GoalTracker` simulates "run segmentation every N steps" as the
"batched re-window" cycle validated in bench_sam3_window.py: keep a ring
buffer of recent RGB frames, on each `resegment()` call write whatever is
currently in the buffer to a scratch folder, open a *fresh* SAM3 session,
`add_prompt(text=term)` once per vocabulary term, `propagate_in_video()`
forward to the last frame, read that frame's masks, close the session.

`resegment()` does not require a full window — it writes however many
frames are currently buffered, so it can be called immediately at step 0
with just one frame (the plan's "one immediate resegment+classify pass on
frame 0 so the episode doesn't start goal-less").

Categorization (which vocabulary term a mask "is") is intentionally not
returned here — that's CLIP's job (clip_goal_classifier.py), run per mask
on top of these results, per the plan's design decision that SAM3 narrows
the candidate set by concept while CLIP scores/labels/re-identifies within
it.
"""

import os
import shutil
from collections import deque

import numpy as np
import torch
from PIL import Image


class Sam3GoalTracker:
    """Builds a SAM3.1 predictor once; call `push_frame` every step and
    `resegment` on the periodic cadence."""

    def __init__(
        self,
        vocab_terms: list[str],
        window_frames: int = 5,
        version: str = "sam3.1",
        checkpoint_path: str | None = None,
        compile_: bool = True,
        use_fa3: bool = False,
        output_prob_thresh: float = 0.5,
        scratch_dir: str = "/tmp/segment-anything-3/goal_tracker",
    ):
        if not vocab_terms:
            raise ValueError("vocab_terms must be non-empty")
        if window_frames < 1:
            raise ValueError("window_frames must be >= 1")

        from sam3 import build_sam3_predictor

        self.vocab_terms = list(vocab_terms)
        self.window_frames = window_frames
        self.output_prob_thresh = output_prob_thresh
        self.scratch_dir = scratch_dir
        self.ring_buffer: deque[np.ndarray] = deque(maxlen=window_frames)

        build_kwargs = dict(
            version=version,
            compile=compile_,
            async_loading_frames=False,
            use_fa3=use_fa3,
        )
        if checkpoint_path:
            build_kwargs["checkpoint_path"] = checkpoint_path
        if version == "sam3.1":
            build_kwargs["warm_up"] = compile_
            # each vocabulary term can match several instances of that
            # concept in one frame (open-vocab detection, not 1 obj per term)
            build_kwargs["max_num_objects"] = max(len(self.vocab_terms), 1) * 4
        self.predictor = build_sam3_predictor(**build_kwargs)

    def push_frame(self, rgb: np.ndarray) -> None:
        """Append one live RGB frame (H, W, 3 uint8) to the ring buffer."""
        self.ring_buffer.append(rgb)

    def resegment(self, step: int, out_dir: str | None = None) -> dict[int, np.ndarray]:
        """One batched-re-window cycle over the current ring buffer contents.

        Returns {sam3_obj_id: mask} for the last (most recent) frame in the
        window — bool ndarrays of shape (H, W). Empty if no frames have been
        pushed yet.

        If `out_dir` is given, the written window is left on disk for the
        caller to inspect (e.g. calibrate_clip_stock_scene.py); otherwise a
        scratch directory is used and cleaned up before returning.
        """
        if not self.ring_buffer:
            return {}

        auto_dir = out_dir is None
        window_dir = (
            out_dir if out_dir else os.path.join(self.scratch_dir, f"step_{step:08d}")
        )
        self._write_window(window_dir)

        try:
            resp = self.predictor.handle_request(
                {"type": "start_session", "resource_path": window_dir}
            )
            session_id = resp["session_id"]

            for term in self.vocab_terms:
                self.predictor.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": 0,
                        "text": term,
                        "output_prob_thresh": self.output_prob_thresh,
                    }
                )

            last_outputs = None
            last_frame_idx = -1
            for step_result in self.predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "forward",
                }
            ):
                if step_result["frame_index"] >= last_frame_idx:
                    last_frame_idx = step_result["frame_index"]
                    last_outputs = step_result["outputs"]
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            self.predictor.handle_request(
                {"type": "close_session", "session_id": session_id}
            )
        finally:
            if auto_dir:
                shutil.rmtree(window_dir, ignore_errors=True)

        if last_outputs is None:
            return {}

        out_obj_ids = last_outputs.get("out_obj_ids", [])
        out_binary_masks = last_outputs.get("out_binary_masks", [])
        return {
            int(obj_id): np.asarray(mask, dtype=bool)
            for obj_id, mask in zip(out_obj_ids, out_binary_masks)
        }

    def _write_window(self, out_dir: str) -> None:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        for i, rgb in enumerate(self.ring_buffer):
            Image.fromarray(rgb).save(os.path.join(out_dir, f"{i:03d}.jpg"))
