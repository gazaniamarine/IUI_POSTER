"""
Standalone per-episode JSONL logging for vl_direction, mirroring the STYLE
of sam_vla/logging/episode_logger.py (async-queue-drained-by-daemon-thread,
one JSON object per line) without importing it -- this module must stay
self-contained (next.md sec 7). One JSONL stream (directives.jsonl) is
enough here since there's only one event kind, unlike sam_vla's logger which
has four streams for different rollout event types.
"""

import dataclasses
import json
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from vl_direction import config
from vl_direction.schemas import VLDirectiveResult

_SENTINEL = object()


def _json_default(obj: Any) -> Any:
    # Tolerate numpy scalars/arrays showing up in logged payloads, same
    # rationale as sam_vla/logging/episode_logger.py's identical helper.
    item = getattr(obj, "item", None)
    if callable(item) and hasattr(obj, "shape") and obj.shape == ():
        return item()
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class VLDirectionEpisodeLogger:
    def __init__(
        self,
        run_id: str,
        config_dict: Dict[str, Any],
        log_root: str = config.DEFAULT_LOG_ROOT,
        flush_interval_s: float = 1.0,
    ):
        self.run_id = run_id
        self.run_dir = Path(log_root) / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self._config = dict(config_dict)
        self._config.setdefault("run_id", run_id)
        self._config.setdefault("timestamp_start", _now_iso())
        (self.run_dir / "config.json").write_text(
            json.dumps(self._config, indent=2, default=_json_default)
        )

        self._directive_count = 0
        self._file = open(self.run_dir / "directives.jsonl", "a", encoding="utf-8")
        self._queue: "queue.Queue" = queue.Queue()
        self._flush_interval_s = flush_interval_s
        self._closed = False
        self._worker = threading.Thread(
            target=self._drain_loop, name=f"vl-direction-logger-{run_id}", daemon=True
        )
        self._worker.start()

    def _drain_loop(self) -> None:
        last_flush = time.monotonic()
        while True:
            try:
                item = self._queue.get(timeout=self._flush_interval_s)
            except queue.Empty:
                item = None

            if item is _SENTINEL:
                self._file.flush()
                break
            if item is not None:
                self._file.write(
                    json.dumps(item, separators=(",", ":"), default=_json_default)
                    + "\n"
                )

            now = time.monotonic()
            if now - last_flush >= self._flush_interval_s:
                self._file.flush()
                last_flush = now

        self._file.close()

    def log_directive(self, result: VLDirectiveResult) -> None:
        self._queue.put(dataclasses.asdict(result))
        self._directive_count += 1

    def finalize(self, summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        computed = {
            "run_id": self.run_id,
            "timestamp_end": _now_iso(),
            "num_directives": self._directive_count,
        }
        computed.update(summary or {})
        (self.run_dir / "summary.json").write_text(
            json.dumps(computed, indent=2, default=_json_default)
        )
        self.close()
        return computed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put(_SENTINEL)
        self._worker.join(timeout=10.0)

    def __enter__(self) -> "VLDirectionEpisodeLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self._closed:
            self.finalize(
                {"termination_reason": "exception" if exc_type else "unfinalized"}
            )
        return False


if __name__ == "__main__":
    import shutil
    import tempfile

    import numpy as np

    from vl_direction.client import MockInternVLClient
    from vl_direction.directive_engine import query
    from vl_direction.schemas import CBFContext

    tmp_root = tempfile.mkdtemp(prefix="vl_direction_logger_demo_")
    try:
        logger = VLDirectionEpisodeLogger(
            "demo-run", {"study_arm": "vl_only"}, log_root=tmp_root
        )
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        for _ in range(3):
            result = query(
                "cbf",
                [frame],
                CBFContext(bbox_xyxy=(1, 1, 2, 2), frame_wh=(4, 4)),
                "demo-run",
                client=MockInternVLClient(canned_response="LEFT"),
            )
            logger.log_directive(result)
        summary = logger.finalize()
        print("summary:", summary)

        with open(Path(tmp_root) / "demo-run" / "directives.jsonl") as f:
            lines = f.readlines()
        assert len(lines) == 3
        json.loads(lines[0])
        print("OK: directives.jsonl has 3 well-formed lines.")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
