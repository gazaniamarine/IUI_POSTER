"""Structured event log for RoverController runs -- goal detections and
goal-reached events, printed to the CLI and appended to a per-run log file
under nav/logs/ (gitignored via the repo-wide `logs` pattern in .gitignore,
same as sam_vla's other run-output directories).

Deliberately not routed through the stdlib `logging` module -- this is a
handful of one-line, human-readable events consumed by a person watching the
terminal or grepping a log file after the fact, not a general logging
subsystem.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

LOG_DIR = Path(__file__).resolve().parent / "logs"


class EventLogger:
    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.path = LOG_DIR / f"{self.run_id}.log"
        self._fh = open(self.path, "a", buffering=1)

    def log(self, event: str, **fields) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        detail = " ".join(f"{k}={v!r}" for k, v in fields.items() if v is not None)
        line = f"[{ts}] {event}" + (f"  {detail}" if detail else "")
        print(f"[nav log] {line}")
        self._fh.write(line + "\n")

    def close(self) -> None:
        self._fh.close()
