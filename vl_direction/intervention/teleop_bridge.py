"""
Adapter that would accept teleop events from the existing keyboard-teleop
system (next.md sec 6.2) and flip SessionMode accordingly. Does NOT
implement teleop itself and does NOT modify kb_teleop_env.py -- that file's
on_key() handler (lines ~237-286) has no existing hook for external
listeners, and wiring one in is an explicit non-goal of this pass. A future
integration would just call bridge.notify_teleop_event(key=key) from inside
on_key() after its existing WASD handling.
"""

import time
from dataclasses import dataclass
from typing import List, Optional

from vl_direction import config
from vl_direction.intervention.mode_flag import SessionMode, get_current_mode, set_mode


@dataclass
class TeleopEvent:
    key: str
    timestamp: float
    shadow_vl_direction: Optional[str] = (
        None  # optional sec 6.4 shadow-mode comparison value
    )


class TeleopBridge:
    def __init__(
        self, resume_timeout_s: float = config.TELEOP_RESUME_AUTONOMY_TIMEOUT_S
    ):
        self.resume_timeout_s = resume_timeout_s
        self.events: List[TeleopEvent] = []
        self._last_event_time: Optional[float] = None

    def notify_teleop_event(
        self,
        key: str,
        timestamp: Optional[float] = None,
        shadow_vl_direction: Optional[str] = None,
    ) -> TeleopEvent:
        timestamp = timestamp if timestamp is not None else time.monotonic()
        event = TeleopEvent(
            key=key, timestamp=timestamp, shadow_vl_direction=shadow_vl_direction
        )
        self.events.append(event)
        self._last_event_time = timestamp
        set_mode(SessionMode.HUMAN_INTERVENED)
        return event

    def resume_autonomy(self) -> None:
        set_mode(SessionMode.AUTONOMOUS)
        self._last_event_time = None

    def tick(self, now: Optional[float] = None) -> None:
        """Call periodically from the orchestrator loop: auto-resumes
        AUTONOMOUS after resume_timeout_s of no teleop input."""
        if (
            get_current_mode() != SessionMode.HUMAN_INTERVENED
            or self._last_event_time is None
        ):
            return
        now = now if now is not None else time.monotonic()
        if now - self._last_event_time >= self.resume_timeout_s:
            self.resume_autonomy()


if __name__ == "__main__":
    from vl_direction.intervention.mode_flag import reset

    reset()
    bridge = TeleopBridge(resume_timeout_s=0.1)
    assert get_current_mode() == SessionMode.AUTONOMOUS
    bridge.notify_teleop_event(key="a", timestamp=0.0)
    assert get_current_mode() == SessionMode.HUMAN_INTERVENED
    bridge.tick(now=0.05)
    assert (
        get_current_mode() == SessionMode.HUMAN_INTERVENED
    ), "should not resume before timeout"
    bridge.tick(now=0.2)
    assert (
        get_current_mode() == SessionMode.AUTONOMOUS
    ), "should auto-resume after timeout"
    print("OK: teleop_bridge mode transitions behave as expected.")
    reset()
