"""
Session-level flag tracking whether the rover is currently under autonomous
VL-directive control or human teleop override, for the HCI ablation study
(next.md sec 6.1). This is genuinely new state -- no AUTONOMOUS/
HUMAN_INTERVENED concept exists elsewhere in the codebase to reuse.

A module-global guarded by a lock, not a class instance, because the flag is
read by directive_engine.query() (main rollout thread) and written by
teleop_bridge.py (a UI-callback thread) -- both need to see the same session
state without the caller threading an object through every call site.
"""

import threading
from enum import Enum


class SessionMode(str, Enum):
    AUTONOMOUS = "autonomous"
    HUMAN_INTERVENED = "human_intervened"


_lock = threading.Lock()
_current_mode = SessionMode.AUTONOMOUS


def get_current_mode() -> SessionMode:
    with _lock:
        return _current_mode


def set_mode(mode: SessionMode) -> None:
    global _current_mode
    with _lock:
        _current_mode = mode


def reset() -> None:
    """Forces AUTONOMOUS. Call at episode boundaries / in test teardown so
    state doesn't leak across episodes or tests."""
    set_mode(SessionMode.AUTONOMOUS)


if __name__ == "__main__":
    assert get_current_mode() == SessionMode.AUTONOMOUS
    set_mode(SessionMode.HUMAN_INTERVENED)
    assert get_current_mode() == SessionMode.HUMAN_INTERVENED
    reset()
    assert get_current_mode() == SessionMode.AUTONOMOUS
    print("OK: mode_flag transitions behave as expected.")
