"""
Manual end-to-end verification script (not part of pytest): exercises
directive_engine.query() for all three modes -- including both phases of
the uncertainty sub-flow via UncertaintySession -- using MockInternVLClient,
so it runs with no live model or network. Run with:
    python -m vl_direction.smoke_test
"""

import numpy as np

from vl_direction.client import MockInternVLClient
from vl_direction.directive_engine import query
from vl_direction.intervention.mode_flag import SessionMode, reset, set_mode
from vl_direction.schemas import (
    CBFContext,
    Direction,
    ExplorationContext,
    IdentityToken,
    UncertaintyStatus,
)
from vl_direction.uncertainty_session import UncertaintySession

_FRAME = np.zeros((8, 8, 3), dtype=np.uint8)
_EPISODE_ID = "smoke-test-episode"


def _check_cbf():
    result = query(
        "cbf",
        [_FRAME],
        CBFContext(bbox_xyxy=(1, 1, 4, 4), frame_wh=(8, 8)),
        _EPISODE_ID,
        client=MockInternVLClient(canned_response="LEFT"),
    )
    assert result.identity_token == IdentityToken.CBF
    assert result.direction == Direction.LEFT
    assert result.parse_ok is True
    assert result.uncertainty_payload is None
    print("cbf ->", result)


def _check_exploration():
    result = query(
        "exploration",
        [_FRAME],
        ExplorationContext(
            task_str="find the sample site", vague_hint="try somewhere else"
        ),
        _EPISODE_ID,
        client=MockInternVLClient(canned_response="FRONT"),
    )
    assert result.identity_token == IdentityToken.EXPLORATION
    assert result.direction == Direction.FRONT
    assert result.parse_ok is True
    print("exploration ->", result)


def _check_exploration_parse_failure():
    result = query(
        "exploration",
        [_FRAME],
        ExplorationContext(task_str="find the sample site"),
        _EPISODE_ID,
        client=MockInternVLClient(canned_response="I'm not sure where to go."),
    )
    assert result.parse_ok is False
    assert result.direction is None
    print("exploration (unparseable) ->", result)


def _check_uncertainty():
    session = UncertaintySession(
        episode_id=_EPISODE_ID,
        covariance_threshold=1.0,
        covariance_value=3.0,
        client=MockInternVLClient(
            canned_response="rocky terrain to the left, open ground ahead"
        ),
    )
    r1 = session.request_human_heading(_FRAME)
    assert r1.identity_token == IdentityToken.UNCERTAINTY
    assert r1.direction is None
    assert r1.uncertainty_payload.status == UncertaintyStatus.NEEDS_HUMAN_INPUT
    print("uncertainty (request) ->", r1)

    r2 = session.submit_heading(angle_deg=35.0)
    assert r2.uncertainty_payload.status == UncertaintyStatus.HEADING_DIRECTIVE
    assert r2.uncertainty_payload.heading_deg == 35.0
    print("uncertainty (submit) ->", r2)

    r3 = session.retry(_FRAME)
    assert r3.uncertainty_payload.attempt == 1
    print("uncertainty (retry) ->", r3)


def _check_session_mode_tagging():
    reset()
    set_mode(SessionMode.HUMAN_INTERVENED)
    result = query(
        "cbf",
        [_FRAME],
        CBFContext(bbox_xyxy=(1, 1, 4, 4), frame_wh=(8, 8)),
        _EPISODE_ID,
        client=MockInternVLClient(canned_response="RIGHT"),
    )
    assert result.session_mode == SessionMode.HUMAN_INTERVENED
    reset()
    print("session_mode tagging ->", result.session_mode)


def main():
    _check_cbf()
    _check_exploration()
    _check_exploration_parse_failure()
    _check_uncertainty()
    _check_session_mode_tagging()
    print("\nOK: all vl_direction smoke checks passed.")


if __name__ == "__main__":
    main()
