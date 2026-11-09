import numpy as np

from vl_direction.client import MockInternVLClient
from vl_direction.schemas import UncertaintyStatus
from vl_direction.uncertainty_session import UncertaintySession

_FRAME = np.zeros((4, 4, 3), dtype=np.uint8)


def _session():
    return UncertaintySession(
        episode_id="test-episode",
        covariance_threshold=1.0,
        covariance_value=2.5,
        client=MockInternVLClient(canned_response="dunes ahead"),
    )


def test_request_human_heading_returns_needs_human_input():
    session = _session()
    result = session.request_human_heading(_FRAME)
    assert result.uncertainty_payload.status == UncertaintyStatus.NEEDS_HUMAN_INPUT
    assert result.uncertainty_payload.attempt == 0
    assert result.direction is None


def test_submit_heading_returns_heading_directive():
    session = _session()
    session.request_human_heading(_FRAME)
    result = session.submit_heading(angle_deg=35.0)
    assert result.uncertainty_payload.status == UncertaintyStatus.HEADING_DIRECTIVE
    assert result.uncertainty_payload.heading_deg == 35.0
    assert result.uncertainty_payload.attempt == 0


def test_submit_heading_accepts_angle_range():
    session = _session()
    result = session.submit_heading(angle_range_deg=(70.0, 80.0))
    assert result.uncertainty_payload.heading_range_deg == (70.0, 80.0)
    assert result.uncertainty_payload.heading_deg is None


def test_retry_increments_attempt_counter():
    session = _session()
    assert session.attempt == 0
    r1 = session.retry(_FRAME)
    assert session.attempt == 1
    assert r1.uncertainty_payload.attempt == 1
    r2 = session.retry(_FRAME)
    assert session.attempt == 2
    assert r2.uncertainty_payload.attempt == 2


def test_submit_heading_uses_default_max_units_when_not_specified():
    from vl_direction import config

    session = _session()
    result = session.submit_heading(angle_deg=10.0)
    assert result.uncertainty_payload.max_units == config.DEFAULT_MAX_UNITS


def test_submit_heading_respects_explicit_max_units():
    session = _session()
    result = session.submit_heading(angle_deg=10.0, max_units=12.0)
    assert result.uncertainty_payload.max_units == 12.0
