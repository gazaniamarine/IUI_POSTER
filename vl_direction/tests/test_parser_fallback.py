import pytest

from vl_direction.parser import parse_direction, parse_ghost_mask_json
from vl_direction.schemas import Direction

_ALLOWED = (Direction.LEFT, Direction.RIGHT, Direction.FRONT, Direction.BACK)


@pytest.mark.parametrize(
    "raw_text,expected",
    [
        ("LEFT", Direction.LEFT),
        ("left", Direction.LEFT),
        ("LEFT.", Direction.LEFT),
        ("I think it should go LEFT.", Direction.LEFT),
        ("Right side seems safer", Direction.RIGHT),
    ],
)
def test_parses_expected_direction(raw_text, expected):
    direction, parse_ok = parse_direction(raw_text, _ALLOWED)
    assert parse_ok is True
    assert direction == expected


def test_unparseable_text_returns_parse_ok_false():
    direction, parse_ok = parse_direction("I'm not sure what to do here.", _ALLOWED)
    assert parse_ok is False
    assert direction is None


def test_ambiguous_text_with_multiple_directions_is_unparseable():
    direction, parse_ok = parse_direction("left or maybe right", _ALLOWED)
    assert parse_ok is False
    assert direction is None


def test_binary_alphabet_rejects_front_back_only_text():
    direction, parse_ok = parse_direction("FRONT", (Direction.LEFT, Direction.RIGHT))
    assert parse_ok is False
    assert direction is None


def test_ghost_mask_parses_clean_json():
    payload, parse_ok = parse_ghost_mask_json(
        '{"u": 320, "v": 240, "radius_u_px": 60, "radius_v_px": 90}',
        (640, 480),
        3.0,
        260.0,
    )
    assert parse_ok is True
    assert payload.u == 320.0
    assert payload.v == 240.0
    assert payload.radius_u_px == 60.0
    assert payload.radius_v_px == 90.0


def test_ghost_mask_parses_json_wrapped_in_prose():
    payload, parse_ok = parse_ghost_mask_json(
        'Sure, here it is: {"u": 100, "v": 50, "radius_u_px": 20, "radius_v_px": 30} '
        "-- hope that helps!",
        (640, 480),
        3.0,
        260.0,
    )
    assert parse_ok is True
    assert payload.u == 100.0


def test_ghost_mask_clamps_out_of_bounds_values():
    payload, parse_ok = parse_ghost_mask_json(
        '{"u": -50, "v": 9999, "radius_u_px": 1, "radius_v_px": 1}',
        (640, 480),
        3.0,
        260.0,
    )
    assert parse_ok is True
    assert payload.u == 0.0
    assert payload.v == 479.0
    assert payload.radius_u_px == 3.0
    assert payload.radius_v_px == 3.0


def test_ghost_mask_clamps_oversized_radius():
    payload, parse_ok = parse_ghost_mask_json(
        '{"u": 100, "v": 100, "radius_u_px": 99999, "radius_v_px": 99999}',
        (640, 480),
        3.0,
        260.0,
    )
    assert parse_ok is True
    assert payload.radius_u_px == 260.0
    assert payload.radius_v_px == 260.0


def test_ghost_mask_missing_fields_is_unparseable():
    payload, parse_ok = parse_ghost_mask_json(
        '{"u": 100, "v": 100, "radius_u_px": 5}', (640, 480), 3.0, 260.0
    )
    assert parse_ok is False
    assert payload is None


def test_ghost_mask_no_json_is_unparseable():
    payload, parse_ok = parse_ghost_mask_json(
        "I'm not sure where the goal is.", (640, 480), 3.0, 260.0
    )
    assert parse_ok is False
    assert payload is None
