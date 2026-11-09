"""
Strict output parsers for vl_direction's modes.

parse_direction() (next.md sec 5.2/9, cbf/exploration): tries an exact
normalized match first, falls back to a single-word regex search, and only
ever returns parse_ok=True when exactly one allowed direction is identifiable
-- ambiguous text (multiple direction words present) is treated the same as
no match, per next.md's "don't guess" instruction.

parse_ghost_mask_json() (ghost_mask mode): extracts and clamps a {u, v,
radius_u_px, radius_v_px} JSON object from free-form model output -- see its
own docstring.
"""

import json
import re

from vl_direction.schemas import Direction, GhostMaskPayload

_DIRECTION_PATTERNS = {
    Direction.LEFT: re.compile(r"\bleft\b", re.IGNORECASE),
    Direction.RIGHT: re.compile(r"\bright\b", re.IGNORECASE),
    Direction.FRONT: re.compile(r"\b(front|forward|straight)\b", re.IGNORECASE),
    Direction.BACK: re.compile(r"\b(back|backward|reverse)\b", re.IGNORECASE),
}

_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_direction(raw_text: str, allowed: tuple) -> tuple:
    """Returns (direction_or_None, parse_ok)."""
    cleaned = raw_text.strip().strip(".!\"'").upper()
    for d in allowed:
        if cleaned == d.value:
            return d, True

    matches = [d for d in allowed if _DIRECTION_PATTERNS[d].search(raw_text)]
    if len(matches) == 1:
        return matches[0], True
    return None, False


def parse_ghost_mask_json(
    raw_text: str,
    frame_wh: tuple,
    min_radius_px: float,
    max_radius_px: float,
) -> tuple:
    """Extracts {"u", "v", "radius_u_px", "radius_v_px"} from free-form model
    output (a 3B VLM routinely wraps JSON in prose/code fences) and clamps
    into frame bounds / the radius range, so a model that overshoots still
    yields a safe, renderable mask rather than a rejected parse. Returns
    (GhostMaskPayload_or_None, parse_ok) -- parse_ok is False only when no
    JSON object with all four numeric fields could be found at all."""
    match = _JSON_OBJECT_RE.search(raw_text)
    if match is None:
        return None, False

    try:
        obj = json.loads(match.group(0))
        u = float(obj["u"])
        v = float(obj["v"])
        radius_u_px = float(obj["radius_u_px"])
        radius_v_px = float(obj["radius_v_px"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None, False

    w, h = frame_wh
    u = float(np_clip(u, 0.0, float(w - 1)))
    v = float(np_clip(v, 0.0, float(h - 1)))
    radius_u_px = float(np_clip(radius_u_px, float(min_radius_px), float(max_radius_px)))
    radius_v_px = float(np_clip(radius_v_px, float(min_radius_px), float(max_radius_px)))
    return (
        GhostMaskPayload(u=u, v=v, radius_u_px=radius_u_px, radius_v_px=radius_v_px),
        True,
    )


def np_clip(value: float, low: float, high: float) -> float:
    """Tiny local clamp -- avoids pulling numpy into this stdlib-only parser
    module just for a min/max."""
    return low if value < low else high if value > high else value


if __name__ == "__main__":
    allowed = (Direction.LEFT, Direction.RIGHT)
    battery = [
        "LEFT",
        "left",
        "LEFT.",
        "I think it should go LEFT.",
        "Right side seems safer",
        "I'm not sure.",
    ]
    for raw in battery:
        print(f"{raw!r} -> {parse_direction(raw, allowed)}")
