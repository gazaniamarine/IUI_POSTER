import json
import re

from sam_vla.core.types import Action

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fences(raw_text: str) -> str:
    return _FENCE_RE.sub("", raw_text.strip()).strip()


def _repair_truncated_json(stripped: str) -> str:
    """Best-effort repair for a response cut off mid-string-value (observed
    live: a long 'reasoning' field missing its closing quote, e.g. '...in
    place.}' instead of '...in place."}') -- the model still emitted a
    complete-looking object apart from that one missing quote. If the
    string ends inside an open, unescaped string literal (odd count of
    unescaped '"' before the end), close it, then append however many '}'
    are needed to match unclosed '{'. Only ever called as a fallback after
    a plain json.loads has already failed; returns the original text
    unchanged if this heuristic doesn't obviously apply, so the caller's
    second parse attempt just fails the same way plain parsing would."""
    text = stripped
    # Count unescaped double-quotes to tell whether we're inside a string.
    in_string = False
    escaped = False
    depth = 0
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
    if in_string:
        text += '"'
    if depth > 0:
        text += "}" * depth
    return text


def _load_json_object(raw_text: str) -> dict:
    stripped = _strip_fences(raw_text)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as first_error:
        try:
            parsed = json.loads(_repair_truncated_json(stripped))
        except json.JSONDecodeError:
            raise ValueError(
                f"Could not parse JSON from model response: {first_error}"
                f"\nraw_text={raw_text!r}"
            ) from first_error
    if not isinstance(parsed, dict):
        raise ValueError(
            f"Expected a JSON object, got {type(parsed).__name__}\nraw_text={raw_text!r}"
        )
    return parsed


def parse_select_goal_response(raw_text: str) -> dict:
    parsed = _load_json_object(raw_text)
    goal_index = parsed.get("goal_index")
    if not isinstance(goal_index, int) or isinstance(goal_index, bool):
        raise ValueError(
            f"Missing or non-integer 'goal_index' in response\nraw_text={raw_text!r}"
        )
    return parsed


def parse_goal_vocabulary_response(raw_text: str) -> dict:
    """Parse a build_goal_vocabulary_prompt response: a non-empty list of
    non-empty string terms plus an instruction sentence."""
    parsed = _load_json_object(raw_text)

    terms = parsed.get("terms")
    if not isinstance(terms, list) or not terms:
        raise ValueError(
            f"Missing or empty 'terms' list in response\nraw_text={raw_text!r}"
        )
    for term in terms:
        if not isinstance(term, str) or not term.strip():
            raise ValueError(
                f"'terms' must be a list of non-empty strings\nraw_text={raw_text!r}"
            )

    instruction_text = parsed.get("instruction_text")
    if not isinstance(instruction_text, str) or not instruction_text.strip():
        raise ValueError(
            f"Missing or non-string 'instruction_text' in response\nraw_text={raw_text!r}"
        )

    return parsed


DIRECTIONS = ("forward", "turn_left", "turn_right")


def parse_direction_response(raw_text: str) -> dict:
    """Parse a discrete-direction response (build_direction_prompt), validating
    that 'direction' is one of DIRECTIONS rather than an arbitrary continuous
    action."""
    parsed = _load_json_object(raw_text)

    direction = parsed.get("direction")
    if not isinstance(direction, str) or direction not in DIRECTIONS:
        raise ValueError(
            f"Missing or invalid 'direction' in response (expected one of "
            f"{DIRECTIONS})\nraw_text={raw_text!r}"
        )
    return parsed


NAV_COMMAND_DIRECTIONS = (
    "turn_left",
    "turn_right",
    "turn_back",
    "go_left",
    "go_right",
    "go_front",
    # Bare legacy tokens (no verb prefix) -- build_parse_nav_command_prompt no
    # longer asks for these, but mission.py's _TURN_HEADINGS still accepts
    # them as a turn-only fallback if the model ever reverts to them, so they
    # must not be rejected here either (see mission.py's docstring on why
    # this is treated as the safer turn-only reading rather than an error).
    "left",
    "right",
    "back",
    "front",
)


def parse_nav_command_response(raw_text: str) -> dict:
    """Parse a build_parse_nav_command_prompt response: a 'directions' list
    (closed vocabulary, NAV_COMMAND_DIRECTIONS) and a 'goals' list (free-text
    target phrases, kept as-is), each in the order the model infers the
    rover should pursue them. Either list may be empty, but not both."""
    parsed = _load_json_object(raw_text)

    directions = parsed.get("directions")
    if not isinstance(directions, list):
        raise ValueError(
            f"Missing or non-list 'directions' in response\nraw_text={raw_text!r}"
        )
    for direction in directions:
        if direction not in NAV_COMMAND_DIRECTIONS:
            raise ValueError(
                f"'directions' must only contain one of {NAV_COMMAND_DIRECTIONS}"
                f"\nraw_text={raw_text!r}"
            )

    goals = parsed.get("goals")
    if not isinstance(goals, list):
        raise ValueError(
            f"Missing or non-list 'goals' in response\nraw_text={raw_text!r}"
        )
    for goal in goals:
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError(
                f"'goals' must be a list of non-empty strings\nraw_text={raw_text!r}"
            )

    if not directions and not goals:
        raise ValueError(
            f"Both 'directions' and 'goals' are empty in response\nraw_text={raw_text!r}"
        )

    return {"directions": directions, "goals": goals}


def parse_split_instruction_response(raw_text: str) -> dict:
    """Parse a build_split_instruction_prompt response: a non-empty list of
    non-empty instruction strings, same list-of-strings validation as
    parse_goal_vocabulary_response's 'terms'."""
    parsed = _load_json_object(raw_text)

    instructions = parsed.get("instructions")
    if not isinstance(instructions, list) or not instructions:
        raise ValueError(
            f"Missing or empty 'instructions' list in response\nraw_text={raw_text!r}"
        )
    for instr in instructions:
        if not isinstance(instr, str) or not instr.strip():
            raise ValueError(
                f"'instructions' must be a list of non-empty strings\nraw_text={raw_text!r}"
            )

    return {"instructions": instructions}


def parse_ground_object_response(raw_text: str) -> dict:
    """Parse a build_ground_object_prompt response: either found=True with a
    valid (u, v) point, or found=False with u=v=None -- both are valid
    outcomes (the object legitimately isn't in frame), only a malformed
    response raises."""
    parsed = _load_json_object(raw_text)

    found = parsed.get("found")
    if not isinstance(found, bool):
        raise ValueError(
            f"Missing or non-boolean 'found' in response\nraw_text={raw_text!r}"
        )

    if not found:
        return {
            "found": False,
            "u": None,
            "v": None,
            "reasoning": parsed.get("reasoning", ""),
        }

    u, v = parsed.get("u"), parsed.get("v")
    if (
        not isinstance(u, (int, float))
        or isinstance(u, bool)
        or not isinstance(v, (int, float))
        or isinstance(v, bool)
    ):
        raise ValueError(
            f"found=true but 'u'/'v' are missing/non-numeric\nraw_text={raw_text!r}"
        )

    return {
        "found": True,
        "u": _clamp(float(u), 0.0, 1.0),
        "v": _clamp(float(v), 0.0, 1.0),
        "reasoning": parsed.get("reasoning", ""),
    }


def parse_goal_touching_bottom_response(raw_text: str) -> dict:
    """Parse a build_goal_touching_bottom_prompt response: a plain boolean
    'stop' signal, same isinstance(bool) validation as
    parse_ground_object_response's 'found'."""
    parsed = _load_json_object(raw_text)

    stop = parsed.get("stop")
    if not isinstance(stop, bool):
        raise ValueError(
            f"Missing or non-boolean 'stop' in response\nraw_text={raw_text!r}"
        )
    return {"stop": stop, "reasoning": parsed.get("reasoning", "")}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def parse_drive_action_response(raw_text: str) -> Action:
    parsed = _load_json_object(raw_text)

    for key in ("v_fwd", "v_lat", "yaw_rate"):
        value = parsed.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                f"Missing or non-numeric '{key}' in response\nraw_text={raw_text!r}"
            )

    v_fwd = _clamp(float(parsed["v_fwd"]), 0.0, 1.0)
    v_lat = _clamp(float(parsed["v_lat"]), -1.0, 1.0)
    yaw_rate = _clamp(float(parsed["yaw_rate"]), -1.0, 1.0)

    return Action(v_fwd=v_fwd, v_lat=v_lat, yaw_rate=yaw_rate)


def parse_ghost_mask_belief_response(
    raw_text: str,
    frame_w: int,
    frame_h: int,
    min_radius_px: float,
    max_radius_px: float,
) -> dict:
    """Parse a build_ghost_mask_belief_prompt response: pixel center clamped
    into the frame, both radii clamped into [min_radius_px, max_radius_px] --
    same clamp-not-reject philosophy as parse_drive_action_response, so a
    model response that's directionally right but numerically sloppy still
    produces a usable placement instead of being thrown away."""
    parsed = _load_json_object(raw_text)

    for key in ("u", "v", "radius_u_px", "radius_v_px"):
        value = parsed.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                f"Missing or non-numeric '{key}' in response\nraw_text={raw_text!r}"
            )

    u = _clamp(float(parsed["u"]), 0.0, float(frame_w - 1))
    v = _clamp(float(parsed["v"]), 0.0, float(frame_h - 1))
    radius_u_px = _clamp(float(parsed["radius_u_px"]), min_radius_px, max_radius_px)
    radius_v_px = _clamp(float(parsed["radius_v_px"]), min_radius_px, max_radius_px)

    return {"u": u, "v": v, "radius_u_px": radius_u_px, "radius_v_px": radius_v_px}


if __name__ == "__main__":
    well_formed = '{"v_fwd": 0.5, "v_lat": -0.2, "yaw_rate": 0.1}'
    fenced = """```json
{"goal_index": 2}
```"""
    malformed = '{"v_fwd": 0.5, "v_lat": "left", yaw_rate: 0.1}'

    print("parsed drive action:")
    print(parse_drive_action_response(well_formed))

    print("\n fenced goal selection:")
    print(parse_select_goal_response(fenced))

    print("\n primitive input:")
    try:
        parse_drive_action_response(malformed)
    except ValueError as e:
        print(f"caught ValueError: {e}")

    print("\n parsed direction:")
    print(
        parse_direction_response(
            '{"direction": "turn_left", "reasoning": "goal is left"}'
        )
    )

    print("\n invalid direction:")
    try:
        parse_direction_response('{"direction": "spin", "reasoning": "nope"}')
    except ValueError as e:
        print(f"caught ValueError: {e}")

    print("\n parsed nav command:")
    print(
        parse_nav_command_response(
            '{"directions": ["left"], "goals": ["flag", "home base"]}'
        )
    )

    print("\n parsed split instruction:")
    print(
        parse_split_instruction_response(
            '{"instructions": ["Go straight until you reach the door.", '
            '"Turn left.", "Search for a water bottle."]}'
        )
    )

    print("\n empty split instruction list:")
    try:
        parse_split_instruction_response('{"instructions": []}')
    except ValueError as e:
        print(f"caught ValueError: {e}")

    print("\n non-string entry in split instruction list:")
    try:
        parse_split_instruction_response('{"instructions": ["fine", 42]}')
    except ValueError as e:
        print(f"caught ValueError: {e}")

    print("\n missing 'instructions' key:")
    try:
        parse_split_instruction_response('{"directions": ["left"]}')
    except ValueError as e:
        print(f"caught ValueError: {e}")
