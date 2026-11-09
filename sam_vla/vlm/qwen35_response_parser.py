from sam_vla.vlm.qwen_response_parser import _clamp, _load_json_object


_TURN_DIRECTIONS = ("left", "right")
_DEFAULT_TURN_DEGREES = 90.0
_MAX_TURN_DEGREES = 180.0
_MAX_FORWARD_M = 50.0


def _parse_reasoning(parsed: dict) -> str:
    """Flattens the two-block {"target_localization": ..., "path_planning":
    ...} reasoning structure (see build_point_goal_history_prompt's
    docstring) into a single string for logging/history round-tripping --
    downstream code (event logs, VlmPointNavigator's history buffer) only
    ever wanted a human-readable string, never the block structure itself.
    A bare string is still accepted for backward compatibility with
    responses from before this schema existed."""
    reasoning = parsed.get("reasoning", "")
    if isinstance(reasoning, dict):
        loc = reasoning.get("target_localization", "")
        plan = reasoning.get("path_planning", "")
        return f"{loc} {plan}".strip()
    return reasoning if isinstance(reasoning, str) else ""


def parse_point_goal_history_response(raw_text: str) -> dict:
    """Parse a build_point_goal_history_prompt response: one of two shapes,
    both distinguished by 'action'.

    'action': 'turn' -- an in-place-rotation response, {'direction': one of
    _TURN_DIRECTIONS, 'degrees': a magnitude in (0, _MAX_TURN_DEGREES]}. No
    'advance'/'affordance'/'target' -- see the prompt's own docstring for
    why (physical turn completion advances the instruction, not the
    model's own judgment). 'direction': 'back' (this schema's shape before
    degrees existed) is accepted as a legacy alias for {'direction': 'left',
    'degrees': 180}.

    'action': 'point' (also the implicit default when 'action' is absent)
    -- two-pixel-slot schema (2026-08-27, adapted from ABot-N1, arXiv
    2607.10383 -- see the prompt's own docstring): 'affordance'
    {'found', 'u', 'v'} is the always-attempted next-driving-point, mapped
    onto this function's returned 'found'/'u'/'v' keys (unchanged names,
    so nav.vlm_point_navigator.VlmPointNavigator's existing driving logic
    needs no changes -- only the SOURCE of that point moved). 'target'
    {'visible', 'u', 'v'} is the instruction's actual end goal, only
    populated once genuinely visible/reached -- returned as
    'target_visible'/'target_u'/'target_v'. Coherence check: 'advance'
    can only be True when 'target_visible' is also True (ABot-N1's own
    design makes target-visibility the necessary condition for arrival;
    an advance=true with no visible target is an inconsistent response,
    silently downgraded to False here rather than trusted at face value
    -- this is the free correctness check the old single-point schema
    couldn't express at all)."""
    parsed = _load_json_object(raw_text)

    action = parsed.get("action", "point")
    if action == "turn":
        direction = parsed.get("direction")
        degrees = parsed.get("degrees")
        if direction == "back":
            direction = "left"
            if degrees is None:
                degrees = _MAX_TURN_DEGREES
        if direction not in _TURN_DIRECTIONS:
            raise ValueError(
                f"action='turn' but 'direction' is not one of {_TURN_DIRECTIONS}"
                f" (or legacy 'back')\nraw_text={raw_text!r}"
            )
        if degrees is None:
            degrees = _DEFAULT_TURN_DEGREES
        if (
            not isinstance(degrees, (int, float))
            or isinstance(degrees, bool)
            or not (0.0 < float(degrees) <= _MAX_TURN_DEGREES)
        ):
            raise ValueError(
                f"action='turn' but 'degrees' is not a number in "
                f"(0, {_MAX_TURN_DEGREES}]\nraw_text={raw_text!r}"
            )
        return {
            "action": "turn",
            "direction": direction,
            "degrees": float(degrees),
            "reasoning": _parse_reasoning(parsed),
        }
    if action == "forward":
        distance = parsed.get("distance_m")
        if (
            not isinstance(distance, (int, float))
            or isinstance(distance, bool)
            or not (0.0 < float(distance) <= _MAX_FORWARD_M)
        ):
            raise ValueError(
                f"action='forward' but 'distance_m' is not a number in "
                f"(0, {_MAX_FORWARD_M}]\nraw_text={raw_text!r}"
            )
        return {
            "action": "forward",
            "distance_m": float(distance),
            "reasoning": _parse_reasoning(parsed),
        }
    if action in ("stop", "home"):
        return {"action": action, "reasoning": _parse_reasoning(parsed)}
    if action != "point":
        raise ValueError(
            f"Unknown 'action' {action!r} in response\nraw_text={raw_text!r}"
        )

    affordance = parsed.get("affordance")
    if not isinstance(affordance, dict):
        raise ValueError(
            f"Missing or non-object 'affordance' in response\nraw_text={raw_text!r}"
        )
    found = affordance.get("found")
    if not isinstance(found, bool):
        raise ValueError(
            f"Missing or non-boolean 'affordance.found' in response\nraw_text={raw_text!r}"
        )

    target = parsed.get("target")
    if not isinstance(target, dict):
        raise ValueError(
            f"Missing or non-object 'target' in response\nraw_text={raw_text!r}"
        )
    target_visible = target.get("visible")
    if not isinstance(target_visible, bool):
        raise ValueError(
            f"Missing or non-boolean 'target.visible' in response\nraw_text={raw_text!r}"
        )

    advance = parsed.get("advance")
    if not isinstance(advance, bool):
        raise ValueError(
            f"Missing or non-boolean 'advance' in response\nraw_text={raw_text!r}"
        )
    # Coherence check -- see this function's docstring.
    advance = advance and target_visible

    target_u = target_v = None
    if target_visible:
        tu, tv = target.get("u"), target.get("v")
        if (
            isinstance(tu, (int, float)) and not isinstance(tu, bool)
            and isinstance(tv, (int, float)) and not isinstance(tv, bool)
        ):
            target_u = _clamp(float(tu), 0.0, 1.0)
            target_v = _clamp(float(tv), 0.0, 1.0)
        # target.visible=true with missing/malformed u,v is tolerated (not
        # raised) -- the coherence-checked 'advance' above is what actually
        # matters downstream; a target believed visible but not pinpointed
        # still legitimately means "we've arrived."

    reasoning = _parse_reasoning(parsed)

    if not found:
        return {
            "action": "point",
            "found": False,
            "u": None,
            "v": None,
            "target_visible": target_visible,
            "target_u": target_u,
            "target_v": target_v,
            "advance": advance,
            "reasoning": reasoning,
        }

    u, v = affordance.get("u"), affordance.get("v")
    if (
        not isinstance(u, (int, float))
        or isinstance(u, bool)
        or not isinstance(v, (int, float))
        or isinstance(v, bool)
    ):
        raise ValueError(
            f"affordance.found=true but 'u'/'v' are missing/non-numeric\nraw_text={raw_text!r}"
        )

    return {
        "action": "point",
        "found": True,
        "u": _clamp(float(u), 0.0, 1.0),
        "v": _clamp(float(v), 0.0, 1.0),
        "target_visible": target_visible,
        "target_u": target_u,
        "target_v": target_v,
        "advance": advance,
        "reasoning": reasoning,
    }


if __name__ == "__main__":
    print(
        parse_point_goal_history_response(
            '{"affordance": {"found": true, "u": 0.5, "v": 0.7}, '
            '"target": {"visible": false, "u": null, "v": null}, '
            '"advance": false, '
            '"reasoning": {"target_localization": "rock cluster ahead", '
            '"path_planning": "heading toward it"}}'
        )
    )
    print(
        parse_point_goal_history_response(
            '{"affordance": {"found": true, "u": 0.5, "v": 0.9}, '
            '"target": {"visible": true, "u": 0.5, "v": 0.85}, '
            '"advance": true, '
            '"reasoning": {"target_localization": "door directly ahead", '
            '"path_planning": "arrived at threshold"}}'
        )
    )
    print(
        # inconsistent response -- advance=true claimed with no visible
        # target -- should be silently downgraded to advance=False.
        parse_point_goal_history_response(
            '{"affordance": {"found": true, "u": 0.5, "v": 0.7}, '
            '"target": {"visible": false, "u": null, "v": null}, '
            '"advance": true, '
            '"reasoning": {"target_localization": "not yet visible", '
            '"path_planning": "continuing"}}'
        )
    )
