"""
Builds text prompts sent to Qwen2.5-VL for goal selection and driving.
"""


def build_ghost_mask_belief_prompt(
    bearing_deg: float,
    distance_m: float,
    bearing_uncertainty_deg: float,
    distance_uncertainty_m: float,
    frame_w: int,
    frame_h: int,
    min_radius_px: float,
    max_radius_px: float,
) -> str:
    """GHOST_MASK placement prompt for a full Gaussian belief (mean +
    covariance in robot [forward, left] frame), already reduced by
    sam_vla.core.ghost_mask.belief_to_bearing_range_uncertainty to bearing,
    distance, and two independent uncertainty scalars -- bearing_uncertainty
    (how sure of the direction) and distance_uncertainty (how sure of the
    range). The goal is frequently entirely out of frame (behind or far to
    the side of the camera), so this does not attempt a geometric pixel
    projection -- the model must nudge the mask toward the correct screen
    edge from the text description alone, the same way
    vl_direction/prompts/ghost_mask_prompt.py's scalar-uncertainty version
    does, generalized here to an anisotropic ellipse: bearing_uncertainty
    sizes the horizontal (u) radius, distance_uncertainty sizes the vertical
    (v) radius, so a confidently-bearinged-but-distance-vague belief comes
    out tall and narrow rather than a uniform blob.
    """
    cx = frame_w / 2.0
    side = "left" if bearing_deg > 0 else "right" if bearing_deg < 0 else "ahead"
    behind = abs(bearing_deg) > 90.0

    behind_note = (
        f" The bearing magnitude ({bearing_deg:+.0f} degrees) means the goal is "
        f"actually BEHIND the rover, not just off to one side -- pin the ellipse "
        f"at the extreme {side} edge of the frame, as close to the edge as "
        f"possible, rather than partway across.\n"
        if behind
        else ""
    )

    system_framing = (
        "You are a spatial-reasoning assistant for a Mars rover. You never explain, "
        "you only answer with JSON."
    )
    context_block = (
        f"The rover's goal is currently out of view. Belief estimate (from the "
        f"rover's tracked position belief, not direct observation): bearing "
        f"{bearing_deg:+.0f} degrees from straight ahead ({side}), distance "
        f"{distance_m:.1f}m.\n"
        f"Direction confidence: bearing uncertainty is +/-{bearing_uncertainty_deg:.0f} "
        f"degrees (small = confident in the {side} direction, large = the goal "
        f"could be anywhere across a wide arc).\n"
        f"Distance confidence: distance uncertainty is +/-{distance_uncertainty_m:.1f}m "
        f"(small = confident how far away, large = could be much nearer or farther).\n"
        f"Frame size is {frame_w}x{frame_h} pixels; horizontal center is u={cx:.0f}.\n"
        f"{behind_note}"
        f"Place an ellipse marking where you believe the goal most likely is. The "
        f"bearing says {side} -- unless it is within a few degrees of straight "
        f"ahead, the ellipse center must sit clearly off-center toward the {side} "
        f"edge, NOT near u={cx:.0f}. Scale how far off-center by the bearing "
        f"magnitude: a bearing near +/-90 degrees or beyond should land at or "
        f"past the frame's {side} edge, a small bearing (e.g. +/-10) only "
        f"slightly off center. u={cx:.0f} means 'straight ahead', which would "
        f"contradict the goal being out of view.\n"
        f"Grow radius_u_px (horizontal) with bearing uncertainty and radius_v_px "
        f"(vertical) with distance uncertainty, independently -- they do not have "
        f"to match. Both stay within [{min_radius_px:.0f}, {max_radius_px:.0f}]px. "
        f"Keep the ellipse's center within the frame even when it sits at the "
        f"extreme edge."
    )
    json_schema_line = (
        'Respond with exactly one JSON object: {"u": <int>, "v": <int>, '
        '"radius_u_px": <int>, "radius_v_px": <int>} -- u is the horizontal pixel '
        "coordinate (0=left edge, increasing right), v is the vertical pixel "
        "coordinate (0=top edge, increasing down), radius_u_px/radius_v_px are the "
        "ellipse's horizontal/vertical radii in pixels. No other text."
    )

    if bearing_deg > 5.0:
        example_u = frame_w * 0.15
    elif bearing_deg < -5.0:
        example_u = frame_w * 0.85
    else:
        example_u = frame_w * 0.5
    example_v = frame_h * 0.45
    example_ru = min(max(min_radius_px, 50.0), max_radius_px)
    example_rv = min(max(min_radius_px, 90.0), max_radius_px)

    return (
        f"{system_framing}\n\n"
        f"{context_block}\n\n"
        f"{json_schema_line}\n"
        f"Example (format only -- your own numbers must reflect the actual bearing "
        f'and uncertainties above, not copy these): {{"u": {int(example_u)}, '
        f'"v": {int(example_v)}, "radius_u_px": {int(example_ru)}, '
        f'"radius_v_px": {int(example_rv)}}}'
    )


def build_select_goal_prompt(detections: list[dict]) -> str:
    lines = []
    for i, det in enumerate(detections):
        lines.append(
            f'{i}: class="{det["class_name"]}", bbox_norm={det["bbox_norm"]}, '
            f'confidence={det["confidence"]:.2f}'
        )
    detections_block = "\n".join(lines)

    return (
        "You are the vision system for a Mars rover. The image shows the "
        "rover's current camera view. Below is a list of detected rock "
        "instances, each with a normalized bounding box [x_min, y_min, x_max, "
        "y_max] in [0, 1] image coordinates.\n\n"
        f"{detections_block}\n\n"
        "Pick exactly ONE detection from the list above to serve as the "
        "rover's navigation goal (the rock the rover should drive to)."
        "Respond with ONLY a JSON object in this exact format, no other text:\n"
        '{"goal_index": <int, index into the list above>, '
        '"reasoning": <str, brief explanation>}'
    )


def build_goal_vocabulary_prompt() -> str:
    """For the multi-goal path: instead of picking one detection as the goal,
    Qwen describes a small open-vocabulary set of goal-worthy object terms
    visible in the scene (fed to SAM3's per-term text prompting and CLIP's
    text-embedding bank) plus one instruction sentence covering all of them."""
    return (
        "You are the vision system for a Mars rover. The image shows the "
        "rover's current camera view. The rover will visit multiple "
        "goal-worthy objects in this scene over the course of the episode, "
        "not just one.\n\n"
        "List a small set of short, visually distinct object terms (2-4 "
        "terms) describing the categories of rocks or other objects in this "
        'scene that would make good navigation targets, e.g. "small rock" '
        'vs "big rock" if there is a clear size distinction, or terms '
        "based on shape/color if that better separates the objects you see. "
        "Also give one instruction sentence describing the rover's overall "
        "task across all of them.\n\n"
        "Respond with ONLY a JSON object in this exact format, no other "
        "text:\n"
        '{"terms": [<str>, ...], "instruction_text": <str>, '
        '"reasoning": <str, brief explanation>}'
    )


def build_drive_action_prompt(instruction_text: str, frame_idx: int) -> str:
    return (
        "You are the driving policy for a Mars rover. The image is the "
        f"rover's current camera frame (frame {frame_idx}).\n\n"
        f"Navigation instruction: {instruction_text}\n\n"
        "Output the rover's next action as ONLY a JSON object, no other "
        "text, in this exact format:\n"
        '{"v_fwd": <float in [0, 1]>, "v_lat": <float in [-1, 1]>, '
        '"yaw_rate": <float in [-1, 1]>, "reasoning": <str>}\n\n'
        "v_fwd is normalized forward speed (0 = stop, 1 = full speed), "
        "v_lat is normalized lateral speed (negative = left, positive = "
        "right), and yaw_rate is normalized turn rate (negative = turn "
        "left, positive = turn right). Steer away from any obstacles "
        "visible in the current frame while making progress toward the "
        "goal described in the instruction."
    )


def build_parse_nav_command_prompt(command_text: str) -> str:
    """nav/gui.py's free-text Command panel: segments a natural-language nav
    command into two ordered lists -- "directions" (closed-vocabulary
    movement instructions) and "goals" (short phrases naming specific
    target objects) -- so a multi-part command like "go left, find a flag,
    then come back to home base" can eventually drive goal-sequencing one
    step at a time instead of being handed to goal resolution as one opaque
    string. The image is scene context only (nothing here is graded against
    it yet)."""
    return (
        "You are a command parser for a Mars rover. The image is the "
        "rover's current camera view, given for context only.\n\n"
        f'Rover command: "{command_text}"\n\n'
        "Break this command into two ordered lists:\n"
        '  "directions": plain movement instructions, each normalized to '
        'exactly one of "turn_left", "turn_right", "turn_back", "go_left", '
        '"go_right", "go_front". Two different verbs matter here, and they '
        "mean different things -- do not blur them:\n"
        '    - "turn_left" / "turn_right" / "turn_back": the command says to '
        'turn, face, or rotate in place, with NO forward movement implied '
        '(e.g. "turn left" -> "turn_left", "turn around" -> "turn_back", '
        '"face the other way" -> "turn_back", "face right" -> "turn_right").\n'
        '    - "go_left" / "go_right" / "go_front": the command says to '
        "actually move/head/drive in a direction (e.g. \"go left\" -> "
        '"go_left", "move right" -> "go_right", "head left" -> "go_left", '
        '"keep going straight" -> "go_front", "drive forward" -> '
        '"go_front"). These mean turn to face that way (already true for '
        "go_front) AND THEN keep driving forward -- never downgrade one of "
        'these to the turn_* form just because it looks similar.\n'
        '  Only include an entry here for movement with no named target -- '
        "a command to navigate toward a specific object (even \"go back to "
        'home base") belongs in "goals", not here. This includes bare '
        '"come back" / "go back" / "return" with no object named after it '
        "-- these mean drive all the way back to the start point, not spin "
        'in place, so they ALWAYS belong in "goals" as "home base", never '
        'in "directions" as "turn_back". Only use "turn_back" when the '
        "command explicitly says to turn/face backward in place with no "
        'sense of travelling anywhere (e.g. "turn around", "face the other '
        'way").\n'
        '  "goals": short phrases naming a specific target object to find '
        'or navigate to (e.g. "white flag", "flag on the right", "home '
        'base"), kept as close to the original wording as possible while '
        "staying as short as possible -- still distinguishing it from any "
        'other target in the same command (e.g. keep "on the right" only '
        "if there is more than one flag to tell apart).\n\n"
        "Preserve the order implied by the command within each list "
        "separately. Base both lists ONLY on the words in the command -- "
        'the image is context for later steps, not something to parse '
        'directions out of. If the command names no explicit movement word '
        '(e.g. "go to the flag", "find the rover", "head to the flag now"), '
        '"directions" MUST be an empty list -- never invent a direction '
        "to fill it, and never let a target's apparent position in the "
        "image or its similarity to a target in one of the examples below "
        "imply a direction that was not actually stated.\n\n"
        "Examples:\n"
        '  "go left find a flag and come back to home base" -> '
        '{"directions": ["go_left"], "goals": ["flag", "home base"]}\n'
        '  "go left find a flag and come back" -> '
        '{"directions": ["go_left"], "goals": ["flag", "home base"]}\n'
        '  "turn left then find a flag" -> '
        '{"directions": ["turn_left"], "goals": ["flag"]}\n'
        '  "turn around" -> '
        '{"directions": ["turn_back"], "goals": []}\n'
        '  "keep going straight" -> '
        '{"directions": ["go_front"], "goals": []}\n'
        '  "find the white flag and then go to the flag on the right" -> '
        '{"directions": [], "goals": ["white flag", "flag on the right"]}\n'
        '  "turn right then go straight" -> '
        '{"directions": ["turn_right", "go_front"], "goals": []}\n'
        '  "go to the flag" -> '
        '{"directions": [], "goals": ["flag"]}\n'
        '  "head to the flag now" -> '
        '{"directions": [], "goals": ["flag"]}\n'
        '  "drive to the rover" -> '
        '{"directions": [], "goals": ["rover"]}\n\n'
        "Respond with ONLY a JSON object in this exact format, no other "
        'text:\n{"directions": [<str>, ...], "goals": [<str>, ...]}'
    )


def build_split_instruction_prompt(command_text: str) -> str:
    """Splits one free-text, possibly-compound navigation command into an
    ordered list of self-contained atomic instructions -- feeds
    nav.vlm_point_navigator.VlmPointNavigator's instruction queue (see
    RoverController.start_vlm_navigation_from_command), which drives through
    the list one at a time via qwen35_prompts.build_point_goal_history_prompt.

    Unlike build_parse_nav_command_prompt (which sorts words into two
    closed-vocabulary "directions"/"goals" lists for nav.mission.Mission),
    this keeps each instruction as its own free-text sentence, condition and
    all -- a stopping condition like "until you reach the door" has to
    survive the split intact, since VlmPointNavigator's per-instruction
    history/advance logic is exactly what judges that condition later, one
    instruction at a time, with no memory of the original command or the
    instructions around it (see qwen35_prompts.build_point_goal_history_prompt's
    "advance" field). The image is scene context only, same convention as
    build_parse_nav_command_prompt."""
    return (
        "You are a command parser for a Mars rover. The image is the "
        "rover's current camera view, given for context only.\n\n"
        f'Rover command: "{command_text}"\n\n'
        "Break this command into an ordered list of self-contained "
        "navigation instructions, one per distinct step the rover must "
        "complete before moving to the next. Each instruction must stand "
        "on its own -- keep any stopping condition or target it names "
        '(e.g. "until you reach the door", "search for a water bottle") '
        "inside that instruction's own text, since whoever executes it "
        "later will see only that one instruction, not the original "
        "command or the other instructions around it. Fix any typos from "
        "the original command when you restate it. Do not merge unrelated "
        "steps into one instruction, and do not split a single step into "
        "two.\n\n"
        "Examples:\n"
        '  "go straight untill to reach the door and turn left and serach '
        'for water bottle" -> {"instructions": ["Go straight until you '
        'reach the door.", "Turn left.", "Search for a water bottle."]}\n'
        '  "drive to the large rock then come back to home base" -> '
        '{"instructions": ["Drive to the large rock.", "Return to home '
        'base."]}\n'
        '  "find the flag" -> {"instructions": ["Find the flag."]}\n\n'
        "Respond with ONLY a JSON object in this exact format, no other "
        'text:\n{"instructions": [<str>, ...]}'
    )


def build_ground_object_prompt(target_text: str) -> str:
    """Open-vocabulary grounding: unlike build_select_goal_prompt (which
    picks among SAM2's already-detected rock instances), this asks Qwen to
    localize a named object directly from the raw image with no detector
    upstream -- for object classes SAM2/SAM2-LoRA wasn't trained on (e.g.
    the placed flag markers, the home-base ghost cuboid, see
    sam_vla/env/flag_placement.py, sam_vla/env/home_base.py), not for rocks,
    which stay on the SAM2 path.

    Asks for a single point rather than a bbox: a tight bbox is a harder
    regression target for a VLM to get right than "where roughly is it",
    and the caller only ever needs a point to backproject/draw a ghost mask
    at anyway (see qwen_grounding_resolver, nav/rover_controller.py's
    ground-target ghost-mask overlay)."""
    return (
        "You are the vision system for a Mars rover. The image shows the "
        f'rover\'s current camera view. Look for the "{target_text}" in '
        "this image.\n\n"
        "If it is visible, respond with ONLY a JSON object in this exact "
        "format, no other text:\n"
        '{"found": true, "u": <float in [0, 1]>, "v": <float in [0, 1]>, '
        '"reasoning": <str, brief explanation>}\n'
        f'u/v is the approximate point at the center of the "{target_text}" '
        "in normalized image coordinates (u: 0=left edge, 1=right edge; "
        "v: 0=top edge, 1=bottom edge). Base u/v only on where you can "
        "actually see it in this specific image -- do not estimate a "
        "location for it if you cannot see it.\n\n"
        f'If the "{target_text}" is not visible anywhere in the image, '
        'respond with ONLY {"found": false, "u": null, "v": null, '
        '"reasoning": <str, brief explanation>}'
    )


def build_direction_prompt(instruction_text: str, frame_idx: int) -> str:
    """Same intent as build_drive_action_prompt, but constrains the model to a
    single discrete steering choice instead of a free continuous action. The
    goal region is overlaid in green and obstacle regions in red directly on
    the image (see perception.semantic_overlay), so the instruction leans on
    those overlays rather than the raw scene."""
    return (
        "You are the driving policy for a Mars rover. The image is the "
        f"rover's current camera frame (frame {frame_idx}). The navigation "
        "goal is highlighted with a GREEN overlay and known obstacles are "
        "highlighted with a RED overlay.\n\n"
        f"Navigation instruction: {instruction_text}\n\n"
        "Choose exactly ONE discrete direction for the rover's next move:\n"
        '  "forward"    - drive straight ahead\n'
        '  "turn_left"  - steer left\n'
        '  "turn_right" - steer right\n\n'
        "Pick whichever direction makes the most progress toward the green "
        "goal region while steering clear of any red obstacle regions. Do "
        "not output speeds or turn rates, only the discrete direction.\n\n"
        "Respond with ONLY a JSON object, no other text, in this exact "
        "format:\n"
        '{"direction": <"forward" | "turn_left" | "turn_right">, '
        '"reasoning": <str>}'
    )


def build_goal_touching_bottom_prompt(frame_idx: int) -> str:
    """Reached-goal check for nav/mission.py's sub-goal stepper: the current
    goal region is overlaid in green (see perception.semantic_overlay), same
    overlay convention as build_direction_prompt, and the model is asked
    purely whether that green region touches the bottom edge of the frame --
    the rover-relative signal that the goal is now directly in front of/
    underneath the rover, not just visible somewhere in the shot. Additive
    to nav/rover_controller.py's existing ground-truth belief-distance
    reached check, not a replacement for it."""
    return (
        "You are a reached-goal detector for a Mars rover. The image is the "
        f"rover's current camera frame (frame {frame_idx}). The navigation "
        "goal is highlighted with a GREEN overlay.\n\n"
        "Look at whether the GREEN overlay region touches the bottom edge "
        "of the image -- that happens once the rover has driven close "
        "enough that the goal is directly in front of or underneath it, not "
        "merely visible somewhere in the frame.\n\n"
        "Respond with ONLY a JSON object, no other text, in this exact "
        "format:\n"
        '{"stop": <bool, true if the green overlay touches the bottom edge '
        'of the frame>, "reasoning": <str>}'
    )


if __name__ == "__main__":
    dummy_detections = [
        {
            "class_name": "rock",
            "bbox_norm": [0.12, 0.30, 0.28, 0.55],
            "confidence": 0.91,
        },
        {
            "class_name": "rock",
            "bbox_norm": [0.60, 0.40, 0.80, 0.70],
            "confidence": 0.77,
        },
        {
            "class_name": "obstacle",
            "bbox_norm": [0.40, 0.10, 0.55, 0.35],
            "confidence": 0.65,
        },
    ]

    print("=== build_select_goal_prompt ===")
    print(build_select_goal_prompt(dummy_detections))
    print()
    print("=== build_drive_action_prompt ===")
    print(build_drive_action_prompt("Drive toward the large rock cluster ahead.", 42))
    print()
    print("=== build_direction_prompt ===")
    print(build_direction_prompt("Drive toward the large rock cluster ahead.", 42))
    print()
    print("=== build_parse_nav_command_prompt ===")
    print(
        build_parse_nav_command_prompt("go left find a flag and come back to home base")
    )
