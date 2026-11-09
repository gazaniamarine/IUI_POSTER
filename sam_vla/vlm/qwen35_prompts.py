"""
Builds text prompts sent to Qwen3.5-9B for InternVLA-N1-style point-goal
navigation with instruction history (see nav.vlm_point_navigator). Sibling
of qwen_prompts.py (Qwen2.5-VL-3B's goal-selection/driving prompts), kept
separate since this is a different model/server/port -- see qwen35_config.py.
"""


def build_point_goal_history_prompt(instruction_text: str, frame_idx: int) -> str:
    """No detector upstream -- same "ask for a point, not a bbox" reasoning
    as qwen_prompts.build_ground_object_prompt (a tight bbox is a harder
    regression target, and the caller only ever needs a point to backproject
    through depth anyway, see sam_vla.core.goal_geometry.pixel_to_world),
    applied here to a full navigation instruction instead of a single named
    object. The image is the CURRENT frame only -- prior frames/instructions/
    reasoning are supplied as real chat history by
    qwen35_model_runner.run_qwen35_inference_with_history, not restated in
    this prompt text.

    Two-pixel-slot schema (2026-08-27, replaces a single conflated "point" +
    separate "advance" boolean) adapted from studying Alibaba AMAP CV Lab's
    ABot-N1 (arXiv 2607.10383) -- their slow reasoner emits a distinct
    "affordance pixel" (always given: next safe traversable waypoint) and
    "target pixel" (only given once the actual instruction target becomes
    visible/near), rather than one point standing in for both roles. Our
    old single "u,v" field asked the model to pick a point that meant BOTH
    "where to drive" and, implicitly, "is this the goal" -- exactly the kind
    of conflation that made the advance decision (the diagnosed R2R-CE
    benchmark bottleneck, see nav/benchmark/FINDINGS.md) hard to get right.
    Splitting them mirrors a real, published SOTA design rather than a
    guess: `affordance` still drives every tick (feeds VlmStepResult.
    world_xz unchanged, so RoverController's driving logic needs no
    changes), `target` is only ever populated when the model believes the
    instruction's target is actually visible/near -- and `advance=true`
    without `target.visible=true` is treated as an inconsistent response
    (see nav.vlm_point_navigator.VlmPointNavigator.maybe_step), a free
    coherence check ABot-N1's design makes possible that the old
    single-point schema couldn't express at all.

    "reasoning" is also now the same two-named-block structure ABot-N1's
    CoT trace uses ("Target Localization and Key Context" then "Path
    Planning") instead of one unstructured free-text sentence -- scaffolds
    the same two judgments (what's visible / what to do about it) the
    schema itself now separates, rather than leaving both to run together
    in prose."""
    return (
        "You are the navigation goal-setter for a Mars rover, part of a "
        "two-system driving stack: you look at the rover's current camera "
        "view and past views/decisions you made, and pick points on the "
        "ground for a separate driving system to steer toward -- you never "
        "drive the rover yourself, only choose where it should go next.\n\n"
        f"Current instruction: {instruction_text}\n"
        f"This is frame {frame_idx}. Use your own prior frames/points/"
        "reasoning (given to you as conversation history, if any) to judge "
        "progress on this instruction -- do not treat this call as your "
        "first look at the scene if history is present.\n\n"
        "First decide which of two response shapes applies:\n\n"
        '1. POINT -- the instruction is (or currently requires) driving '
        "toward a spot on the ground -- e.g. \"go straight until you find "
        'the door", "find a chair", or a turn instruction where something '
        "in view already tells you which way to steer. Respond with ONLY:\n"
        '{"action": "point", '
        '"affordance": {"found": <bool>, "u": <float in [0,1]|null>, '
        '"v": <float in [0,1]|null>}, '
        '"target": {"visible": <bool>, "u": <float in [0,1]|null>, '
        '"v": <float in [0,1]|null>}, '
        '"advance": <bool>, '
        '"reasoning": {"target_localization": <str, 1 sentence>, '
        '"path_planning": <str, 1 sentence>}}\n\n'
        '"affordance" is the point on the ground the rover should drive '
        "toward NEXT -- always attempt this one. u/v in normalized image "
        "coordinates (u: 0=left edge, 1=right edge; v: 0=top edge, "
        "1=bottom edge). Base it only on what you can actually see in this "
        "specific frame -- do not estimate a location you cannot see. If "
        "nothing in the current frame tells you where to go next, set "
        'affordance to {"found": false, "u": null, "v": null}.\n\n'
        '"target" is the instruction\'s actual end goal (the door/object/'
        "spot the instruction names), NOT the next step toward it -- only "
        "set visible=true and give u/v once that end goal itself is "
        "actually visible in THIS frame or the rover is right at/in front "
        "of it. Most frames along the way there will have "
        '{"visible": false, "u": null, "v": null} -- that is normal and '
        "expected, not a failure; only mark visible=true once you would "
        "genuinely point a finger at the named target itself on screen.\n\n"
        '2. TURN -- the instruction is a pure in-place rotation with no '
        'forward-driving component (e.g. "turn left", "turn around", '
        '"face right", "turn left 45 degrees") and picking a ground point '
        "would not actually make the rover rotate (this matters most once "
        "the scene ahead is already featureless/unchanging turn over turn "
        "-- keep responding TURN rather than repeating the same point). "
        "Respond with ONLY:\n"
        '{"action": "turn", "direction": "left"|"right", '
        '"degrees": <float in (0, 180]>, '
        '"reasoning": {"target_localization": <str, 1 sentence>, '
        '"path_planning": <str, 1 sentence>}}\n'
        "direction is which way to rotate (left = counter-clockwise, "
        "right = clockwise); degrees is how far, up to a full reversal. "
        "This is a single, complete rotation, executed exactly once -- "
        "there is no follow-up step to refine it, so pick the exact "
        "values that match the instruction's own wording, never a partial "
        'guess: "turn left"/"face left" with no angle given -> '
        '{"direction": "left", "degrees": 90}; "turn right"/"face right" '
        'with no angle given -> {"direction": "right", "degrees": 90}; '
        '"turn slightly left"/"turn left a bit" -> smaller, e.g. 30; "turn '
        'left 45 degrees"/"turn 45 degrees to the left" -> {"direction": '
        '"left", "degrees": 45} (use the instruction\'s own stated angle '
        'whenever it gives one, for either direction); "turn around"/"turn '
        'back"/"face the other way"/"180 degrees" -> degrees=180 with '
        "either direction (both reach the same heading, so pick either) "
        "-- never substitute a smaller angle for this, even as a first "
        "step, since nothing will follow up on it.\n"
        "Obstacles visible in the frame are NOT your concern for TURN -- a "
        "separate safety system independently keeps the rover clear of "
        "collisions while it physically rotates, regardless of which "
        "direction/degrees you pick. Decide direction/degrees from the "
        "instruction's wording and the rover's current facing only; do "
        "not reason about what is or isn't in the way, and do not let an "
        "obstacle change your answer or fall back to POINT.\n"
        "(no \"advance\" field for TURN -- the physical turn's own "
        "completion is what advances the instruction, not your judgment; "
        "you will be re-queried automatically once it finishes turning.)\n\n"
        '"advance" (POINT only) is true only if the current instruction is '
        "now satisfied and the rover should move on to its next instruction "
        "-- otherwise false. advance=true only makes sense once "
        '"target.visible" is also true for THIS instruction\'s own named '
        "target -- if the target isn't visible/reached yet, advance must "
        "be false. The affordance point you give this call should still be "
        "the best point for finishing the current instruction even when "
        "advance is true.\n"
        "For an instruction phrased as arriving somewhere -- \"wait at X\", "
        "\"stop in front of X\", \"stop at X\" -- treat it as equivalent to "
        "\"reach X\": X becomes the target, and target.visible=true / "
        "advance=true as soon as the rover is at/right in front of X, even "
        "though nothing will visibly happen once it's there. Do not hold "
        "advance=false waiting to observe the rover \"waiting\" or any "
        "other confirmation -- arrival IS the satisfaction condition for "
        "these instructions.\n\n"
        "For an instruction phrased as passing something -- \"walk/go past "
        "X\", \"go through X\", \"around X\" -- do NOT mark target.visible "
        "or advance=true just because you intend to go past it or can see "
        "it ahead; X must already be beside/behind the rover, or the rover "
        "must already be past/through the doorway, in the CURRENT frame. If "
        "X is still ahead of the rover (even close ahead), target.visible "
        "stays false, advance stays false, and your affordance point should "
        "continue driving toward/past it.\n\n"
        '"reasoning.target_localization" states what, if anything relevant '
        'to the instruction, is visible right now (one sentence). '
        '"reasoning.path_planning" states your decision -- where you\'re '
        "pointing and why (one sentence). Do not deliberate multiple "
        "candidate answers or restate the rules above -- decide once and "
        "report the result."
    )


def build_reasoning_goal_prompt(instruction_text: str, frame_idx: int) -> str:
    """Compact reasoning-first variant for smaller VLMs (e.g. Qwen3-VL-4B):
    the model writes its short scene analysis BEFORE committing to pixels, so
    the point is conditioned on the reasoning instead of the reasoning
    rationalizing a point. Single instruction only -- no sequencing or
    staging. Same JSON schema as build_point_goal_history_prompt, so
    parse_point_goal_history_response handles it unchanged."""
    return (
        "You are the navigation goal-setter for a Mars rover. Look at the "
        "current camera view (and your earlier views/decisions in the "
        "conversation, if any) and choose where the rover should drive next "
        "to reach the goal. A separate system does the driving and obstacle "
        "avoidance.\n\n"
        f"Goal: {instruction_text}\n"
        f"Frame: {frame_idx}\n\n"
        "First decide the action type from the goal's wording.\n\n"
        "TURN -- a pure in-place rotation: \"turn left/right\", \"face "
        "left/right\", \"turn around\", \"go/walk backward\". "
        "The rover cannot reverse, so backward = turn around. Respond with "
        "ONLY:\n"
        '{"reasoning": {"target_localization": <str>, "path_planning": <str>}, '
        '"action": "turn", "direction": "left"|"right", "degrees": <float in (0,180]>}\n'
        'Degrees: no angle given -> 90; "slightly"/"a bit" -> 30; a stated '
        'angle -> use it exactly; "turn around"/"backward"/"back"/"180" -> '
        "180 (either direction). One complete rotation, executed once. "
        "Ignore obstacles for TURN.\n\n"
        "FORWARD -- a stated distance straight ahead: \"go forward 3 "
        "meters\", \"walk 5 m\". Respond with ONLY:\n"
        '{"reasoning": {"target_localization": <str>, "path_planning": <str>}, '
        '"action": "forward", "distance_m": <float in (0,50]>}\n'
        "Use the stated distance exactly (convert feet etc. to meters).\n\n"
        "STOP -- \"stop\", \"wait\", \"hold\", \"stay\": respond with ONLY "
        '{"reasoning": {...}, "action": "stop"}\n\n'
        "HOME -- the rover's start/base location, whether or not it is "
        "visible: \"go home\", \"return to base\", \"go back to start\", "
        '"come back", "return". NOT a turn and NOT forward (FORWARD needs '
        "a stated distance in meters; if none is stated it is never "
        "FORWARD). Respond with ONLY "
        '{"reasoning": {...}, "action": "home"}\n\n'
        "POINT -- anything else that means driving: \"go straight\", \"walk "
        "forward\" (no distance), \"go to X\", \"find X\". For "
        "straight/forward/walk with no named target and no distance, pick "
        "traversable ground straight ahead (u near 0.5), as far as is safe, "
        "and keep advance=false. Otherwise drive toward the named target.\n\n"
        "Think first, then answer. For POINT respond with ONLY this JSON "
        "object, reasoning first:\n"
        '{"reasoning": {"target_localization": <str: is the goal visible '
        "now, and where in the image?>, "
        '"path_planning": <str: which traversable ground to take toward it, '
        "avoiding rocks/slopes, and why>}, "
        '"action": "point", '
        '"affordance": {"found": <bool>, "u": <float 0-1|null>, "v": <float 0-1|null>}, '
        '"target": {"visible": <bool>, "u": <float 0-1|null>, "v": <float 0-1|null>}, '
        '"advance": <bool>}\n\n'
        "u/v are normalized image coordinates (u: 0=left, 1=right; v: 0=top, "
        "1=bottom).\n"
        '- "affordance": the traversable ground point to drive toward NEXT, '
        "visible in this frame. If nothing tells you where to go, "
        'found=false with null u/v.\n'
        '- "target": the goal itself. visible=true only if the goal is '
        "actually seen in THIS frame; give its u/v.\n"
        '- "advance": true only when the rover has reached the goal '
        "(target.visible=true and it is right in front of the rover), else "
        "false.\n"
        "Decide once; do not list alternatives."
    )


if __name__ == "__main__":
    print(build_point_goal_history_prompt("Drive to the large rock cluster ahead.", 0))
