"""
Prompt template for GHOST_MASK mode: the model is handed the belief state
(bearing/distance to a currently-unseen goal, plus two independent
uncertainty scalars -- how sure of the direction vs. how sure of the
distance) as text, plus the current frame, and decides where on the frame a
translucent "ghost mask" ellipse should be drawn to mark the goal's believed
location -- placement is the model's spatial-reasoning call, not caller-side
trigonometry. The two uncertainty axes let a caller with a full Gaussian
belief (mean + covariance in robot frame) hand over an anisotropic spread
instead of collapsing it to one number; a caller with only a scalar
uncertainty passes it for both and gets the old circle back.

A live smoke test against Qwen2.5-VL-3B-Instruct (see debug_ghost_mask.py,
16 real cases spanning every bearing bucket) showed the model's placement
essentially ignoring the stated bearing magnitude -- a 12 degree bearing and
a 147 degree bearing produced the *same* pixel offset, and two cases landed
exactly on the disallowed center pixel despite an explicit "not near
u=<center>" instruction. Small VLMs are known to be weak at free-form
numeric interpolation from prose (see _worked_example's original docstring
on few-shot dominance); this version leans into that finding rather than
fighting it: build_ghost_mask_prompt now computes the correct placement
itself (same bucket logic a caller-side deterministic path would use) and
hands it to the model as an explicit "suggested" anchor + a short bucket
table, framing the model's job as "retrieve/lightly-adjust" rather than
"compute from scratch". The model is also allowed one short sentence of
reasoning before the JSON (previously forbidden by "you never explain") --
letting a small model restate the bucket it's using measurably improves
numeric compliance versus demanding silent JSON straight away. The parser
(_JSON_OBJECT_RE) already searches for a JSON object anywhere in the
response, so prose before it is harmless as long as the model doesn't stuff
literal braces into its sentence.

Output is strict JSON (u, v, radius_u_px, radius_v_px in pixels) as the
final line, not a closed-vocabulary word.
"""

from vl_direction.schemas import GhostMaskContext

_SYSTEM_FRAMING = (
    "You are a spatial-reasoning assistant for a Mars rover. You may reason "
    "very briefly -- at most one short sentence -- before answering, but the "
    "LAST thing in your response must be exactly one JSON object and nothing "
    "after it."
)

_JSON_SCHEMA_LINE = (
    'The final line of your response must be exactly one JSON object: '
    '{"u": <int>, "v": <int>, "radius_u_px": <int>, "radius_v_px": <int>} -- '
    "u is the horizontal pixel coordinate (0=left edge, increasing right), v "
    "is the vertical pixel coordinate (0=top edge, increasing down), and "
    "radius_u_px/radius_v_px are the ellipse's horizontal/vertical radii in "
    "pixels."
)

# Bearing magnitude -> fraction of the distance from frame-center to the
# relevant edge that the ellipse center should sit at. Deliberately coarse
# (7 buckets, not a continuous formula spelled out in prose) because a small
# VLM reliably picks the right row out of a short table but does not
# reliably interpolate a continuous formula -- see module docstring.
_BEARING_BUCKETS = (
    (10.0, 0.00),   # within ~10deg of dead ahead -> no offset, legitimately centered
    (25.0, 0.35),
    (45.0, 0.55),
    (70.0, 0.75),
    (90.0, 0.90),
    (135.0, 0.97),
    (180.0, 1.00),  # directly behind -> pinned at the extreme edge
)

# Uncertainty (degrees / meters) at which the suggested radius saturates to
# max_radius_px -- heuristic reference points, not measured from real belief
# data (this module has no belief tracker of its own, see schemas.py).
_REFERENCE_BEARING_UNCERTAINTY_DEG = 45.0
_REFERENCE_DISTANCE_UNCERTAINTY_M = 3.0

# Bearing magnitude beyond which the goal is close enough to directly-behind
# that "left of center" / "right of center" stops being the useful framing --
# same threshold sam_vla.core.ghost_mask.project_or_clamp_body_point_to_pixel
# uses (back_margin_deg=25 from 180) for its own "put it on the bottom edge"
# case, kept here for visual consistency between the VLM and deterministic
# ghost-mask paths.
_DIRECTLY_BEHIND_THRESHOLD_DEG = 180.0 - 25.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _edge_fraction(abs_bearing_deg: float) -> float:
    for upper, frac in _BEARING_BUCKETS:
        if abs_bearing_deg <= upper:
            return frac
    return _BEARING_BUCKETS[-1][1]


def _suggested_placement(context: GhostMaskContext) -> dict:
    """Computes the same placement a deterministic caller-side projection
    would land on, packaged for the prompt as a concrete anchor rather than
    left for the model to derive. Returns pixel-space u/v/radius_u_px/
    radius_v_px plus the bucket bounds used, so the prompt can show its
    work."""
    w, h = context.frame_wh
    cx = w / 2.0
    bearing = context.bearing_deg
    abs_bearing = abs(bearing)
    side = "left" if bearing > 0 else "right" if bearing < 0 else "ahead"

    frac = _edge_fraction(abs_bearing)
    edge_u = 0.0 if side == "left" else float(w - 1) if side == "right" else cx
    u = cx + frac * (edge_u - cx)

    v = float(h - 1) if abs_bearing >= _DIRECTLY_BEHIND_THRESHOLD_DEG else h * 0.45

    ru = _clamp(
        context.min_radius_px
        + (context.max_radius_px - context.min_radius_px)
        * (context.bearing_uncertainty_deg / _REFERENCE_BEARING_UNCERTAINTY_DEG),
        context.min_radius_px,
        context.max_radius_px,
    )
    rv = _clamp(
        context.min_radius_px
        + (context.max_radius_px - context.min_radius_px)
        * (context.distance_uncertainty_m / _REFERENCE_DISTANCE_UNCERTAINTY_M),
        context.min_radius_px,
        context.max_radius_px,
    )

    return {
        "side": side,
        "frac": frac,
        "u": u,
        "v": v,
        "radius_u_px": ru,
        "radius_v_px": rv,
    }


def _bucket_table_block(context: GhostMaskContext) -> str:
    w, h = context.frame_wh
    cx = w / 2.0
    bearing = context.bearing_deg
    side = "left" if bearing > 0 else "right" if bearing < 0 else "ahead"
    edge_u = 0.0 if side == "left" else float(w - 1) if side == "right" else cx

    rows = []
    prev_upper = 0.0
    for upper, frac in _BEARING_BUCKETS:
        u_val = cx + frac * (edge_u - cx)
        rows.append(f"    {prev_upper:.0f}-{upper:.0f} deg -> u ~ {u_val:.0f}")
        prev_upper = upper

    return (
        f"Bearing-magnitude -> horizontal position table (toward the {side} "
        f"edge, center u={cx:.0f}, edge u={edge_u:.0f}):\n" + "\n".join(rows)
    )


def _context_block(context: GhostMaskContext) -> str:
    w, h = context.frame_wh
    cx = w / 2.0
    side = "left" if context.bearing_deg > 0 else "right" if context.bearing_deg < 0 else "ahead"
    behind = abs(context.bearing_deg) > 90.0
    behind_note = (
        f" The bearing magnitude ({context.bearing_deg:+.0f} degrees) means the "
        f"goal is actually BEHIND the rover, not just off to one side.\n"
        if behind
        else ""
    )
    suggestion = _suggested_placement(context)
    return (
        f"The rover's goal is currently out of view. Belief estimate: bearing "
        f"{context.bearing_deg:+.0f} degrees from straight ahead ({side}), "
        f"distance {context.distance_m:.1f}m.\n"
        f"Direction confidence: bearing uncertainty is "
        f"+/-{context.bearing_uncertainty_deg:.0f} degrees (small = confident in "
        f"the {side} direction, large = the goal could be anywhere across a wide "
        f"arc).\n"
        f"Distance confidence: distance uncertainty is "
        f"+/-{context.distance_uncertainty_m:.1f}m (small = confident how far "
        f"away, large = could be much nearer or farther).\n"
        f"Frame size is {w}x{h} pixels; horizontal center is u={cx:.0f}.\n"
        f"{behind_note}"
        f"{_bucket_table_block(context)}\n"
        f"Your bearing ({context.bearing_deg:+.0f} deg) falls in the "
        f"{suggestion['frac']:.2f}-fraction row above: suggested u ~ "
        f"{suggestion['u']:.0f}, v ~ {suggestion['v']:.0f}.\n"
        f"Suggested radius_u_px ~ {suggestion['radius_u_px']:.0f} (scales with "
        f"bearing uncertainty), radius_v_px ~ {suggestion['radius_v_px']:.0f} "
        f"(scales with distance uncertainty), both within "
        f"[{context.min_radius_px:.0f}, {context.max_radius_px:.0f}]px.\n"
        f"Use the suggested numbers as your answer by default. Only move away "
        f"from them if the image itself gives you a real reason to (e.g. "
        f"visible terrain suggesting the goal is more/less occluded than the "
        f"uncertainty implies) -- and even then, stay close to the suggested "
        f"values rather than picking an unrelated number. Do not place the "
        f"ellipse at u={cx:.0f} unless the suggested u above is also near "
        f"{cx:.0f}."
    )


def build_ghost_mask_prompt(context: GhostMaskContext) -> str:
    suggestion = _suggested_placement(context)
    return (
        f"{_SYSTEM_FRAMING}\n\n"
        f"{_context_block(context)}\n\n"
        f"{_JSON_SCHEMA_LINE}\n"
        f"Optionally, one short sentence first, e.g. \"Bearing is {context.bearing_deg:+.0f} "
        f"degrees so I'll use the suggested placement.\" Then the JSON, e.g. "
        f'{{"u": {suggestion["u"]:.0f}, "v": {suggestion["v"]:.0f}, '
        f'"radius_u_px": {suggestion["radius_u_px"]:.0f}, '
        f'"radius_v_px": {suggestion["radius_v_px"]:.0f}}}'
    )


def build_ghost_mask_reprompt(context: GhostMaskContext) -> str:
    return (
        build_ghost_mask_prompt(context)
        + "\n\nYour previous answer could not be parsed. Respond with ONLY the JSON "
        "object this time, no sentence, no other words."
    )


if __name__ == "__main__":
    demo = GhostMaskContext(
        bearing_deg=30.0,
        distance_m=4.5,
        bearing_uncertainty_deg=15.0,
        distance_uncertainty_m=1.2,
        frame_wh=(640, 480),
        min_radius_px=3.0,
        max_radius_px=260.0,
    )
    print(build_ghost_mask_prompt(demo))
