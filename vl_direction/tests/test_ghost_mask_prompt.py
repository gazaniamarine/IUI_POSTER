from vl_direction.prompts.ghost_mask_prompt import (
    build_ghost_mask_prompt,
    build_ghost_mask_reprompt,
)
from vl_direction.schemas import GhostMaskContext


def _context(**overrides):
    defaults = dict(
        bearing_deg=30.0,
        distance_m=4.5,
        bearing_uncertainty_deg=12.0,
        distance_uncertainty_m=1.2,
        frame_wh=(640, 480),
        min_radius_px=3.0,
        max_radius_px=100.0,
    )
    defaults.update(overrides)
    return GhostMaskContext(**defaults)


def test_prompt_contains_belief_numbers():
    prompt = build_ghost_mask_prompt(_context())
    assert "30" in prompt
    assert "4.5" in prompt
    assert "12" in prompt
    assert "1.2" in prompt


def test_prompt_contains_frame_size_and_radius_bounds():
    prompt = build_ghost_mask_prompt(_context(frame_wh=(320, 240)))
    assert "320x240" in prompt
    assert "3" in prompt
    assert "100" in prompt


def test_prompt_requests_json_schema():
    prompt = build_ghost_mask_prompt(_context())
    assert '"u"' in prompt
    assert '"v"' in prompt
    assert '"radius_u_px"' in prompt
    assert '"radius_v_px"' in prompt


def test_prompt_flags_behind_camera_bearing():
    prompt = build_ghost_mask_prompt(_context(bearing_deg=140.0))
    assert "BEHIND" in prompt


def test_reprompt_adds_corrective_instruction():
    context = _context()
    prompt = build_ghost_mask_prompt(context)
    reprompt = build_ghost_mask_reprompt(context)
    assert reprompt.startswith(prompt)
    assert "ONLY the JSON" in reprompt
