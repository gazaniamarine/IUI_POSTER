from vl_direction.prompts.cbf_prompt import build_cbf_prompt, build_cbf_reprompt
from vl_direction.schemas import CBFContext


def _context():
    return CBFContext(bbox_xyxy=(300, 200, 420, 340), frame_wh=(640, 480))


def test_prompt_contains_bbox_and_frame_size():
    prompt = build_cbf_prompt(_context())
    assert "(300, 200)-(420, 340)" in prompt
    assert "640x480" in prompt


def test_prompt_contains_binary_output_constraint():
    prompt = build_cbf_prompt(_context())
    assert "LEFT" in prompt
    assert "RIGHT" in prompt
    assert "FRONT" not in prompt
    assert "BACK" not in prompt


def test_reprompt_adds_corrective_instruction():
    prompt = build_cbf_prompt(_context())
    reprompt = build_cbf_reprompt(_context())
    assert reprompt.startswith(prompt)
    assert "must answer" in reprompt.lower()
