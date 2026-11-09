"""System-2 pixel-goal grounding: a frozen VLM that turns an instruction into a
goal mask the existing object-agnostic policy already consumes.

Design (DualVLN-style dual system, but the slow VLM is FROZEN -- no finetuning):

  * System 2 (slow, every `every` steps): a frozen VLM grounds the language
    instruction to a 2D pixel goal (u, v) in the current RGB frame.
  * That pixel is rendered into a small goal-mask blob and handed to the SAME
    pipeline slot the ground-truth category mask used to fill -- so belief,
    occupancy foresight, ForesightGate and EpistemicGate are all unchanged.
  * System 1 (fast, every step): between VLM calls the goal mask is empty, and
    the SubgoalBeliefBank propagates the last grounded goal by odometry. The
    belief/occlusion machinery IS the async slow/fast bridge.

The only new capability is instruction -> pixel. Everything else is reuse.

`QwenVLPixelGoal` runs Qwen2.5-VL frozen (4-bit fits a 24 GB 4090 at inference;
you cannot *train* it there, and you don't need to). `StubPixelGoal` lets you
test the whole wiring without loading the 7B.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PixelGoal:
    u: float          # column (x) in image pixels
    v: float          # row (y) in image pixels
    confidence: float
    in_view: bool = True


class PixelGoalGrounder:
    """Interface: (rgb, instruction) -> PixelGoal."""

    def ground(self, rgb: np.ndarray, instruction: str) -> PixelGoal:  # pragma: no cover - interface
        raise NotImplementedError


class StubPixelGoal(PixelGoalGrounder):
    """Deterministic grounder for testing the wiring without a VLM.

    `uv` is either a fixed (u, v) or (fraction_x, fraction_y) point, or a
    callable (rgb, instruction) -> (u, v). Lets you validate the rollout
    integration end-to-end before pointing it at the 7B model.
    """

    def __init__(
        self,
        uv: Tuple[float, float] | Callable = (0.5, 0.5),
        confidence: float = 1.0,
        as_fraction: bool = True,
    ):
        self.uv = uv
        self.confidence = float(confidence)
        self.as_fraction = bool(as_fraction)

    def ground(self, rgb: np.ndarray, instruction: str) -> PixelGoal:
        h, w = rgb.shape[:2]
        if callable(self.uv):
            u, v = self.uv(rgb, instruction)
        elif self.as_fraction:
            u, v = self.uv[0] * (w - 1), self.uv[1] * (h - 1)
        else:
            u, v = self.uv
        in_view = 0 <= u < w and 0 <= v < h
        return PixelGoal(float(u), float(v), self.confidence, in_view)


class QwenVLPixelGoal(PixelGoalGrounder):
    """Frozen Qwen2.5-VL grounder. Inference only -- no gradients, no finetuning.

    Loads on first use. Requires `transformers` + `qwen-vl-utils` and a GPU.
    4-bit loading keeps the 7B within a 24 GB 4090; use the 3B model for more
    headroom. The model is never trained -- this is zero-shot spatial grounding.
    """

    DEFAULT_PROMPT = (
        "You are guiding a ground robot. Instruction: \"{instruction}\". "
        "Point to the single best next waypoint to move toward in THIS image. "
        "Reply with only one pixel coordinate as (x, y)."
    )

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_new_tokens: int = 64,
        prompt_template: Optional[str] = None,
        min_confidence: float = 0.0,
    ):
        self.model_id = model_id
        self.device = device
        self.load_in_4bit = bool(load_in_4bit)
        self.max_new_tokens = int(max_new_tokens)
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.min_confidence = float(min_confidence)
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        kwargs = {"torch_dtype": torch.float16, "device_map": self.device}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs.pop("device_map", None)
            kwargs["device_map"] = "auto"
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_id, **kwargs).eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)

    def ground(self, rgb: np.ndarray, instruction: str) -> PixelGoal:
        self._ensure_loaded()
        import torch
        from PIL import Image

        h, w = rgb.shape[:2]
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        prompt = self.prompt_template.format(instruction=instruction)
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[0, inputs["input_ids"].shape[1]:]
        answer = self._processor.decode(gen, skip_special_tokens=True)
        uv = parse_pixel_coordinate(answer, image_size=(w, h))
        if uv is None:
            return PixelGoal(w / 2.0, h / 2.0, 0.0, in_view=False)
        u, v = uv
        return PixelGoal(float(u), float(v), 1.0, in_view=(0 <= u < w and 0 <= v < h))


def parse_pixel_coordinate(text: str, image_size: Tuple[int, int]) -> Optional[Tuple[float, float]]:
    """Pull the first (x, y) pixel pair out of a VLM answer.

    Handles bare ``(x, y)``, ``x, y``, JSON-ish ``[x, y]`` and Qwen box tags.
    If the numbers look normalized (<=1) they are scaled by the image size; if
    they look like Qwen's 0-1000 grounding scale they are rescaled too.
    """
    w, h = image_size
    nums = re.findall(r"-?\d+\.?\d*", text)
    if len(nums) < 2:
        return None
    x, y = float(nums[0]), float(nums[1])
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:  # normalized
        return x * (w - 1), y * (h - 1)
    if x > w or y > h:  # likely Qwen 0-1000 grounding scale
        return x / 1000.0 * (w - 1), y / 1000.0 * (h - 1)
    return x, y


def render_goal_mask(
    goal: Optional[PixelGoal],
    height: int,
    width: int,
    radius: int = 12,
) -> np.ndarray:
    """Render a pixel goal into a disc blob -- a drop-in for the goal mask channel."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if goal is None or not goal.in_view:
        return mask
    cu = int(round(goal.u))
    cv = int(round(goal.v))
    r = int(max(radius, 1))
    y0, y1 = max(cv - r, 0), min(cv + r + 1, height)
    x0, x1 = max(cu - r, 0), min(cu + r + 1, width)
    if y0 >= y1 or x0 >= x1:
        return mask
    ys, xs = np.ogrid[y0:y1, x0:x1]
    disc = (xs - cu) ** 2 + (ys - cv) ** 2 <= r * r
    mask[y0:y1, x0:x1][disc] = 1
    return mask


@dataclass
class ExploreDecision:
    direction: str        # "left" | "right" | "back" | "forward"
    raw: str = ""         # the VLM's raw answer (for logging/debug)


class QwenExplorer:
    """Frozen Qwen2.5-VL as a FREE-SPACE explorer -- picks the next explore
    primitive (left/right/back/forward) from the current view + a short memory of
    directions already tried.

    It is NOT a target detector. In an unfamiliar Mars scene the VLM has no
    semantic goal to recognise, so the *only* thing asked of it is a geometric,
    zero-shot judgement it CAN make: "which way is the most open, drivable
    terrain to explore?". The "found" decision is made elsewhere (the goal-mesh
    mask reappearing), never by this model.

    Runs on the slow System-2 cadence (see the rollout's --qwen-explore-every);
    between calls the last decision is held and the fast diffusion policy keeps
    executing it. The chosen primitive is written into the SAME command channel
    the VLA adapter already consumes, so no change to the policy path is needed.

    4-bit loading keeps the 7B within a 24 GB 4090; use the 3B for headroom.
    """

    DEFAULT_PROMPT = (
        "You are steering a ground rover over rough, unfamiliar Mars-like terrain. "
        "A goal is currently OUT OF VIEW and you must explore to reveal it. "
        "Looking ONLY at this image, which single direction leads to the most open, "
        "flat, drivable terrain that would uncover new area? "
        "Recently tried directions (avoid repeating if they led nowhere): {memory}. "
        "Answer with exactly one word: LEFT, RIGHT, BACK, or FORWARD."
    )

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
        load_in_4bit: bool = True,
        max_new_tokens: int = 16,
        memory_len: int = 4,
        prompt_template: Optional[str] = None,
    ):
        self.model_id = model_id
        self.device = device
        self.load_in_4bit = bool(load_in_4bit)
        self.max_new_tokens = int(max_new_tokens)
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.memory: list = []               # recent decisions, most-recent last
        self.memory_len = max(int(memory_len), 0)
        self._model = None
        self._processor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        kwargs = {"torch_dtype": torch.float16, "device_map": self.device}
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs.pop("device_map", None)
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            kwargs["device_map"] = "auto"
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_id, **kwargs).eval()
        self._processor = AutoProcessor.from_pretrained(self.model_id)

    def decide(self, rgb: np.ndarray, extra_hint: str = "") -> ExploreDecision:
        self._ensure_loaded()
        import torch
        from PIL import Image

        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        mem = ", ".join(self.memory) if self.memory else "none yet"
        prompt = self.prompt_template.format(memory=mem)
        if extra_hint:
            prompt = prompt + " " + extra_hint
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[0, inputs["input_ids"].shape[1]:]
        answer = self._processor.decode(gen, skip_special_tokens=True)
        direction = parse_explore_direction(answer)
        if self.memory_len:
            self.memory.append(direction)
            self.memory = self.memory[-self.memory_len:]
        return ExploreDecision(direction=direction, raw=answer.strip())


def parse_explore_direction(text: str, default: str = "left") -> str:
    """Pull a single explore primitive out of a VLM answer. Order matters: check
    the more specific words first. Falls back to `default` if none is found."""
    t = (text or "").lower()
    for word in ("forward", "straight", "back", "reverse", "left", "right"):
        if word in t:
            if word in ("straight",):
                return "forward"
            if word in ("reverse",):
                return "back"
            return word
    return default


class System2Scheduler:
    """Run the slow grounder every `every` steps; hold the last pixel goal.

    Returns ``(goal, refreshed)`` each step. ``refreshed`` is True only on the
    slow steps when the VLM actually ran -- on the fast steps in between, the
    caller should leave the goal mask empty so the belief bank propagates the
    last grounded goal by odometry (the async slow/fast bridge).
    """

    def __init__(self, grounder: PixelGoalGrounder, every: int = 15, min_confidence: float = 0.0):
        self.grounder = grounder
        self.every = max(int(every), 1)
        self.min_confidence = float(min_confidence)
        self.last_goal: Optional[PixelGoal] = None

    def step(self, t: int, rgb: np.ndarray, instruction: str) -> Tuple[Optional[PixelGoal], bool]:
        if t % self.every != 0:
            return self.last_goal, False
        goal = self.grounder.ground(rgb, instruction)
        if goal.confidence < self.min_confidence:
            return self.last_goal, False
        self.last_goal = goal
        return goal, True
