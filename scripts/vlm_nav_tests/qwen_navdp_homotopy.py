# #!/usr/bin/env python3
# """Visual Qwen homotopy selection for custom NavDP.

# Qwen has exactly one role here: when metric obstacle pixels become relevant,
# inspect the RGB frame overlaid with red obstacles and a green fixed PointGoal,
# then choose LEFT or RIGHT. The choice is latched for that obstacle episode and
# is converted to one circulation sign shared by every NavDP/S2Diff candidate.
# Qwen never parses movement commands and never creates goals or actions.
# """

# from __future__ import annotations

# import json
# import re
# from dataclasses import dataclass
# from typing import Any, Optional

# import numpy as np
# from PIL import Image


# QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


# def _extract_json(text: str) -> dict[str, Any]:
#     cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
#     try:
#         payload = json.loads(cleaned)
#     except json.JSONDecodeError:
#         start, end = cleaned.find("{"), cleaned.rfind("}")
#         if start < 0 or end <= start:
#             raise ValueError(f"Qwen returned no JSON object: {text!r}")
#         payload = json.loads(cleaned[start : end + 1])
#     if not isinstance(payload, dict):
#         raise ValueError("Qwen response must be a JSON object")
#     return payload


# @dataclass(frozen=True)
# class HomotopyDecision:
#     side: str
#     circulation_sign: float
#     confidence: float
#     obstacle_relevant: bool
#     queried_qwen: bool
#     raw_response: Optional[str]
#     repeated_sides: tuple[str, ...] = ()
#     repeated_confidences: tuple[float, ...] = ()
#     consistency_rate: float = 1.0
#     used_fallback: bool = False


# class VisualQwenHomotopySelector:
#     """Choose and latch one obstacle-passing side per obstacle episode."""

#     def __init__(
#         self,
#         *,
#         model_id: str = QWEN_MODEL_ID,
#         device: str = "auto",
#         minimum_obstacle_pixels: int = 30,
#         release_clear_frames: int = 8,
#         consistency_repeats: int = 5,
#         max_new_tokens: int = 64,
#     ) -> None:
#         if minimum_obstacle_pixels < 1:
#             raise ValueError("minimum_obstacle_pixels must be positive")
#         if release_clear_frames < 1:
#             raise ValueError("release_clear_frames must be positive")
#         if consistency_repeats < 1:
#             raise ValueError("consistency_repeats must be at least one")
#         from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

#         self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#             model_id, dtype="auto", device_map=device
#         )
#         self.processor = AutoProcessor.from_pretrained(model_id)
#         self.minimum_obstacle_pixels = int(minimum_obstacle_pixels)
#         self.release_clear_frames = int(release_clear_frames)
#         self.consistency_repeats = int(consistency_repeats)
#         self.max_new_tokens = int(max_new_tokens)
#         self._latched_side: Optional[str] = None
#         self._latched_confidence = 0.0
#         self._clear_frames = 0

#     @staticmethod
#     def side_to_sign(side: str) -> float:
#         side = str(side).upper()
#         if side == "LEFT":
#             return -1.0
#         if side == "RIGHT":
#             return 1.0
#         raise ValueError("side must be LEFT or RIGHT")

#     @staticmethod
#     def fallback_side(obstacle_mask: np.ndarray) -> str:
#         mask = np.asarray(obstacle_mask) > 0
#         midpoint = mask.shape[1] // 2
#         left_occupied = int(mask[:, :midpoint].sum())
#         right_occupied = int(mask[:, midpoint:].sum())
#         return "LEFT" if left_occupied <= right_occupied else "RIGHT"

#     @staticmethod
#     def prompt() -> str:
#         return """You are the homotopy selector inside a rover trajectory planner.
# The image shows RED obstacle pixels and the fixed PointGoal in GREEN. Choose
# one side on which ALL trajectory candidates should pass the blocking obstacle.
# LEFT means pass on the image-left side; RIGHT means pass on the image-right
# side. Consider free visible space from the rover at bottom-centre toward the
# green goal. Do not create a goal, turn command, waypoint, or motor action.
# Return JSON only: {"pass_side":"LEFT|RIGHT","confidence":0.0}"""

#     def _query(self, overlaid_rgb: np.ndarray) -> tuple[str, float, str]:
#         import torch

#         image = Image.fromarray(np.asarray(overlaid_rgb, dtype=np.uint8))
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": image},
#                     {"type": "text", "text": self.prompt()},
#                 ],
#             }
#         ]
#         chat_text = self.processor.apply_chat_template(
#             messages, tokenize=False, add_generation_prompt=True
#         )
#         inputs = self.processor(
#             text=[chat_text], images=[image], padding=True, return_tensors="pt"
#         ).to(self.model.device)
#         with torch.inference_mode():
#             generated = self.model.generate(
#                 **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
#             )
#         trimmed = [
#             output_ids[len(input_ids) :]
#             for input_ids, output_ids in zip(inputs.input_ids, generated)
#         ]
#         raw = self.processor.batch_decode(
#             trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#         )[0]
#         payload = _extract_json(raw)
#         side = str(payload.get("pass_side", "")).strip().upper()
#         if side not in {"LEFT", "RIGHT"}:
#             raise ValueError("Qwen pass_side must be LEFT or RIGHT")
#         confidence = float(np.clip(float(payload.get("confidence", 0.0)), 0.0, 1.0))
#         return side, confidence, raw

#     def step(
#         self, overlaid_rgb: np.ndarray, obstacle_mask: np.ndarray
#     ) -> HomotopyDecision:
#         obstacle_mask = np.asarray(obstacle_mask)
#         relevant = int(np.count_nonzero(obstacle_mask)) >= self.minimum_obstacle_pixels
#         if not relevant:
#             self._clear_frames += 1
#             if self._clear_frames >= self.release_clear_frames:
#                 self._latched_side = None
#                 self._latched_confidence = 0.0
#             if self._latched_side is None:
#                 return HomotopyDecision("AUTO", 0.0, 0.0, False, False, None)
#             return HomotopyDecision(
#                 self._latched_side,
#                 self.side_to_sign(self._latched_side),
#                 self._latched_confidence,
#                 False,
#                 False,
#                 None,
#             )

#         self._clear_frames = 0
#         if self._latched_side is not None:
#             return HomotopyDecision(
#                 self._latched_side,
#                 self.side_to_sign(self._latched_side),
#                 self._latched_confidence,
#                 True,
#                 False,
#                 None,
#             )

#         fallback = self.fallback_side(obstacle_mask)
#         valid_results: list[tuple[str, float, str]] = []
#         errors: list[str] = []
#         for repeat_index in range(self.consistency_repeats):
#             try:
#                 valid_results.append(self._query(overlaid_rgb))
#             except Exception as error:
#                 errors.append(f"repeat {repeat_index}: {error}")

#         repeated_sides = tuple(item[0] for item in valid_results)
#         repeated_confidences = tuple(item[1] for item in valid_results)
#         used_fallback = False
#         if valid_results:
#             left_count = repeated_sides.count("LEFT")
#             right_count = repeated_sides.count("RIGHT")
#             if left_count == right_count:
#                 side, used_fallback = fallback, True
#             else:
#                 side = "LEFT" if left_count > right_count else "RIGHT"
#             side_confidences = [item[1] for item in valid_results if item[0] == side]
#             confidence = float(np.mean(side_confidences)) if side_confidences else 0.0
#             consistency_rate = max(left_count, right_count) / len(valid_results)
#         else:
#             side, confidence, consistency_rate = fallback, 0.0, 0.0
#             used_fallback = True

#         raw = json.dumps(
#             {
#                 "repeat_sides": repeated_sides,
#                 "repeat_confidences": repeated_confidences,
#                 "raw_responses": [item[2] for item in valid_results],
#                 "errors": errors,
#                 "majority_side": side,
#                 "consistency_rate": consistency_rate,
#                 "used_fallback": used_fallback,
#             }
#         )
#         self._latched_side = side
#         self._latched_confidence = confidence
#         return HomotopyDecision(
#             side=side,
#             circulation_sign=self.side_to_sign(side),
#             confidence=confidence,
#             obstacle_relevant=True,
#             queried_qwen=True,
#             raw_response=raw,
#             repeated_sides=repeated_sides,
#             repeated_confidences=repeated_confidences,
#             consistency_rate=consistency_rate,
#             used_fallback=used_fallback,
#         )


#!/usr/bin/env python3
"""Visual Qwen homotopy selection for custom NavDP.

Qwen has exactly one role here: when metric obstacle pixels become relevant,
inspect the RGB frame overlaid with red obstacles and a green fixed PointGoal,
then choose LEFT or RIGHT. The choice is latched for that obstacle episode and
is converted to one circulation sign shared by every NavDP/S2Diff candidate.
In the optional round-trip mode, the same isolated model also classifies one
user utterance at the outward goal as RETURN or STOP. It never creates goals,
waypoints, masks, or motor actions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from PIL import Image

QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"Qwen returned no JSON object: {text!r}")
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Qwen response must be a JSON object")
    return payload


@dataclass(frozen=True)
class HomotopyDecision:
    side: str
    circulation_sign: float
    confidence: float
    obstacle_relevant: bool
    queried_qwen: bool
    raw_response: Optional[str]
    repeated_sides: tuple[str, ...] = ()
    repeated_confidences: tuple[float, ...] = ()
    consistency_rate: float = 1.0
    used_fallback: bool = False


@dataclass(frozen=True)
class NavigationCommandDecision:
    command: str
    confidence: float
    raw_response: str


@dataclass(frozen=True)
class MissionPlanDecision:
    plan: tuple[str, ...]
    confidence: float
    raw_response: str


class VisualQwenHomotopySelector:
    """Choose and latch one obstacle-passing side per obstacle episode."""

    def __init__(
        self,
        *,
        model_id: str = QWEN_MODEL_ID,
        device: str = "auto",
        minimum_obstacle_pixels: int = 30,
        release_clear_frames: int = 8,
        consistency_repeats: int = 5,
        max_new_tokens: int = 64,
    ) -> None:
        if minimum_obstacle_pixels < 1:
            raise ValueError("minimum_obstacle_pixels must be positive")
        if release_clear_frames < 1:
            raise ValueError("release_clear_frames must be positive")
        if consistency_repeats < 1:
            raise ValueError("consistency_repeats must be at least one")
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, dtype="auto", device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.minimum_obstacle_pixels = int(minimum_obstacle_pixels)
        self.release_clear_frames = int(release_clear_frames)
        self.consistency_repeats = int(consistency_repeats)
        self.max_new_tokens = int(max_new_tokens)
        self._latched_side: Optional[str] = None
        self._latched_confidence = 0.0
        self._clear_frames = 0

    @staticmethod
    def side_to_sign(side: str) -> float:
        side = str(side).upper()
        if side == "LEFT":
            return -1.0
        if side == "RIGHT":
            return 1.0
        raise ValueError("side must be LEFT or RIGHT")

    @staticmethod
    def fallback_side(obstacle_mask: np.ndarray) -> str:
        mask = np.asarray(obstacle_mask) > 0
        midpoint = mask.shape[1] // 2
        left_occupied = int(mask[:, :midpoint].sum())
        right_occupied = int(mask[:, midpoint:].sum())
        return "LEFT" if left_occupied <= right_occupied else "RIGHT"

    @staticmethod
    def prompt() -> str:
        return """You are the homotopy selector inside a rover trajectory planner.
The image shows RED obstacle pixels and the fixed PointGoal in GREEN. Choose
one side on which ALL trajectory candidates should pass the blocking obstacle.
LEFT means pass on the image-left side; RIGHT means pass on the image-right
side. Consider free visible space from the rover at bottom-centre toward the
green goal. Do not create a goal, turn command, waypoint, or motor action.
Return JSON only: {"pass_side":"LEFT|RIGHT","confidence":0.0}"""

    def _query(self, overlaid_rgb: np.ndarray) -> tuple[str, float, str]:
        import torch

        image = Image.fromarray(np.asarray(overlaid_rgb, dtype=np.uint8))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt()},
                ],
            }
        ]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[chat_text], images=[image], padding=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated)
        ]
        raw = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        payload = _extract_json(raw)
        side = str(payload.get("pass_side", "")).strip().upper()
        if side not in {"LEFT", "RIGHT"}:
            raise ValueError("Qwen pass_side must be LEFT or RIGHT")
        confidence = float(np.clip(float(payload.get("confidence", 0.0)), 0.0, 1.0))
        return side, confidence, raw

    @staticmethod
    def command_prompt(user_command: str) -> str:
        return f"""You are the command gate for a rover that has just reached its
outward destination. Classify the user's request as exactly one command:
RETURN means go back to the recorded spawn/home position; STOP means remain at
the reached destination. Do not create a path, pixel goal, mask, waypoint, or
motor action. Treat phrases such as come back, return, go home, go back, and
return to where you started as RETURN. Treat stop, stay, wait, done, and finish
as STOP.
User request: {json.dumps(str(user_command))}
Return JSON only: {{"command":"RETURN|STOP","confidence":0.0}}"""

    def classify_command(
        self, image_rgb: np.ndarray, user_command: str
    ) -> NavigationCommandDecision:
        import torch

        if not str(user_command).strip():
            raise ValueError("user_command must not be empty")
        image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.command_prompt(user_command)},
                ],
            }
        ]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[chat_text], images=[image], padding=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated)
        ]
        raw = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        payload = _extract_json(raw)
        command = str(payload.get("command", "")).strip().upper()
        if command not in {"RETURN", "STOP"}:
            raise ValueError("Qwen command must be RETURN or STOP")
        confidence = float(np.clip(float(payload.get("confidence", 0.0)), 0.0, 1.0))
        return NavigationCommandDecision(
            command=command,
            confidence=confidence,
            raw_response=raw,
        )

    @staticmethod
    def mission_prompt(user_command: str) -> str:
        return f"""You translate one free-form rover instruction into a tiny
ordered mission plan. The rover starts at HOME and has one OUTWARD GOAL.

Allowed plans:
1. ["GO_TO_GOAL"] ? visit/reach the outward goal and stop there.
2. ["GO_TO_GOAL", "RETURN_HOME"] ? visit/reach the outward goal, then come
   back/return/report back to the recorded spawn position.

Choose the round trip whenever the wording implies returning, coming back,
going back home, making a round trip, visiting and returning, reporting back,
or returning to the starting/previous position. Choose goal-only only when the
instruction clearly ends at the outward goal. Do not invent directions,
waypoints, masks, or motor commands. The navigation controller?not you?decides
when each phase is reached.
User instruction: {json.dumps(str(user_command))}
Return JSON only:
{{"plan":["GO_TO_GOAL","RETURN_HOME"],"confidence":0.0}}"""

    @staticmethod
    def parse_mission_payload(
        payload: dict[str, Any], raw_response: str
    ) -> MissionPlanDecision:
        raw_plan = payload.get("plan")
        if not isinstance(raw_plan, list):
            raise ValueError("Qwen mission plan must be a JSON list")
        plan = tuple(str(item).strip().upper() for item in raw_plan)
        allowed = {
            ("GO_TO_GOAL",),
            ("GO_TO_GOAL", "RETURN_HOME"),
        }
        if plan not in allowed:
            raise ValueError(
                "Qwen mission plan must be GO_TO_GOAL, optionally followed "
                "by RETURN_HOME"
            )
        confidence = float(np.clip(float(payload.get("confidence", 0.0)), 0.0, 1.0))
        return MissionPlanDecision(plan, confidence, raw_response)

    def classify_mission(
        self, image_rgb: np.ndarray, user_command: str
    ) -> MissionPlanDecision:
        import torch

        if not str(user_command).strip():
            raise ValueError("user_command must not be empty")
        image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.mission_prompt(user_command)},
                ],
            }
        ]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[chat_text], images=[image], padding=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated)
        ]
        raw = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return self.parse_mission_payload(_extract_json(raw), raw)

    def step(
        self, overlaid_rgb: np.ndarray, obstacle_mask: np.ndarray
    ) -> HomotopyDecision:
        obstacle_mask = np.asarray(obstacle_mask)
        relevant = int(np.count_nonzero(obstacle_mask)) >= self.minimum_obstacle_pixels
        if not relevant:
            self._clear_frames += 1
            if self._clear_frames >= self.release_clear_frames:
                self._latched_side = None
                self._latched_confidence = 0.0
            if self._latched_side is None:
                return HomotopyDecision("AUTO", 0.0, 0.0, False, False, None)
            return HomotopyDecision(
                self._latched_side,
                self.side_to_sign(self._latched_side),
                self._latched_confidence,
                False,
                False,
                None,
            )

        self._clear_frames = 0
        if self._latched_side is not None:
            return HomotopyDecision(
                self._latched_side,
                self.side_to_sign(self._latched_side),
                self._latched_confidence,
                True,
                False,
                None,
            )

        fallback = self.fallback_side(obstacle_mask)
        valid_results: list[tuple[str, float, str]] = []
        errors: list[str] = []
        for repeat_index in range(self.consistency_repeats):
            try:
                valid_results.append(self._query(overlaid_rgb))
            except Exception as error:
                errors.append(f"repeat {repeat_index}: {error}")

        repeated_sides = tuple(item[0] for item in valid_results)
        repeated_confidences = tuple(item[1] for item in valid_results)
        used_fallback = False
        if valid_results:
            left_count = repeated_sides.count("LEFT")
            right_count = repeated_sides.count("RIGHT")
            if left_count == right_count:
                side, used_fallback = fallback, True
            else:
                side = "LEFT" if left_count > right_count else "RIGHT"
            side_confidences = [item[1] for item in valid_results if item[0] == side]
            confidence = float(np.mean(side_confidences)) if side_confidences else 0.0
            consistency_rate = max(left_count, right_count) / len(valid_results)
        else:
            side, confidence, consistency_rate = fallback, 0.0, 0.0
            used_fallback = True

        raw = json.dumps(
            {
                "repeat_sides": repeated_sides,
                "repeat_confidences": repeated_confidences,
                "raw_responses": [item[2] for item in valid_results],
                "errors": errors,
                "majority_side": side,
                "consistency_rate": consistency_rate,
                "used_fallback": used_fallback,
            }
        )
        self._latched_side = side
        self._latched_confidence = confidence
        return HomotopyDecision(
            side=side,
            circulation_sign=self.side_to_sign(side),
            confidence=confidence,
            obstacle_relevant=True,
            queried_qwen=True,
            raw_response=raw,
            repeated_sides=repeated_sides,
            repeated_confidences=repeated_confidences,
            consistency_rate=consistency_rate,
            used_fallback=used_fallback,
        )
