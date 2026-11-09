#!/usr/bin/env python3
"""Translate vague language into ghost-mask PointGoals for custom NavDP.

Qwen is called once to produce a small STOP/TURN/STRAIGHT program.  At every
control frame this module renders the active command as a green ghost mask,
converts that mask to NavDP's numeric ``[forward, left]`` PointGoal, and returns
it to the rollout.  It never emits a velocity or yaw-rate action: custom NavDP
and its particle guidance remain responsible for all rover motion.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image


QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""

    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class MovementCommand:
    """One validated command produced by Qwen."""

    kind: str
    direction: Optional[str] = None
    angle_degrees: float = 90.0
    distance_m: Optional[float] = None
    until_object: Optional[str] = None


@dataclass(frozen=True)
class ObjectDetection:
    label: str
    score: float
    box_xyxy: np.ndarray


@dataclass(frozen=True)
class MovementDecision:
    """Ghost-mask PointGoal and command state for one control frame."""

    point_goal: np.ndarray
    goal_mask: np.ndarray
    command_index: int
    command_name: str
    done: bool
    target_visible: bool
    target_detection: Optional[ObjectDetection]


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"Qwen did not return a JSON object: {text!r}")
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Qwen response must be a JSON object")
    return value


def validate_command_program(
    response_text: str,
    *,
    default_turn_degrees: float = 90.0,
    default_straight_distance_m: float = 5.0,
) -> list[MovementCommand]:
    """Validate Qwen JSON; arbitrary model text never reaches the controller."""

    payload = _extract_json(response_text)
    raw_commands = payload.get("commands")
    if not isinstance(raw_commands, list) or not raw_commands:
        raise ValueError("Qwen JSON must contain a non-empty 'commands' list")

    commands: list[MovementCommand] = []
    for index, raw in enumerate(raw_commands):
        if not isinstance(raw, dict):
            raise ValueError(f"commands[{index}] must be an object")
        kind = str(raw.get("type", "")).strip().upper()

        if kind == "STOP":
            commands.append(MovementCommand(kind="STOP"))
            continue

        if kind == "TURN":
            direction = str(raw.get("direction", "")).strip().upper()
            if direction not in {"LEFT", "RIGHT", "AUTO"}:
                raise ValueError(
                    f"commands[{index}].direction must be LEFT, RIGHT, or AUTO"
                )
            angle = float(raw.get("angle_degrees", default_turn_degrees))
            if not 0.0 < angle <= 180.0:
                raise ValueError(
                    f"commands[{index}].angle_degrees must be in (0, 180]"
                )
            commands.append(
                MovementCommand(kind="TURN", direction=direction, angle_degrees=angle)
            )
            continue

        if kind == "STRAIGHT":
            distance_value = raw.get("distance_m")
            until_value = raw.get("until")
            until_object: Optional[str] = None
            if isinstance(until_value, dict):
                until_type = str(until_value.get("type", "")).strip().upper()
                if until_type != "OBJECT_VISIBLE":
                    raise ValueError(
                        f"commands[{index}].until.type must be OBJECT_VISIBLE"
                    )
                until_object = str(until_value.get("label", "")).strip()
                if not until_object:
                    raise ValueError(f"commands[{index}].until.label cannot be empty")
            elif until_value is not None:
                raise ValueError(f"commands[{index}].until must be an object")

            if distance_value is None and until_object is None:
                distance_m: Optional[float] = float(default_straight_distance_m)
            elif distance_value is None:
                distance_m = None
            else:
                distance_m = float(distance_value)
                if distance_m <= 0.0:
                    raise ValueError(f"commands[{index}].distance_m must be positive")
            commands.append(
                MovementCommand(
                    kind="STRAIGHT",
                    distance_m=distance_m,
                    until_object=until_object,
                )
            )
            continue

        raise ValueError(f"commands[{index}].type must be STOP, TURN, or STRAIGHT")

    # STOP is terminal. Removing commands after it prevents accidental motion.
    first_stop = next(
        (index for index, command in enumerate(commands) if command.kind == "STOP"),
        None,
    )
    return commands if first_stop is None else commands[: first_stop + 1]


def qwen_prompt(instruction: str) -> str:
    """Restricted prompt for vague rover language."""

    return f"""You are a rover movement-command parser.
Convert the user's vague instruction into JSON only. Do not add prose.

Allowed commands:
1. {{"type":"STOP"}}
2. {{"type":"TURN","direction":"LEFT|RIGHT|AUTO","angle_degrees":number}}
3. {{"type":"STRAIGHT","distance_m":number}}
4. {{"type":"STRAIGHT","until":{{"type":"OBJECT_VISIBLE","label":"object"}}}}

Rules:
- stop, halt, wait, or do not move -> STOP only.
- turn/rotate left or right -> TURN; use 90 degrees when no angle is stated.
- turn/rotate with no side -> TURN AUTO; use 90 degrees when no angle is stated.
- move/go left -> TURN LEFT 90, then STRAIGHT.
- move/go right -> TURN RIGHT 90, then STRAIGHT.
- go back, move back, come back, or turn around -> TURN AUTO 180; add
  STRAIGHT when the user asks to travel back rather than only face backward.
- straight/forward -> STRAIGHT; omit distance_m when it is vague.
- find/continue until an object is seen -> STRAIGHT with OBJECT_VISIBLE.
- preserve action order.
- never output velocities, yaw rates, pixels, masks, coordinates, or explanations.

Return exactly {{"commands":[...]}}.
User instruction: {instruction}
"""


def parse_instruction_with_qwen(
    instruction: str,
    *,
    model_id: str = QWEN_MODEL_ID,
    device: str = "auto",
    max_new_tokens: int = 256,
    default_turn_degrees: float = 90.0,
    default_straight_distance_m: float = 5.0,
) -> tuple[list[MovementCommand], str]:
    """Run Qwen once and release it before loading the navigation models."""

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_id)
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": qwen_prompt(instruction)}],
        }
    ]
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[chat_text], padding=True, return_tensors="pt").to(
        model.device
    )
    generated = model.generate(
        **inputs, max_new_tokens=int(max_new_tokens), do_sample=False
    )
    trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated)
    ]
    response_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    commands = validate_command_program(
        response_text,
        default_turn_degrees=default_turn_degrees,
        default_straight_distance_m=default_straight_distance_m,
    )
    del generated, inputs, model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return commands, response_text


class GroundingDinoDetector:
    """Reusable detector for STRAIGHT-until-object commands."""

    def __init__(
        self,
        *,
        model_id: str = GROUNDING_DINO_MODEL_ID,
        device: str = "auto",
        box_threshold: float = 0.40,
        text_threshold: float = 0.30,
    ) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, device_map=device
        )
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)

    def detect(self, rgb: np.ndarray, label: str) -> Optional[ObjectDetection]:
        import torch

        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        inputs = self.processor(images=image, text=[[label]], return_tensors="pt").to(
            self.model.device
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        if len(result["scores"]) == 0:
            return None
        best = int(torch.argmax(result["scores"]).item())
        return ObjectDetection(
            label=label,
            score=float(result["scores"][best].item()),
            box_xyxy=result["boxes"][best].detach().cpu().numpy().astype(np.float32),
        )


def circle_mask(
    height: int, width: int, u: float, v: float, radius: int
) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    return (((xx - u) ** 2 + (yy - v) ** 2) <= radius**2).astype(np.uint8)


def mask_to_pointgoal(
    mask: np.ndarray, intrinsic: np.ndarray, distance_m: float
) -> np.ndarray:
    """Convert a mask centroid into NavDP ``[forward, left]`` coordinates."""

    rows, columns = np.nonzero(np.asarray(mask) > 0)
    if columns.size == 0:
        raise ValueError("goal mask is empty")
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ValueError("intrinsic must have shape [3,3]")
    u = float(np.mean(columns))
    image_right = (u - float(intrinsic[0, 2])) / float(intrinsic[0, 0])
    direction = np.asarray([1.0, -image_right], dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    return (float(distance_m) * direction).astype(np.float32)


class QwenMovementExecutor:
    """Turn the validated program into NavDP-only ghost-mask PointGoals."""

    def __init__(
        self,
        commands: Sequence[MovementCommand],
        *,
        pointgoal_lookahead_m: float = 4.0,
        turn_tolerance_degrees: float = 4.0,
        ghost_mask_radius: int = 18,
        object_confirm_frames: int = 3,
        detector: Optional[GroundingDinoDetector] = None,
    ) -> None:
        if not commands:
            raise ValueError("at least one movement command is required")
        if pointgoal_lookahead_m <= 0.0:
            raise ValueError("pointgoal_lookahead_m must be positive")
        if ghost_mask_radius < 1:
            raise ValueError("ghost_mask_radius must be positive")
        if object_confirm_frames < 1:
            raise ValueError("object_confirm_frames must be at least one")
        self.commands = list(commands)
        self.pointgoal_lookahead_m = float(pointgoal_lookahead_m)
        self.turn_tolerance = math.radians(float(turn_tolerance_degrees))
        self.ghost_mask_radius = int(ghost_mask_radius)
        self.object_confirm_frames = int(object_confirm_frames)
        self.detector = detector

        self.command_index = 0
        self._entered_index = -1
        self._target_heading = 0.0
        self._straight_start_xz = np.zeros(2, dtype=np.float64)
        self._previous_yaw = 0.0
        self._turn_progress = 0.0
        self._turn_sign = 1.0
        self._consecutive_detections = 0

    @property
    def done(self) -> bool:
        return self.command_index >= len(self.commands)

    def _enter_current_command(self, position_xz: np.ndarray, yaw: float) -> None:
        if self.done or self._entered_index == self.command_index:
            return
        command = self.commands[self.command_index]
        if command.kind == "TURN":
            # AUTO is intentionally stable: choose left once and let custom
            # NavDP/particle guidance avoid obstacles while executing the turn.
            self._turn_sign = -1.0 if command.direction == "RIGHT" else 1.0
            self._turn_progress = 0.0
            self._previous_yaw = wrap_angle(yaw)
        elif command.kind == "STRAIGHT":
            self._target_heading = wrap_angle(yaw)
            self._straight_start_xz = np.asarray(position_xz, dtype=np.float64).copy()
            self._consecutive_detections = 0
        self._entered_index = self.command_index

    def _advance(self) -> None:
        self.command_index += 1
        self._entered_index = -1

    def _mask_for_bearing(
        self, rgb: np.ndarray, intrinsic: np.ndarray, bearing: float
    ) -> np.ndarray:
        height, width = np.asarray(rgb).shape[:2]
        intrinsic = np.asarray(intrinsic, dtype=np.float64)
        radius = min(self.ghost_mask_radius, max(min(height, width) // 5, 1))
        fx, cx = float(intrinsic[0, 0]), float(intrinsic[0, 2])
        margin = radius + 2
        maximum_bearing = math.atan2(max(cx - margin, 1.0), max(fx, 1e-6))
        bearing = float(np.clip(bearing, -maximum_bearing, maximum_bearing))
        # Positive local bearing means left, while increasing image u means right.
        u = cx - fx * math.tan(bearing)
        u = float(np.clip(u, margin, width - margin - 1))
        v = float(np.clip(0.62 * height, margin, height - margin - 1))
        return circle_mask(height, width, u, v, radius)

    def _decision(
        self,
        *,
        rgb: np.ndarray,
        intrinsic: np.ndarray,
        bearing: float,
        command_name: str,
        detection: Optional[ObjectDetection] = None,
    ) -> MovementDecision:
        mask = self._mask_for_bearing(rgb, intrinsic, bearing)
        return MovementDecision(
            point_goal=mask_to_pointgoal(
                mask, intrinsic, self.pointgoal_lookahead_m
            ),
            goal_mask=mask,
            command_index=self.command_index,
            command_name=command_name,
            done=False,
            target_visible=detection is not None,
            target_detection=detection,
        )

    def step(
        self,
        *,
        rgb: np.ndarray,
        position_xz: np.ndarray,
        yaw: float,
        intrinsic: np.ndarray,
    ) -> MovementDecision:
        """Return only a ghost-mask PointGoal; NavDP computes every action."""

        position_xz = np.asarray(position_xz, dtype=np.float64).reshape(2)
        while not self.done:
            self._enter_current_command(position_xz, yaw)
            command = self.commands[self.command_index]

            if command.kind == "STOP":
                self.command_index = len(self.commands)
                return self._done_decision(rgb)

            if command.kind == "TURN":
                yaw_delta = wrap_angle(yaw - self._previous_yaw)
                self._turn_progress = max(
                    0.0, self._turn_progress + self._turn_sign * yaw_delta
                )
                self._previous_yaw = wrap_angle(yaw)
                remaining = math.radians(command.angle_degrees) - self._turn_progress
                if remaining <= self.turn_tolerance:
                    self._advance()
                    continue
                # Keep the ghost goal on the image edge for a large turn.  It
                # naturally moves to the centre during the final visible angle.
                desired_bearing = self._turn_sign * remaining
                return self._decision(
                    rgb=rgb,
                    intrinsic=intrinsic,
                    bearing=desired_bearing,
                    command_name=f"TURN_{command.direction}",
                )

            detection = None
            if command.until_object is not None:
                if self.detector is None:
                    raise RuntimeError(
                        "OBJECT_VISIBLE needs Grounding DINO, but no detector exists"
                    )
                detection = self.detector.detect(rgb, command.until_object)
                self._consecutive_detections = (
                    self._consecutive_detections + 1 if detection is not None else 0
                )
                if self._consecutive_detections >= self.object_confirm_frames:
                    self._advance()
                    if self.done:
                        return self._done_decision(rgb, detection)
                    continue

            travelled = float(np.linalg.norm(position_xz - self._straight_start_xz))
            if command.distance_m is not None and travelled >= command.distance_m:
                self._advance()
                if self.done:
                    return self._done_decision(rgb, detection)
                continue

            heading_error = wrap_angle(self._target_heading - yaw)
            return self._decision(
                rgb=rgb,
                intrinsic=intrinsic,
                bearing=heading_error,
                command_name="STRAIGHT",
                detection=detection,
            )

        return self._done_decision(rgb)

    def _done_decision(
        self,
        rgb: np.ndarray,
        detection: Optional[ObjectDetection] = None,
    ) -> MovementDecision:
        height, width = np.asarray(rgb).shape[:2]
        return MovementDecision(
            point_goal=np.zeros(2, dtype=np.float32),
            goal_mask=np.zeros((height, width), dtype=np.uint8),
            command_index=self.command_index,
            command_name="DONE",
            done=True,
            target_visible=detection is not None,
            target_detection=detection,
        )


@dataclass(frozen=True)
class HomotopyDecision:
    """Latched visual choice used to condition every S2Diff candidate."""

    side: str
    circulation_sign: float
    confidence: float
    obstacle_relevant: bool
    queried_qwen: bool
    raw_response: Optional[str]


class VisualQwenHomotopySelector:
    """Ask Qwen once per obstacle episode whether to pass LEFT or RIGHT.

    The input must already contain the red obstacle and green goal overlays.
    LEFT maps to circulation sign -1 in the planner's [forward,left] frame;
    RIGHT maps to +1. The sign remains latched until obstacles have disappeared
    for ``release_clear_frames`` consecutive observations.
    """

    def __init__(
        self,
        *,
        model_id: str = QWEN_MODEL_ID,
        device: str = "auto",
        minimum_obstacle_pixels: int = 30,
        release_clear_frames: int = 8,
        max_new_tokens: int = 64,
    ) -> None:
        if minimum_obstacle_pixels < 1:
            raise ValueError("minimum_obstacle_pixels must be positive")
        if release_clear_frames < 1:
            raise ValueError("release_clear_frames must be positive")
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype="auto", device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.minimum_obstacle_pixels = int(minimum_obstacle_pixels)
        self.release_clear_frames = int(release_clear_frames)
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
        """Deterministic safe fallback if Qwen output cannot be parsed."""

        mask = np.asarray(obstacle_mask) > 0
        midpoint = mask.shape[1] // 2
        left_occupied = int(mask[:, :midpoint].sum())
        right_occupied = int(mask[:, midpoint:].sum())
        return "LEFT" if left_occupied <= right_occupied else "RIGHT"

    @staticmethod
    def prompt() -> str:
        return """You are selecting one obstacle-passing side for a rover.
The current camera image has RED obstacle pixels and a GREEN navigation goal.
Choose the side with a feasible, wider route from the rover (bottom centre)
toward the green goal. LEFT means pass on the image-left side of the red
obstacle; RIGHT means pass on its image-right side. Do not output a trajectory.
Return JSON only: {"pass_side":"LEFT|RIGHT","confidence":0.0}"""

    def _query(self, overlaid_rgb: np.ndarray) -> tuple[str, float, str]:
        import torch

        image = Image.fromarray(np.asarray(overlaid_rgb, dtype=np.uint8), mode="RGB")
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
        raw: Optional[str] = None
        try:
            side, confidence, raw = self._query(overlaid_rgb)
        except Exception as error:
            side, confidence = fallback, 0.0
            raw = f"Qwen side query failed; deterministic fallback={fallback}: {error}"
        self._latched_side = side
        self._latched_confidence = confidence
        return HomotopyDecision(
            side,
            self.side_to_sign(side),
            confidence,
            True,
            True,
            raw,
        )

def build_executor_from_instruction(
    instruction: str,
    *,
    qwen_model_id: str = QWEN_MODEL_ID,
    grounding_dino_model_id: str = GROUNDING_DINO_MODEL_ID,
    qwen_device: str = "auto",
    detector_device: str = "auto",
    default_turn_degrees: float = 90.0,
    default_straight_distance_m: float = 5.0,
    pointgoal_lookahead_m: float = 4.0,
    turn_tolerance_degrees: float = 4.0,
    ghost_mask_radius: int = 18,
    object_confirm_frames: int = 3,
) -> tuple[QwenMovementExecutor, str]:
    """Build Qwen parser, optional Grounding DINO, and command executor."""

    commands, raw_response = parse_instruction_with_qwen(
        instruction,
        model_id=qwen_model_id,
        device=qwen_device,
        default_turn_degrees=default_turn_degrees,
        default_straight_distance_m=default_straight_distance_m,
    )
    needs_detector = any(command.until_object for command in commands)
    detector = (
        GroundingDinoDetector(
            model_id=grounding_dino_model_id, device=detector_device
        )
        if needs_detector
        else None
    )
    return (
        QwenMovementExecutor(
            commands,
            pointgoal_lookahead_m=pointgoal_lookahead_m,
            turn_tolerance_degrees=turn_tolerance_degrees,
            ghost_mask_radius=ghost_mask_radius,
            object_confirm_frames=object_confirm_frames,
            detector=detector,
        ),
        raw_response,
    )


def command_to_dict(command: MovementCommand) -> dict[str, Any]:
    return {
        "type": command.kind,
        "direction": command.direction,
        "angle_degrees": command.angle_degrees,
        "distance_m": command.distance_m,
        "until_object": command.until_object,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a command with Qwen")
    parser.add_argument("instruction")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--qwen-model-id", default=QWEN_MODEL_ID)
    parser.add_argument("--default-turn-degrees", type=float, default=90.0)
    parser.add_argument("--default-straight-distance-m", type=float, default=5.0)
    args = parser.parse_args()

    commands, raw_response = parse_instruction_with_qwen(
        args.instruction,
        model_id=args.qwen_model_id,
        device=args.device,
        default_turn_degrees=args.default_turn_degrees,
        default_straight_distance_m=args.default_straight_distance_m,
    )
    print("Raw Qwen response:")
    print(raw_response)
    print("Validated command program:")
    print(json.dumps({"commands": [command_to_dict(c) for c in commands]}, indent=2))


if __name__ == "__main__":
    main()
