"""
Shared enums and dataclasses for vl_direction, per next.md sec 2/3.2. Plain
@dataclass + validate() raising ValueError, matching sam_vla/core/types.py's
style -- not pydantic, despite next.md's loose "(pydantic/dataclass)"
suggestion -- so this module has zero non-stdlib dependencies.

Frame representation: next.md types query()'s input as list[Frame], but no
Frame type exists anywhere in this codebase -- every other module
(Observation.rgb in sam_vla/core/types.py, kb_teleop_env.py) just passes
raw np.ndarray (uint8, HWC, RGB) images around. This module does the same;
there is no Frame class here, callers pass list[np.ndarray] directly.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from vl_direction.intervention.mode_flag import SessionMode


class Direction(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FRONT = "FRONT"
    BACK = "BACK"


class IdentityToken(str, Enum):
    CBF = "cbf"
    EXPLORATION = "exploration"
    UNCERTAINTY = "uncertainty"
    GHOST_MASK = "ghost_mask"


class UncertaintyStatus(str, Enum):
    NEEDS_HUMAN_INPUT = "NEEDS_HUMAN_INPUT"
    HEADING_DIRECTIVE = "HEADING_DIRECTIVE"


def _validate_bbox_xyxy(bbox: tuple, frame_wh: tuple, name: str) -> None:
    x1, y1, x2, y2 = bbox
    w, h = frame_wh
    if w <= 0 or h <= 0:
        raise ValueError(f"{name}: frame_wh must be positive, got {frame_wh!r}")
    for coord_name, coord in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
        if coord < 0:
            raise ValueError(f"{name}: {coord_name}={coord!r} must be >= 0")
    if x1 >= x2:
        raise ValueError(f"{name}: x1 ({x1!r}) must be < x2 ({x2!r})")
    if y1 >= y2:
        raise ValueError(f"{name}: y1 ({y1!r}) must be < y2 ({y2!r})")
    if x2 > w or y2 > h:
        raise ValueError(f"{name}: bbox {bbox!r} exceeds frame_wh {frame_wh!r}")


@dataclass
class CBFContext:
    bbox_xyxy: tuple[int, int, int, int]
    frame_wh: tuple[int, int]

    def validate(self) -> None:
        _validate_bbox_xyxy(self.bbox_xyxy, self.frame_wh, "CBFContext.bbox_xyxy")


@dataclass
class ExplorationContext:
    task_str: str
    vague_hint: Optional[str] = None

    def validate(self) -> None:
        if not self.task_str.strip():
            raise ValueError("ExplorationContext.task_str must be non-empty")


@dataclass
class GhostMaskContext:
    """Belief state handed to the model for GHOST_MASK mode: the model, not
    caller-side geometry, decides where/how large the ghost mask should be.
    bearing_deg is body-frame, 0=straight ahead, positive=left, matching
    BeliefGoalTracker's [forward, left] convention; distance_m is the
    straight-line body-frame range to the belief point.

    bearing_uncertainty_deg / distance_uncertainty_m are two independent
    spread scalars, not one -- callers with a full Gaussian belief (mean +
    2x2 covariance in robot [forward, left] frame) resolve the covariance
    into the belief's own radial/tangential axes via
    sam_vla.core.ghost_mask.belief_to_bearing_range_uncertainty rather than
    handing the model raw matrix entries in a Cartesian frame it has no
    spatial intuition for; callers with only a scalar uncertainty (no real
    covariance tracked) pass it for both. This lets the model draw an
    ellipse whose horizontal/vertical extent independently reflect
    direction-confidence vs. distance-confidence instead of a uniform
    circle."""

    bearing_deg: float
    distance_m: float
    bearing_uncertainty_deg: float
    distance_uncertainty_m: float
    frame_wh: tuple[int, int]
    min_radius_px: float
    max_radius_px: float

    def validate(self) -> None:
        w, h = self.frame_wh
        if w <= 0 or h <= 0:
            raise ValueError(
                f"GhostMaskContext.frame_wh must be positive, got {self.frame_wh!r}"
            )
        if self.distance_m < 0:
            raise ValueError(
                f"GhostMaskContext.distance_m must be >= 0, got {self.distance_m!r}"
            )
        if self.bearing_uncertainty_deg < 0:
            raise ValueError(
                "GhostMaskContext.bearing_uncertainty_deg must be >= 0, got "
                f"{self.bearing_uncertainty_deg!r}"
            )
        if self.distance_uncertainty_m < 0:
            raise ValueError(
                "GhostMaskContext.distance_uncertainty_m must be >= 0, got "
                f"{self.distance_uncertainty_m!r}"
            )
        if self.min_radius_px < 0 or self.max_radius_px < self.min_radius_px:
            raise ValueError(
                "GhostMaskContext radius bounds invalid: "
                f"min_radius_px={self.min_radius_px!r}, max_radius_px={self.max_radius_px!r}"
            )


@dataclass
class GhostMaskPayload:
    """The model's chosen ghost-mask placement -- pixel center + an
    axis-aligned ellipse's horizontal/vertical radii. Always clamped into
    frame_wh / [min_radius_px, max_radius_px] by the parser before this is
    constructed, so downstream rendering code never needs to re-validate
    model output. radius_u_px == radius_v_px draws the same shape the old
    single-radius circle did."""

    u: float
    v: float
    radius_u_px: float
    radius_v_px: float


@dataclass
class HeadingResponse:
    angle_deg: Optional[float] = None
    angle_range_deg: Optional[tuple[float, float]] = None

    def validate(self) -> None:
        if self.angle_deg is None and self.angle_range_deg is None:
            raise ValueError("HeadingResponse requires angle_deg or angle_range_deg")
        if self.angle_deg is not None and self.angle_range_deg is not None:
            raise ValueError(
                "HeadingResponse: supply angle_deg OR angle_range_deg, not both"
            )
        if (
            self.angle_range_deg is not None
            and self.angle_range_deg[0] >= self.angle_range_deg[1]
        ):
            raise ValueError(
                f"HeadingResponse.angle_range_deg must be (low, high) with low < high, "
                f"got {self.angle_range_deg!r}"
            )


@dataclass
class UncertaintyContext:
    covariance_value: float
    threshold_used: float
    rover_front_reference_deg: float = 0.0
    human_heading_response: Optional[HeadingResponse] = None
    # Retry/traversal-budget state, threaded through this per-call-pure
    # function by the caller (see uncertainty_session.py) -- additive to
    # next.md's literal sec 3.2 listing, required by sec 4's retry loop and
    # sec 9's "attempt counter increments" test requirement.
    attempt: int = 0
    max_units: Optional[float] = None

    def validate(self) -> None:
        if self.human_heading_response is not None:
            self.human_heading_response.validate()
        if self.attempt < 0:
            raise ValueError(
                f"UncertaintyContext.attempt must be >= 0, got {self.attempt!r}"
            )


@dataclass
class UncertaintyPayload:
    status: UncertaintyStatus
    rover_front_reference_deg: float
    heading_deg: Optional[float] = None
    heading_range_deg: Optional[tuple[float, float]] = None
    max_units: Optional[float] = None
    attempt: int = 0


@dataclass
class VLDirectiveResult:
    identity_token: IdentityToken
    direction: Optional[Direction]
    confidence: float
    raw_response: str
    parse_ok: bool
    latency_ms: float
    frame_count: int
    timestamp: str
    episode_id: str
    call_id: str
    session_mode: SessionMode
    uncertainty_payload: Optional[UncertaintyPayload] = None
    ghost_mask_payload: Optional[GhostMaskPayload] = None

    def validate(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"VLDirectiveResult.confidence must be in [0, 1], got {self.confidence!r}"
            )
        is_uncertainty = self.identity_token == IdentityToken.UNCERTAINTY
        is_ghost_mask = self.identity_token == IdentityToken.GHOST_MASK
        if is_uncertainty:
            if self.direction is not None:
                raise ValueError(
                    "VLDirectiveResult.direction must be None for uncertainty mode"
                )
            if self.uncertainty_payload is None:
                raise ValueError(
                    "VLDirectiveResult.uncertainty_payload must be set for uncertainty mode"
                )
            if self.ghost_mask_payload is not None:
                raise ValueError(
                    "VLDirectiveResult.ghost_mask_payload must be None for uncertainty mode"
                )
        elif is_ghost_mask:
            if self.direction is not None:
                raise ValueError(
                    "VLDirectiveResult.direction must be None for ghost_mask mode"
                )
            if self.uncertainty_payload is not None:
                raise ValueError(
                    "VLDirectiveResult.uncertainty_payload must be None for ghost_mask mode"
                )
            if self.parse_ok and self.ghost_mask_payload is None:
                raise ValueError(
                    "VLDirectiveResult.ghost_mask_payload must be set when parse_ok is True"
                )
        else:
            if self.uncertainty_payload is not None:
                raise ValueError(
                    f"VLDirectiveResult.uncertainty_payload must be None for {self.identity_token!r} mode"
                )
            if self.ghost_mask_payload is not None:
                raise ValueError(
                    f"VLDirectiveResult.ghost_mask_payload must be None for {self.identity_token!r} mode"
                )
            if self.parse_ok and self.direction is None:
                raise ValueError(
                    "VLDirectiveResult.direction must be set when parse_ok is True"
                )
