"""InternVLA-N1-style, detector-free point-goal navigator: Qwen3.5-9B looks at
the raw camera frame plus its own rolling history and directly predicts a
ground point for the current instruction in an ordered instruction queue
(see qwen35_prompts.build_point_goal_history_prompt) -- no SAM/DINO/mask in
this path. RoverController owns one of these (see start_vlm_navigation) and
feeds its output straight into the existing MODE_POINT machinery
(nav.goal_math.body_frame_goal -> NavdpUpstreamPolicy.set_goal_body), the
same ground-truth-point driving path random_goal()/go_home()/
request_forward_point() already use -- this class is only responsible for
deciding *where that point is* and *when to move to the next instruction*.

Re-queried every `replan_every` ticks (mirrors InternVLA-N1's coarser
System-2 cadence vs. System-1 refreshing every tick against the
last-set point, see Nihil/INTERNVL_HIWONDER_CHECKLIST.md section 6); the
rolling history buffer (default cap 8, matching InternVLA-N1's
num_history=8) is cleared whenever the active instruction advances, mirroring
InternVLA-N1's agent.reset() on a new instruction -- history is per-
instruction, not cross-instruction (see
Nihil/reference/internvla_dualvln_zenoh_node.py:544-560).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from sam_vla.core.goal_geometry import pixel_to_world
from sam_vla.core.types import Observation
from nav.object_belief import ObjectBeliefMemory
from sam_vla.vlm import qwen35_client

# Arrival is only accepted if the (verified) target is also measurably close.
ARRIVE_MAX_DIST_M = 3.5
# Instructions that name a destination to find (vs. "go straight", which has
# no target to look for) -- only these trigger the scan-search rotation.
_SEEK_RE = re.compile(
    r"\b(go to|go back to|return to|come back to|back to|find|reach|toward|towards|approach|navigate to|walk to|"
    r"drive to|head to|locate|search|look for|get to|stop (at|near|in front of))\b",
    re.I,
)
_UNIT_M = {"m": 1.0, "meter": 1.0, "metre": 1.0, "ft": 0.3048, "feet": 0.3048,
           "foot": 0.3048, "cm": 0.01}


def parse_simple_command(text: str) -> Optional[dict]:
    """Rule-based parse of unambiguous commands the 4B VLM kept mangling
    (live: "go home" answered as 'forward'). Returns None for anything that
    needs vision (go to <object>, go straight, ...), which goes to Qwen.
    {"action": "stop"|"home"} | {"action": "turn", "direction", "degrees"} |
    {"action": "forward", "distance_m"}."""
    t = text.lower().strip().rstrip(".!")
    if re.match(r"^(please\s+)?(stop|halt|wait|hold|stay|freeze)\b", t):
        return {"action": "stop"}
    if re.search(r"\b(home|base|starting point|start position|spawn)\b", t) and re.search(
        r"\b(go|return|come|head|drive|navigate|back)\b", t
    ):
        return {"action": "home"}
    if re.fullmatch(r"(come back|go back|return)", t):
        return {"action": "home"}
    if re.search(r"\b(backward|backwards|reverse)\b", t):
        return {"action": "turn", "direction": "left", "degrees": 180.0}
    m = re.match(r"^(please\s+)?(turn|rotate|face|spin)\b(.*)$", t)
    if m:
        rest = m.group(3)
        num = re.search(r"(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees|°)?", rest)
        if re.search(r"\b(around|back|about)\b", rest):
            return {"action": "turn", "direction": "left", "degrees": 180.0}
        direction = "left" if "left" in rest else "right" if "right" in rest else None
        if direction is None:
            return None
        if num:
            deg = float(num.group(1))
        elif re.search(r"\b(slightly|a bit|little)\b", rest):
            deg = 30.0
        else:
            deg = 90.0
        return {"action": "turn", "direction": direction, "degrees": min(max(deg, 1.0), 180.0)}
    m = re.match(
        r"^(please\s+)?(go|walk|move|drive|advance|head)\s+(?:straight\s+|forward\s+|ahead\s+)*"
        r"(?:for\s+)?(\d+(?:\.\d+)?)\s*(m|meters?|metres?|ft|feet|foot|cm)\b",
        t,
    )
    if m:
        dist = float(m.group(3)) * _UNIT_M[m.group(4).rstrip("s") if m.group(4) not in ("ft",) else "ft"]
        return {"action": "forward", "distance_m": min(max(dist, 0.1), 50.0)}
    return None


_FLAG_COLORS = ("red", "green", "blue", "yellow", "purple")
_GREY_WORDS = ("grey", "gray", "white", "black")


def named_flag_color(instruction: str) -> Optional[str]:
    """'go to the red flag' -> 'red' (only the sim's solid-colour flags)."""
    t = instruction.lower()
    if "flag" not in t:
        return None
    for c in _FLAG_COLORS:
        if re.search(rf"\b{c}\b", t):
            return c
    if any(re.search(rf"\b{w}\b", t) for w in _GREY_WORDS):
        return "grey"
    return None


def target_phrase(instruction: str) -> Optional[str]:
    """'go to the rover' -> 'rover'; 'find the antenna' -> 'antenna'. None when
    the instruction names no object (e.g. 'go straight')."""
    m = _SEEK_RE.search(instruction)
    if not m:
        return None
    rest = instruction[m.end():].strip(" .,!")
    rest = re.sub(r"^(?:the|a|an|that|this|nearest|closest)\s+", "", rest, flags=re.I)
    rest = re.sub(r"^(?:the|a|an)\s+", "", rest, flags=re.I)
    return rest or None


def detect_color_flag(rgb: np.ndarray, color: str, min_pixels: int = 8, return_mask: bool = False):
    """Find the largest blob of the named solid flag colour. Returns
    (u, v, n_pixels) with u/v normalized to [0,1], or None. Deterministic and
    works on flags only a few pixels wide, where the VLM can't localise."""
    from scipy import ndimage

    a = np.asarray(rgb[..., :3], dtype=np.int32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    if color == "red":
        mask = (r > 70) & (r > 2.0 * g) & (r > 2.0 * b)
    elif color == "green":
        mask = (g > 60) & (g > 1.5 * r) & (g > 1.5 * b)
    elif color == "yellow":
        mask = (r > 120) & (g > 100) & (g > 0.6 * r) & (b < 0.45 * np.minimum(r, g))
    elif color == "purple":
        mask = (b > 110) & (r > 70) & (r > 0.35 * b) & (g < 0.45 * r) & (g < 0.45 * b)
    else:  # blue
        mask = (b > 60) & (b > 2.0 * r) & (b > 2.0 * g)
    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, labels, index=range(1, n + 1))
    best = int(np.argmax(sizes)) + 1
    if sizes[best - 1] < min_pixels:
        return None
    ys, xs = np.nonzero(labels == best)
    h, w = mask.shape
    res = float(xs.mean() / (w - 1)), float(ys.mean() / (h - 1)), int(sizes[best - 1])
    return (*res, labels == best) if return_mask else res


def grey_flag_candidates(rgb: np.ndarray, max_candidates: int = 3):
    """The sim's white flag is a dark grey/black checker -- not distinctive
    by colour (rocks and terrain are grey too), so return up to N compact
    candidate blobs near the horizon band for the VLM to verify. Each is
    (u, v, n_pixels), largest first."""
    from scipy import ndimage

    a = np.asarray(rgb[..., :3], dtype=np.int32)
    mx, mn = a.max(-1), a.min(-1)
    mask = (mx < 150) & (mn > 40) & ((mx - mn) < 22)
    h, w = mask.shape
    mask[: int(h * 0.30)] = False  # sky
    mask[int(h * 0.74):] = False  # ground in front of the rover / its dish
    labels, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labels == i)
        if not (15 <= len(xs) <= 3000):
            continue
        bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        if bw < 0.7 * bh:  # flags are wider than tall; skip tall rocks
            continue
        out.append((float(xs.mean() / (w - 1)), float(ys.mean() / (h - 1)), len(xs)))
    out.sort(key=lambda c: -c[2])
    return out[:max_candidates]


BEARING_ONLY_RANGE_M = 9.0   # assumed range for a sighting with no valid depth
BEARING_ONLY_SIGMA_M = 3.0
STALE_M_PER_POLL = 0.2  # uncertainty added per poll the target stays out of view
UNKNOWN_UNCERTAINTY = 99.0
BLIND_TRAVEL_HANDOVER_M = 10.0  # driven this far at an unmeasured range -> human
# Driving to a remembered (out-of-view) target continues for as long as the
# belief stays under the uncertainty limit. Arriving there (within ARRIVE_M)
# and still not seeing it for ARRIVE_MISSING_POLLS polls -> human.
ARRIVE_M = 2.5
# Turn in place toward a remembered target only when it is well off to the
# side/behind; just outside the (narrow, 55 deg real) view NavDP steers to it
# while driving -- stopping to turn for every few degrees made the rover
# zig-zag (live 2026-11-09).
REMEMBERED_TURN_DEG = 70.0
ARRIVE_MISSING_POLLS = 2
ARRIVE_CONFIRM_POLLS = 2  # consecutive "seen within ARRIVE_M" checks to count as reached
STANDOFF_M = 1.2  # stop this far short of a large object's measured surface point
_ROVER_WORDS = {"rover", "robot", "vehicle", "perseverance", "buggy", "car"}
_ANTENNA_WORDS = {"antenna", "dish", "tower", "station", "base", "satellite"}


def _canonical_key(phrase: str) -> str:
    words = re.findall(r"[a-z]+", phrase.lower())
    if any(w in _ROVER_WORDS for w in words[-1:]):
        return "rover"
    if any(w in _ANTENNA_WORDS for w in words[-1:]):
        return "antenna"
    return " ".join(words) or phrase


def _zoom_crop(rgb: np.ndarray, u: float, v: float, half_w: int = 80, half_h: int = 60,
               scale: int = 3) -> np.ndarray:
    """Crop a window centred on normalized (u, v) and upscale it, so a
    tiny/distant object is big enough for a small VLM to judge its colour."""
    from PIL import Image

    h, w = rgb.shape[:2]
    cx, cy = int(u * (w - 1)), int(v * (h - 1))
    x0, x1 = max(cx - half_w, 0), min(cx + half_w, w)
    y0, y1 = max(cy - half_h, 0), min(cy + half_h, h)
    crop = Image.fromarray(np.asarray(rgb[y0:y1, x0:x1], dtype=np.uint8))
    crop = crop.resize((crop.width * scale, crop.height * scale), Image.BICUBIC)
    return np.asarray(crop)


@dataclass
class VlmStepResult:
    world_xz: Optional[tuple[float, float]]
    advance: bool
    found: bool
    reasoning: str
    instruction_text: str
    # "point" (default) or "turn" -- see qwen35_prompts.build_point_goal_history_prompt.
    # direction is "left"/"right" and degrees a magnitude in (0, 180] when
    # action == "turn", else None -- a 180 request the parser normalized from
    # a legacy 'back' response is direction="left"/degrees=180. A "turn"
    # result always has advance=False/found=False/world_xz=None --
    # RoverController advances a turn instruction off the physical turn's own
    # completion instead (see set_vlm_turn), never off this dataclass's advance.
    action: str = "point"
    direction: Optional[str] = None
    degrees: Optional[float] = None
    # target_visible/target_world_xz (2026-08-27): qwen35_response_parser has
    # carried target.visible/target_u/target_v since the affordance/target
    # split landed (see build_point_goal_history_prompt's docstring) -- the
    # ONLY honest "do I actually see the named target" signal, distinct from
    # `found` (=affordance.found, "is there a reasonable next driving point",
    # true on almost every tick by design, even a plain "keep going forward"
    # guess). Callers that need to know whether the target itself was really
    # seen this tick (e.g. deciding whether to trust/refresh a remembered
    # bearing) were reading `found`/`world_xz` instead, silently treating
    # ordinary forward-progress guesses as real sightings -- these two fields
    # were parsed all along but never surfaced past this class. Default False/
    # None preserves the old behavior for any caller that doesn't check them.
    target_visible: bool = False
    target_world_xz: Optional[tuple[float, float]] = None
    # action == "forward" only: metres to drive straight ahead. ("stop" and
    # "home" carry no extra fields.)
    distance_m: Optional[float] = None
    # Normalized horizontal image coordinate (0=left, 1=right) of the point
    # the model chose (affordance if found, else the visible target), used to
    # steer toward it when depth can't resolve it to a world point.
    steer_u: Optional[float] = None


class VlmPointNavigator:
    def __init__(
        self,
        instructions: list[str],
        hfov_deg: float,
        replan_every: int = 15,
        num_history: int = 8,
        qwen35_port: Optional[int] = None,
        look_down_frame_fn: Optional[Callable[[], np.ndarray]] = None,
        memory: Optional[ObjectBeliefMemory] = None,
        intervention_sigma: float = 3.5,
        intervention_unseen_polls: int = 2,
        intervention_empty_rounds: int = 2,
    ):
        if not instructions:
            raise ValueError("VlmPointNavigator needs at least one instruction")
        self.instructions = list(instructions)
        self.hfov_deg = float(hfov_deg)
        self.replan_every = max(int(replan_every), 1)
        self.num_history = max(int(num_history), 0)
        self.qwen35_port = qwen35_port
        # Optional callback returning a floor-pitched RGB frame (see
        # run_internvla_n1_compare.py's capture_look_down, the real-world
        # look-down sub-flow InternVLA-N1 uses) -- when set, a found=False
        # response (no ground point visible in the forward-facing frame,
        # e.g. an occluded/close-up view) gets ONE retry against this
        # frame before falling back to the normal creep-forward behavior,
        # instead of always blindly creeping. None (the default) preserves
        # the old behavior exactly for any caller that doesn't pass one.
        self.look_down_frame_fn = look_down_frame_fn
        self.idx = 0
        # Scan-search state (see maybe_step): consecutive polls where the
        # target wasn't verified visible, scan rotations done since it was
        # last seen, and drive-polls spent exploring after a full scan.
        self._unseen_streak = 0
        self._scans = 0
        self._explore_polls = 0
        # Where objects are is remembered ONLY by the belief (real navdp
        # SubgoalBeliefBank, one Gaussian slot per named object): a sighting
        # re-seeds the slot, the rover's own motion dead-reckons it, sigma
        # grows while unseen. Persists across instructions (not reset in
        # advance()) so "go to the rover; ...; go to the rover" recalls it.
        # Pass a shared ObjectBeliefMemory so what was learned in an earlier
        # command is still remembered in the next one.
        self._mem = memory if memory is not None else ObjectBeliefMemory()
        self._pending: dict = {}  # this poll's sightings, folded in once per poll
        # Human-intervention triggers: the target is out of view AND (its
        # belief sigma has reached intervention_sigma, or full scans keep
        # coming back empty) -> needs_human is set and the controller hands
        # the rover to the operator. last_uncertainty feeds the GUI readout.
        self.intervention_sigma = float(intervention_sigma)
        self.intervention_unseen_polls = int(intervention_unseen_polls)
        self.intervention_empty_rounds = int(intervention_empty_rounds)
        self.needs_human: Optional[str] = None
        self.last_uncertainty: Optional[float] = None
        # Bumped every check in which the target is actually seen in the
        # image (not just recalled from memory). The controller re-anchors its
        # current-uncertainty tracker only when this changes.
        self.seen_count = 0
        self._unseen_polls = 0
        self._empty_rounds = 0
        self._blind_travel = 0.0
        self._blind_last = None
        self._arrive_hits = 0
        self._poll_n = 0
        self.sweep_every = 3  # passive landmark sweep cadence, in polls
        self._lost_polls = 0
        # (rgb, prompt, raw_response_text) triples, oldest first -- cleared on
        # advance(), capped at num_history (see maybe_step).
        self._history: list[tuple] = []

    @property
    def current_instruction(self) -> str:
        return self.instructions[self.idx]

    @property
    def finished(self) -> bool:
        return self.idx >= len(self.instructions)

    def maybe_step(self, obs: Observation, step: int) -> Optional[VlmStepResult]:
        """Poll wrapper: passively sweeps for every known landmark kind (so the
        belief learns where EVERYTHING is, not just the current target), runs
        the instruction step, then folds all of this poll's sightings into the
        belief in one update (propagation by ego-motion happens there)."""
        if self.finished or step % self.replan_every != 0:
            return None
        self._pending = {}
        if self._poll_n % self.sweep_every == 0:
            try:
                self._sweep(obs)
            except Exception:
                pass  # sweeping is best-effort; never break navigation
        self._poll_n += 1
        result = self._maybe_step_inner(obs, step)
        self._mem.update(obs.pose, self._pending)
        self._pending = {}
        return result

    def _sweep(self, obs: Observation) -> None:
        """One passive look for every landmark kind. Everything found is
        crop-verified (colour blobs and grounding boxes both hallucinate),
        depth-resolved, and recorded in self._pending for the belief."""
        current = self._mem_key_for(self.current_instruction)
        for c in _FLAG_COLORS:
            if c == current:
                continue
            blob = detect_color_flag(obs.rgb, c, return_mask=True)
            if blob is None:
                continue
            if self._verify_crop(obs, blob[0], blob[1], f"go to the {c} flag"):
                self._note_sighting(obs, c, blob[0], blob[1], mask=blob[3])
        for key, phrase in (("rover", "rover"), ("antenna", "satellite dish antenna")):
            if key == current:
                continue
            box = self._best_box(obs, phrase, f"go to the {key}")
            if box is not None:
                self._note_sighting(obs, key, (box[0] + box[2]) / 2, box[1] + 0.65 * (box[3] - box[1]))

    def _verify_crop(self, obs, u, v, instruction, half=None) -> bool:
        try:
            ok, _seen = qwen35_client.verify_target(
                _zoom_crop(obs.rgb, u, v, *(half or (80, 60))) if half is None
                else _zoom_crop(obs.rgb, u, v, half_w=half, half_h=half,
                                scale=max(1, 336 // (2 * half))),
                instruction, port=self.qwen35_port,
            )
        except Exception:
            return False
        return ok

    def _best_box(self, obs, phrase, instruction):
        """Largest grounding box for `phrase` that passes the crop check."""
        try:
            boxes = qwen35_client.ground(obs.rgb, phrase, port=self.qwen35_port)
        except Exception:
            return None
        h_px, w_px = obs.rgb.shape[:2]
        cands = sorted(
            ((x2 - x1) * (y2 - y1), x1, y1, x2, y2)
            for x1, y1, x2, y2, _l in boxes
            if 4e-5 <= (x2 - x1) * (y2 - y1) <= 0.8
        )[::-1][:3]
        for _a, x1, y1, x2, y2 in cands:
            half = int(max(40, 0.75 * max((x2 - x1) * w_px, (y2 - y1) * h_px)))
            if self._verify_crop(obs, (x1 + x2) / 2, (y1 + y2) / 2, instruction, half=half):
                return (x1, y1, x2, y2)
        return None

    def _blob_world(self, obs, u, v, mask=None):
        """World point under a detected blob. With a colour mask, depth is read
        ONLY from the blob's own pixels, and only trusted when a good share of
        them has a valid reading: a far flag beyond the depth range has none, so
        it must NOT inherit the depth of a nearer boulder standing beside or in
        front of it (live: a flag next to a boulder was "measured" at the
        boulder's 3.8 m and the instruction finished 20 m from the flag).
        None means bearing-only."""
        if mask is None or obs.depth is None:
            return pixel_to_world(obs, u, v, self.hfov_deg)
        from dataclasses import replace

        d = np.asarray(obs.depth)
        valid = mask & np.isfinite(d) & (d > 0.0)
        n_mask = int(mask.sum())
        if int(valid.sum()) < max(4, int(0.3 * n_mask)):
            return None
        depth = np.where(valid, d, 0.0)
        # centre the lookup on the valid pixels, not the whole blob's centroid
        ys, xs = np.nonzero(valid)
        h, w = d.shape[:2]
        uu, vv = float(xs.mean() / (w - 1)), float(ys.mean() / (h - 1))
        return pixel_to_world(replace(obs, depth=depth), uu, vv, self.hfov_deg)

    def _note_sighting(self, obs, key, u, v, mask=None) -> None:
        world = self._blob_world(obs, u, v, mask)
        if world is not None:
            self._pending[key] = (world[0], world[2])

    @staticmethod
    def _mem_key_for(instruction: str) -> Optional[str]:
        """Canonical belief key for an instruction's target: flag colour, or
        rover / antenna by noun group, else the raw target phrase."""
        color = named_flag_color(instruction)
        if color is not None:
            return color
        phrase = target_phrase(instruction)
        if phrase is None:
            return None
        return _canonical_key(phrase)

    def _maybe_step_inner(self, obs: Observation, step: int) -> Optional[VlmStepResult]:
        """Returns None on a non-cadence tick or once finished; otherwise
        queries Qwen3.5-9B with the current frame + current instruction +
        this navigator's own rolling history, and returns the resulting
        point (already backprojected to a world x/z) plus whether the
        current instruction should now advance."""
        if self.finished or step % self.replan_every != 0:
            return None

        instruction = self.current_instruction
        simple = parse_simple_command(instruction)
        if simple is not None:
            return VlmStepResult(
                world_xz=None,
                advance=False,
                found=False,
                reasoning=f"[rule-based {simple['action']}]",
                instruction_text=instruction,
                action=simple["action"],
                direction=simple.get("direction"),
                degrees=simple.get("degrees"),
                distance_m=simple.get("distance_m"),
            )
        color = named_flag_color(instruction) if _SEEK_RE.search(instruction) else None
        if color is not None:
            return self._color_flag_step(obs, instruction, color)
        phrase = target_phrase(instruction)
        if phrase is not None:
            return self._color_flag_step(obs, instruction, "ground:" + phrase)
        query_rgb = obs.rgb
        result, prompt, raw_text = self._query(query_rgb, instruction, step)

        if (
            result["action"] == "point"
            and not result["found"]
            and self.look_down_frame_fn is not None
        ):
            look_down_rgb = self.look_down_frame_fn()
            ld_result, ld_prompt, ld_raw_text = self._query(
                look_down_rgb, instruction, step
            )
            if ld_result["action"] == "point" and ld_result["found"]:
                query_rgb, result, prompt, raw_text = (
                    look_down_rgb,
                    ld_result,
                    ld_prompt,
                    ld_raw_text,
                )

        if raw_text:  # skip history for an unparseable reply (see _query)
            self._history.append((query_rgb, prompt, raw_text))
        if len(self._history) > self.num_history:
            self._history.pop(0)

        if result["action"] == "turn":
            return VlmStepResult(
                world_xz=None,
                advance=False,
                found=False,
                reasoning=result.get("reasoning", ""),
                instruction_text=instruction,
                action="turn",
                direction=result["direction"],
                degrees=result["degrees"],
            )

        if result["action"] in ("forward", "stop", "home"):
            return VlmStepResult(
                world_xz=None,
                advance=False,
                found=False,
                reasoning=result.get("reasoning", ""),
                instruction_text=instruction,
                action=result["action"],
                distance_m=result.get("distance_m"),
            )

        world_xz = None
        if result["found"]:
            world = pixel_to_world(obs, result["u"], result["v"], self.hfov_deg)
            if world is not None:
                world_xz = (world[0], world[2])

        target_visible = bool(result.get("target_visible", False))
        note = ""
        # Verify the claim on a zoomed crop (the 4B model hallucinates
        # "red flag visible" for a blue sliver at the image edge). A claim
        # without coordinates can't be verified, so it doesn't count.
        if target_visible:
            if result.get("target_u") is None or result.get("target_v") is None:
                target_visible = False
                note = "[target claimed without position] "
            else:
                try:
                    match, seen = qwen35_client.verify_target(
                        _zoom_crop(obs.rgb, result["target_u"], result["target_v"]),
                        instruction,
                        port=self.qwen35_port,
                    )
                except Exception as exc:  # server hiccup: don't block driving
                    match, seen = True, f"verify error {exc}"
                if not match:
                    target_visible = False
                    note = f"[verify rejected: saw '{seen}'] "
        # The 4B model often says target.visible=false while its own
        # reasoning + drive point clearly name the target ("the red flag is
        # visible on the left"). For a seek instruction, check the drive
        # point itself on a TALL crop (the point may be on the ground under
        # the flag); if the verifier sees the named target there, accept it
        # instead of scanning away from it.
        if (
            not target_visible
            and result["action"] == "point"
            and result["found"]
            and result.get("u") is not None
            and _SEEK_RE.search(instruction)
            and not note
        ):
            try:
                match, seen = qwen35_client.verify_target(
                    _zoom_crop(obs.rgb, result["u"], result["v"], half_w=110, half_h=110, scale=2),
                    instruction,
                    port=self.qwen35_port,
                )
            except Exception:
                match, seen = False, "verify error"
            if match:
                target_visible = True
                result = dict(result, target_u=result["u"], target_v=result["v"])
                note = f"[verified at drive point: {seen}] "
        target_world_xz = None
        if target_visible:
            target_world = pixel_to_world(obs, result["target_u"], result["target_v"], self.hfov_deg)
            if target_world is not None:
                target_world_xz = (target_world[0], target_world[2])

        seeking = bool(_SEEK_RE.search(instruction))
        mem_key = self._mem_key_for(instruction) or instruction.strip().lower()[:40]
        # Verified AND measured sightings re-seed the belief (folded in once
        # per poll by maybe_step); otherwise it is just dead-reckoned. Later
        # noisy replies can't yank the rover around.
        if target_visible and target_world_xz is not None:
            self._pending[mem_key] = target_world_xz
        if target_visible:
            self.seen_count += 1  # verified sighting: resets current uncertainty
        recalled = self._mem.recall(mem_key, obs.pose)
        if (
            not target_visible
            and recalled is not None
            and result["action"] == "point"
        ):
            # Keep driving to where the belief says the target is.
            locked = recalled["world_xz"]
            dist = math.hypot(locked[0] - obs.pose.x, locked[1] - obs.pose.z)
            return VlmStepResult(
                world_xz=locked,
                advance=dist <= 2.5,
                found=True,
                reasoning=f"{note}[locked target, {dist:.1f} m away]",
                instruction_text=instruction,
                action="point",
                target_visible=dist <= 2.5,
                target_world_xz=locked,
            )
        if target_visible:
            self._unseen_streak = 0
            self._scans = 0
            self._explore_polls = 0
        elif seeking:
            self._unseen_streak += 1
            if self._scans < 8:
                # Target not seen: rotate in place to scan the surroundings
                # instead of driving at an arbitrary guessed point.
                self._scans += 1
                return VlmStepResult(
                    world_xz=None,
                    advance=False,
                    found=False,
                    reasoning=f"{note}scanning for target ({self._scans}/8)",
                    instruction_text=instruction,
                    action="turn",
                    direction="right",
                    degrees=max(min(self.hfov_deg * 0.7, 75.0), 20.0),
                )
            # Full circle without seeing it: explore along the model's own
            # affordance for a few polls, then scan again.
            self._explore_polls += 1
            if self._explore_polls >= 4:
                self._scans = 0
                self._explore_polls = 0

        # Two other advance-decision designs were tried and reverted here,
        # both measured on the real R2R-CE benchmark (n=18 episode-runs,
        # 3 reps x 6 episodes) against this plain single-call baseline
        # (SR=0.556±0.497): (1) a minimum-progress gate + majority-vote
        # confirmation via repeated SAMPLED calls (SR dropped to 0.33,
        # z-test not even needed, the drop was consistent across the
        # specific episode that motivated it); (2) splitting "where to
        # point" (sampled, temperature=0.7) from "is this instruction done"
        # (a separate greedy/do_sample=False call, hypothesizing the
        # sampling noise itself was the variance source) -- SR moved to
        # 0.389±0.487, statistically indistinguishable from this baseline
        # (z~1.0, p~0.32) despite doubling Qwen calls on every advance
        # check. Neither showed a proven benefit, so both were reverted --
        # trust the model's own joint judgment directly here.
        # Arrival needs a verified target AND (when depth resolves it) the
        # target must really be close -- never trust "advance" alone.
        advance = bool(result["advance"]) and target_visible
        if advance and target_world_xz is not None:
            if math.hypot(target_world_xz[0] - obs.pose.x, target_world_xz[1] - obs.pose.z) > ARRIVE_MAX_DIST_M:
                advance = False
                note += "[too far to be reached] "
        # Measured arrival: a verified, depth-resolved target within 2.5 m is
        # reached regardless of what the model says.
        if (
            target_visible
            and target_world_xz is not None
            and math.hypot(target_world_xz[0] - obs.pose.x, target_world_xz[1] - obs.pose.z) <= 2.5
        ):
            advance = True
        # Drive to the verified target itself when depth resolves it.
        if target_visible and target_world_xz is not None:
            world_xz = target_world_xz
        return VlmStepResult(
            world_xz=world_xz,
            advance=advance,
            found=bool(result["found"]) or world_xz is not None,
            reasoning=note + result.get("reasoning", ""),
            instruction_text=instruction,
            action="point",
            target_visible=target_visible,
            target_world_xz=target_world_xz,
            steer_u=(
                result.get("target_u")
                if target_visible
                else (result["u"] if result["found"] else None)
            ),
        )

    def _color_flag_step(self, obs: Observation, instruction: str, color: str) -> VlmStepResult:
        """'go to the <colour> flag': find the flag by colour (no VLM call),
        lock its measured world position, drive to it, finish within 2.5 m.
        Not seen and never measured: scan in place, then drive 4 m and
        rescan."""
        label = color
        if color.startswith("ground:"):
            label = color[7:]
            blob = None
            try:
                boxes = qwen35_client.ground(obs.rgb, label, port=self.qwen35_port)
            except Exception:
                boxes = []
            best = None
            cands = []
            for x1, y1, x2, y2, _lab in boxes:
                area = (x2 - x1) * (y2 - y1)
                if area < 4e-5 or area > 0.8:  # speck or whole image
                    continue
                cands.append((area, x1, y1, x2, y2))
            # Grounding always returns *something* (live: the rover's mast and
            # a frame-edge speck came back as "antenna"), so crop-verify each
            # box is the right kind of object, largest first.
            h_px, w_px = obs.rgb.shape[:2]
            for cand in sorted(cands, reverse=True)[:3]:
                _a, x1, y1, x2, y2 = cand
                bw, bh = (x2 - x1) * w_px, (y2 - y1) * h_px
                half = int(max(40, 0.75 * max(bw, bh)))
                try:
                    ok, _seen = qwen35_client.verify_target(
                        _zoom_crop(obs.rgb, (x1 + x2) / 2, (y1 + y2) / 2,
                                   half_w=half, half_h=half,
                                   scale=max(1, 336 // (2 * half))),
                        instruction, port=self.qwen35_port,
                    )
                except Exception:
                    ok = True  # verifier unavailable -> trust grounding
                if ok:
                    best = cand
                    break
            if best is not None:
                _a, x1, y1, x2, y2 = best
                # aim at the lower-middle of the box (where it meets the
                # ground / has valid depth), not its top
                blob = ((x1 + x2) / 2.0, y1 + 0.65 * (y2 - y1), int(best[0] * 1e5))
        elif color == "grey":
            blob = None
            for cu, cv, cn in grey_flag_candidates(obs.rgb):
                try:
                    ok, _seen = qwen35_client.verify_target(
                        _zoom_crop(obs.rgb, cu, cv), instruction, port=self.qwen35_port
                    )
                except Exception:
                    ok = False
                if ok:
                    blob = (cu, cv, cn)
                    break
        else:
            blob = detect_color_flag(obs.rgb, color, return_mask=True)
            if blob is not None:
                # A colour match alone is not a flag (live: red parts of the
                # Perseverance rover model were taken for the red flag and the
                # instruction "arrived" instantly) -- crop-verify the kind.
                try:
                    ok, _seen = qwen35_client.verify_target(
                        _zoom_crop(obs.rgb, blob[0], blob[1]), instruction,
                        port=self.qwen35_port,
                    )
                except Exception:
                    ok = True  # verifier unavailable -> trust the colour match
                if not ok:
                    blob = None
        world_xz = None
        if blob is not None:
            u, v, npx = blob[:3]
            self._lost_polls = 0
            world = self._blob_world(obs, u, v, blob[3] if len(blob) > 3 else None)
            if world is not None:
                world_xz = (world[0], world[2])
            self._scans = 0
            note = f"[detect {label}: u={u:.2f}] "
        else:
            u = None
            note = f"[no {label} in view] "
        # The belief is the memory: a depth-resolved sighting re-seeds it
        # (latest measurement wins); otherwise it is dead-reckoned by ego-motion.
        key = _canonical_key(label)
        if world_xz is not None:
            self._pending[key] = world_xz
        elif blob is not None:
            # Seen but no depth (too far): remember it as a point along its
            # bearing at an assumed range with a large sigma. Real depth
            # readings later pull it in (inverse-variance fusion / gate).
            brg = obs.pose.yaw - math.radians((blob[0] - 0.5) * self.hfov_deg)
            R = BEARING_ONLY_RANGE_M
            self._pending[key] = (
                obs.pose.x + R * -math.sin(brg),
                obs.pose.z + R * -math.cos(brg),
                BEARING_ONLY_SIGMA_M,
            )
        recalled = self._mem.recall(key, obs.pose)
        if (
            blob is None
            and recalled is not None
            and recalled["distance"] < 4.0
            and abs(recalled["bearing"]) < math.radians(self.hfov_deg * 0.35)
        ):
            # The belief says it is plainly in view, yet it was not seen:
            # negative evidence. After a few polls the belief is cleared.
            if self._mem.note_absent(key):
                note += "[belief cleared: not where remembered] "
                recalled = None
        if (world_xz is not None and recalled is not None
                and recalled["sigma"] < BEARING_ONLY_SIGMA_M - 1e-3):
            # A depth reading far from a confident memory is more likely a bad
            # reading (glass, wrong box) than a jump of the object: drive to
            # the memory this check. The reading is still passed to the memory
            # as a candidate; a second agreeing reading moves it.
            gate = 1.0 + 3.0 * recalled["sigma"]
            rx, rz = recalled["world_xz"]
            if (math.hypot(world_xz[0] - rx, world_xz[1] - rz) > gate
                    and not self._mem.candidate_agrees(key, world_xz)):
                note += "[depth outlier ignored, kept memory] "
                world_xz = None
        if blob is not None:
            self.seen_count += 1
            self._unseen_polls = 0
            self._empty_rounds = 0
        else:
            self._unseen_polls += 1
        # Target uncertainty (m): the belief's sigma, plus a staleness term that
        # grows every poll the target stays out of view. Unknown (never
        # located) reads as UNKNOWN_UNCERTAINTY.
        if recalled is None:
            self.last_uncertainty = UNKNOWN_UNCERTAINTY
        else:
            stale = 0.0 if blob is not None else STALE_M_PER_POLL * self._unseen_polls
            self.last_uncertainty = recalled["sigma"] + stale
        # Range-unknown target (seen only by direction, beyond depth range): its
        # distance is an assumed guess. Count how far the rover drives toward it;
        # a real flag would have come within depth range by now.
        range_unknown = (recalled is not None and world_xz is None
                         and recalled["sigma"] >= BEARING_ONLY_SIGMA_M - 1e-3)
        if range_unknown:
            if self._blind_last is not None:
                self._blind_travel += math.hypot(obs.pose.x - self._blind_last[0],
                                                 obs.pose.z - self._blind_last[1])
            self._blind_last = (obs.pose.x, obs.pose.z)
        else:
            self._blind_travel, self._blind_last = 0.0, None
        # Over the threshold -> stop and hand the rover to the human, whether or
        # not the target is currently in view.
        if recalled is None:
            if blob is None and self._scans >= 6:
                # Never located: one full 360-degree look-around found nothing.
                self.needs_human = f"target location unknown: {label} not found"
        elif range_unknown and self._blind_travel >= BLIND_TRAVEL_HANDOVER_M:
            self.needs_human = f"{label} still beyond depth range after {self._blind_travel:.0f} m"
        elif (blob is None and recalled["distance"] <= ARRIVE_M
              and self._unseen_polls >= ARRIVE_MISSING_POLLS):
            # Drove to the remembered place, but the object is not there.
            self.needs_human = f"{label} not at the remembered place"
        elif self.last_uncertainty >= self.intervention_sigma:
            self.needs_human = (
                f"target uncertainty {self.last_uncertainty:.1f} m is over the "
                f"{self.intervention_sigma:.1f} m limit / target "
                f"{'range unknown (beyond depth range)' if recalled['sigma'] >= BEARING_ONLY_SIGMA_M - 1e-3 else str(round(recalled['distance'])) + ' m away'} ({label} "
                f"{'seen only by bearing' if blob is not None else 'out of view'})"
            )
        if blob is not None and world_xz is None and recalled is not None:
            note += f"[belief] "
        if (blob is None and world_xz is None and recalled is not None
                and abs(recalled["bearing"]) > math.radians(max(self.hfov_deg * 0.5, REMEMBERED_TURN_DEG))):
            # Remembered target is outside the camera view (e.g. behind):
            # NavDP only plans forward inside the image, so turn to face the
            # remembered bearing first, then drive (next poll).
            delta = recalled["bearing"]  # +left, body frame
            deg = abs(math.degrees(delta))
            return VlmStepResult(
                world_xz=None, advance=False, found=False,
                reasoning=f"{note}remembered {label} {recalled['distance']:.1f} m, "
                          f"{deg:.0f} deg {'left' if delta > 0 else 'right'}: turning toward it",
                instruction_text=instruction, action="turn",
                direction="left" if delta > 0 else "right",
                degrees=min(deg, 180.0),
            )
        target = world_xz or (recalled["world_xz"] if recalled else None)
        if target is not None:
            dist = math.hypot(target[0] - obs.pose.x, target[1] - obs.pose.z)
            drive = target
            if key in ("rover", "antenna") and dist > STANDOFF_M:
                # large object: aim short of its measured near-surface point
                k = (dist - STANDOFF_M) / dist
                drive = (obs.pose.x + (target[0] - obs.pose.x) * k,
                         obs.pose.z + (target[1] - obs.pose.z) * k)
            # arrival only counts on a trustworthy position, not a bearing guess
            trusted = world_xz is not None or (recalled is not None and recalled["sigma"] < 1.0)
            # Only a measured / well-known position counts as "the target" for
            # arrival and the controller's stuck-forced finish. A bearing-only
            # guess (live: a flag past the 10 m depth range, a point 9 m along
            # its bearing) must never end the instruction as "reached".
            # "Reached" needs the object SEEN this poll: arriving at a
            # remembered spot alone is not arrival (the object may have moved
            # or the dead-reckoned memory drifted) -- see ARRIVE_M above.
            seen_now = blob is not None
            # Two checks in a row must agree that it is seen and close: one
            # bad depth reading alone (live: a door measured at 2.5 m from far
            # away) must not end the instruction as "reached".
            close_now = dist <= ARRIVE_M and trusted and seen_now
            self._arrive_hits = self._arrive_hits + 1 if close_now else 0
            return VlmStepResult(
                world_xz=drive,
                advance=self._arrive_hits >= ARRIVE_CONFIRM_POLLS,
                found=True,
                reasoning=(f"{note}{'driving to' if seen_now else 'driving to remembered'} {label}, "
                           + ("range unknown" if recalled is not None and world_xz is None
                              and recalled["sigma"] >= BEARING_ONLY_SIGMA_M - 1e-3 else f"{dist:.1f} m")),
                instruction_text=instruction,
                action="point",
                target_visible=trusted and seen_now,
                target_world_xz=target if trusted else None,
                steer_u=u,
            )
        if blob is not None:
            # Seen but depth invalid (tiny/far): creep toward its bearing;
            # the controller does this when found=False and steer_u is set.
            return VlmStepResult(
                world_xz=None, advance=False, found=False,
                reasoning=f"{note}depth unresolved, creeping toward it",
                instruction_text=instruction, action="point",
                target_visible=False, steer_u=u,
            )
        # Seen recently but now out of view: turn back toward where it was
        # (a few polls), instead of the blind scan below.
        if recalled is not None and self._lost_polls < 6:
            self._lost_polls += 1
            delta = recalled["bearing"]  # belief bearing, +left, body frame
            deg = abs(math.degrees(delta))
            if deg > 8.0:
                return VlmStepResult(
                    world_xz=None, advance=False, found=False,
                    reasoning=f"{note}flag lost; turning back {deg:.0f} deg toward last bearing",
                    instruction_text=instruction, action="turn",
                    direction="left" if delta > 0 else "right",
                    degrees=min(deg, 120.0),
                )
            return VlmStepResult(  # facing it but hidden: creep ahead
                world_xz=None, advance=False, found=False,
                reasoning=f"{note}flag hidden; creeping toward last bearing",
                instruction_text=instruction, action="point",
                target_visible=False, steer_u=0.5,
            )
        if self._scans < 6:
            self._scans += 1
            return VlmStepResult(
                world_xz=None, advance=False, found=False,
                reasoning=f"{note}scanning ({self._scans}/6)",
                instruction_text=instruction, action="turn", direction="right",
                degrees=max(min(self.hfov_deg * 0.7, 75.0), 20.0),
            )
        self._scans = 0
        self._empty_rounds += 1
        return VlmStepResult(
            world_xz=None, advance=False, found=False,
            reasoning=f"{note}full scan empty, moving 4 m to look elsewhere",
            instruction_text=instruction, action="forward", distance_m=4.0,
        )

    def _query(self, rgb, instruction: str, step: int):
        """select_point_goal with one retry on a malformed model reply
        (ValueError from the parser). Live bug: an uncaught parse error
        ("Missing 'affordance'") propagated up and killed the whole
        controller thread. After two bad replies, return a harmless
        'no point found' so the rover just creeps/scans and asks again."""
        for _ in range(2):
            try:
                return qwen35_client.select_point_goal(
                    rgb, instruction, step, self._history, port=self.qwen35_port
                )
            except ValueError:
                continue
        return (
            {
                "action": "point", "found": False, "u": None, "v": None,
                "target_visible": False, "target_u": None, "target_v": None,
                "advance": False, "reasoning": "[unparseable model reply]",
            },
            "",
            "",
        )

    def advance(self) -> bool:
        """Move to the next queued instruction and reset history (mirrors
        InternVLA-N1's agent.reset() on a new instruction). Returns
        self.finished after advancing."""
        self.idx += 1
        self._history = []
        self._unseen_streak = self._scans = self._explore_polls = 0
        self._lost_polls = 0  # NOT the belief memory -- that persists
        self.needs_human = None
        self._unseen_polls = self._empty_rounds = 0
        self._blind_travel, self._blind_last = 0.0, None
        self._arrive_hits = 0
        return self.finished

    def observe_while_paused(self, obs: Observation, step: int) -> None:
        """While a human drives (autonomy paused), keep the target belief and
        its uncertainty live: look for the current target, fold any sighting
        into the belief, propagate by ego-motion -- but choose no action.
        Called at half the normal poll rate (each look is a blocking VLM call)."""
        if self.finished or step % (self.replan_every * 2) != 0:
            return
        instruction = self.current_instruction
        if parse_simple_command(instruction) is not None:
            return
        color = named_flag_color(instruction) if _SEEK_RE.search(instruction) else None
        if color is None:
            phrase = target_phrase(instruction)
            if phrase is None:
                return
            color = "ground:" + phrase
        self._pending = {}
        self._color_flag_step(obs, instruction, color)  # result discarded
        self._mem.update(obs.pose, self._pending)
        self._pending = {}
        self.needs_human = None

    def resume_after_human(self) -> None:
        """The operator drove and handed control back: forget the 'lost'
        counters (the belief keeps what it knows) and carry on."""
        self.needs_human = None
        self._unseen_polls = self._empty_rounds = self._scans = self._lost_polls = 0
        self._blind_travel, self._blind_last = 0.0, None
        self._arrive_hits = 0
