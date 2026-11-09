# Module: `vl_direction` — InternVL-based Directional Guidance

## 0. Purpose & Scope

A **standalone, agnostic** module that consumes camera frames (+ minimal side-channel state) and
produces **discrete directional tokens** using InternVL as a domain-specialized VLM. It does not
plan paths, does not fuse beliefs, does not touch NavDP, and does not know anything about how the
rest of `sam_vla` is structured internally. It receives a small, explicit input contract and returns
a small, explicit output contract. Nothing else.

This module exists to support an HCI study: **VL-only autonomy vs. human-in-the-loop intervention**,
measured by success rate and efficiency (time/steps to goal) under each condition. So logging and
mode-switching (autonomous vs. teleop-intervened) are first-class citizens of the design, not an
afterthought.

### Explicit non-goals

- No belief/state estimation logic lives here (the module _consumes_ an uncertainty scalar/covariance
  it's handed, it does not compute one).
- No NavDP, no SAM2, no CBF math. This module only **emits the direction token** that a CBF cone
  (or teleop) elsewhere consumes.
- No knowledge of upstream/downstream module internals. Integration is via one function call in,
  one structured object out.

### Hard interface boundary

Everything upstream of this module (frame capture, obstacle bbox detection, covariance computation,
teleop input handling) and everything downstream (CBF cone steering, actuation, goal recalibration)
stays exactly as it is. This module is a pure function: `(frames, context) -> VLDirectiveResult`.

---

## 1. The Three Query Modes

The module supports exactly three prompt _types_, selected by the caller (not inferred internally —
mode selection is the orchestrator's job, per your stated flow: "if near obstacle → CBF prompt, if
not → exploration prompt, if goal identified → dormant"). Each mode has its own prompt template,
schema, and validity rules.

### 1.1 `CBF` — Obstacle-relative avoidance direction

- **Trigger (caller-side):** an obstacle bbox is present / rover is within a proximity threshold.
- **Input:** 1 frame (or short burst, configurable), obstacle bbox `[x1, y1, x2, y2]` in pixel space,
  frame resolution (for normalization).
- **Ask:** "Given the obstacle at this bbox, should the rover pass on the left or the right?"
- **Output alphabet:** `{LEFT, RIGHT}` — binary, no FRONT/BACK (per your spec — this is a go-around
  decision, not an exploration decision).
- **Consumer:** CBF cone generator picks the cone bias side from this token.

### 1.2 `EXPLORATION` — High-level + vague-correction directional bias

- **Trigger (caller-side):** no obstacle in the near field; rover is in open exploration; either (a)
  running the high-level task autonomously, or (b) a human just issued a vague corrective nudge
  ("this area's explored", "go left", "try somewhere else").
- **Input:** 1–N frames (configurable burst, e.g. a short pano-sweep or last-K frames), the original
  high-level task string, optionally the vague human hint string (nullable).
- **Ask:** "Given the scene and the task (and the hint, if present), which direction should
  exploration continue: left, right, front, or back?"
- **Output alphabet:** `{LEFT, RIGHT, FRONT, BACK}`.
- **Consumer:** exploration policy / frontier picker elsewhere uses this as a directional prior.

### 1.3 `UNCERTAINTY` — High-covariance stop-and-ask

- **Trigger (caller-side):** goal-position covariance (handed in, not computed here) exceeds a
  threshold set by the caller.
- **Behavior is different from the other two** — this mode is not "ask once, get a token." It's a
  **stateful sub-interaction**:
  1. Module signals `NEEDS_HUMAN_INPUT` with the rover's current heading reference (0° = rover-front)
     so the UI can render a compass/angle picker.
  2. Rover (externally, not by this module) does a look-around sweep.
  3. Human supplies an angle (`35`) or angle range (`70-80`) relative to rover-front.
  4. Module hands that angle/range downstream as a directive: "traverse heading θ (± range) for up
     to N units, or until goal is visually confirmed."
  5. If the _downstream_ traversal reports goal-not-found after N units, module is re-invoked
     (rotate + re-ask) — this is a loop the **orchestrator** drives, not an internal loop in this
     module (keeps the module stateless/pure per call — see §4).
  6. If goal is found, this module is done for this episode; recalibration of μ happens _outside_
     this module (belief system is explicitly out of scope).
- **Output alphabet:** this mode does not emit LEFT/RIGHT/FRONT/BACK. It emits a distinct schema:
  `{status: NEEDS_HUMAN_INPUT | HEADING_DIRECTIVE, heading_deg / heading_range_deg}`.
- **Identity token:** `uncertainty` — always separately identifiable from `cbf`/`exploration` tokens
  so the UI/logger can special-case it (per your requirement: "only prompt a separately identifiable
  prompt when uncertainty grows too much").

---

## 2. Output Contract (all modes)

Every call returns one `VLDirectiveResult`. Direction-emitting modes (`cbf`, `exploration`) always
conform to the same envelope so downstream consumers don't need mode-specific parsing beyond reading
`mode` + `direction`.

```jsonc
{
  "identity_token": "cbf" | "exploration" | "uncertainty",   // which question type this answers
  "direction": "LEFT" | "RIGHT" | "FRONT" | "BACK" | null,    // null only for uncertainty mode
  "confidence": 0.0-1.0,           // model-reported or logit-derived, see §5.2
  "raw_response": "...",           // verbatim model output, always logged, never parsed loosely
  "parse_ok": true | false,        // did structured parse succeed, or did fallback trigger
  "uncertainty_payload": {         // present only when identity_token == "uncertainty"
    "status": "NEEDS_HUMAN_INPUT" | "HEADING_DIRECTIVE",
    "rover_front_reference_deg": 0,
    "heading_deg": null,           // filled once human responds
    "heading_range_deg": null,     // e.g. [70, 80], alt to heading_deg
    "max_units": null              // traversal budget, set by caller/human, passed through
  },
  "latency_ms": 0,
  "frame_count": 1,
  "timestamp": "ISO8601",
  "episode_id": "...",
  "call_id": "..."                 // unique per invocation, for HCI logging joins
}
```

Design intent: **direction is always a closed-vocabulary enum, never free text**, per your requirement
("output explicitly in directions only"). Free text only ever lives in `raw_response` for audit/debug.

---

## 3. Module Structure

```
vl_direction/
├── __init__.py
├── config.py              # thresholds, model endpoint, timeouts, enum defs — all external
├── schemas.py             # VLDirectiveResult, mode enums, direction enums (pydantic/dataclass)
├── prompts/
│   ├── cbf_prompt.py         # template + few-shot exemplars for obstacle L/R
│   ├── exploration_prompt.py # template for high-level task + optional vague hint
│   └── uncertainty_prompt.py # template for heading-sweep description → structured ask
├── client.py               # thin InternVL inference wrapper (local HF / vLLM / API — swappable)
├── parser.py                # strict output parser: regex/schema-constrained extraction + fallback
├── directive_engine.py     # the ONE public entrypoint: query(mode, frames, context) -> VLDirectiveResult
├── uncertainty_session.py  # small stateful helper for the multi-turn heading-ask loop (see §4)
├── logging/
│   ├── episode_logger.py   # per-call structured log (JSONL), keyed by episode_id + call_id
│   └── hci_metrics.py      # aggregation: success rate, steps-to-goal, intervention count/type
├── intervention/
│   ├── mode_flag.py        # AUTONOMOUS vs HUMAN_INTERVENED session flag + toggling API
│   └── teleop_bridge.py    # adapter that accepts external keyboard-teleop events, tags them
└── tests/
    ├── test_cbf_prompt.py
    ├── test_exploration_prompt.py
    ├── test_uncertainty_flow.py
    └── test_parser_fallback.py
```

### 3.1 Single public entrypoint

```python
def query(
    mode: Literal["cbf", "exploration", "uncertainty"],
    frames: list[Frame],
    context: CBFContext | ExplorationContext | UncertaintyContext,
    episode_id: str,
) -> VLDirectiveResult:
    ...
```

This is the **only** function the rest of `sam_vla` (or any orchestrator) ever calls. Everything else
in the package is an implementation detail. This is what makes the module swappable/agnostic — the
orchestrator doesn't need to know InternVL is behind it; a future model swap only touches `client.py`
and `prompts/`.

### 3.2 Context objects (per mode, keeps `query()` signature stable)

```python
@dataclass
class CBFContext:
    bbox_xyxy: tuple[int, int, int, int]
    frame_wh: tuple[int, int]

@dataclass
class ExplorationContext:
    task_str: str
    vague_hint: str | None = None

@dataclass
class UncertaintyContext:
    covariance_value: float           # passed in, not computed
    threshold_used: float             # logged for HCI analysis
    rover_front_reference_deg: float = 0.0
    human_heading_response: HeadingResponse | None = None  # filled on second call in the loop
```

---

## 4. The Uncertainty Sub-Flow (stateless module, stateful orchestration)

To keep `vl_direction` itself a pure function (important for testability and for the ablation study —
you want identical inputs to give identical/comparable outputs), the multi-turn "sweep → ask → maybe
retry" loop is **not** internal state in this module. `uncertainty_session.py` provides a small
**session helper class** that the orchestrator drives explicitly:

```python
session = UncertaintySession(episode_id=..., covariance_threshold=...)

# Step 1: covariance too high upstream → orchestrator calls:
result = session.request_human_heading(current_frame, rover_front_reference_deg=0)
# → VLDirectiveResult(identity_token="uncertainty", uncertainty_payload={status: NEEDS_HUMAN_INPUT, ...})

# Step 2: UI collects human input (angle or range), orchestrator calls:
result = session.submit_heading(angle_deg=35)  # or angle_range_deg=(70, 80)
# → VLDirectiveResult(..., uncertainty_payload={status: HEADING_DIRECTIVE, heading_deg: 35, max_units: N})

# Step 3 (external): rover traverses under CBF/teleop toward heading, up to max_units or until goal seen.
# Step 4a: goal found → orchestrator ends session, hands recalibration to belief system (out of scope here).
# Step 4b: goal not found after max_units → orchestrator calls session.retry(new_frame) which re-triggers
#          NEEDS_HUMAN_INPUT with an incremented `attempt` counter (logged, for HCI: # of retries needed).
```

This keeps `vl_direction` module-pure per call while still giving you the "rotate and prompt again"
behavior — the _loop_ lives in orchestration code that's explicitly outside this module's boundary
(as it should be, since retry/rotate is a locomotion concern).

---

## 5. InternVL Integration Details

### 5.1 Client abstraction (`client.py`)

Wrap InternVL behind a minimal interface so backend (local HF checkpoint, vLLM server, hosted API)
is swappable without touching prompts/parser:

```python
class InternVLClient(Protocol):
    def generate(self, frames: list[Frame], prompt: str, max_new_tokens: int) -> str: ...
```

### 5.2 Constrained decoding / output discipline

Since direction must be a closed vocabulary, don't rely on free-form generation + regex alone:

- Prefer **grammar-constrained decoding** if the serving stack supports it (e.g. vLLM guided
  decoding / outlines) so the model can only emit one of the valid tokens for that mode.
- If constrained decoding isn't available, fall back to: prompt with explicit output format
  instruction + few-shot exemplars → strict parser (`parser.py`) → if parse fails, one retry with a
  "you must answer with exactly one of: LEFT, RIGHT" corrective reprompt → if that fails, module
  returns `parse_ok: false` and a caller-defined default (e.g. hold position, or hand off to
  intervention) rather than guessing.
- `confidence` is either the model's own stated confidence (if you prompt for it) or derived from
  token logprobs of the decoded direction token, if the serving stack exposes logprobs. Log both if
  available — useful for the HCI writeup (does low VL confidence correlate with when humans had to
  intervene?).

### 5.3 Prompt templates — structure, not literal text

Each `prompts/*.py` should produce a prompt with:

1. Fixed role/system framing ("You are a directional assistant for a Mars rover...").
2. Mode-specific task description.
3. Explicit output format constraint ("Respond with exactly one word: LEFT or RIGHT.").
4. A couple of few-shot exemplars (image description + correct answer) to anchor format.
5. Injected dynamic content (bbox, task string, hint, frame count) via templated slots — keep this
   in `config.py` or the prompt file as string templates, not hardcoded per-call.

---

## 6. Intervention & Mode Tracking (for the HCI study)

### 6.1 Session mode flag

```python
class SessionMode(Enum):
    AUTONOMOUS = "autonomous"        # VL directives only
    HUMAN_INTERVENED = "human_intervened"  # teleop override active for this segment
```

Every `VLDirectiveResult` gets tagged with the `SessionMode` active at call time (`mode_flag.py`
exposes `get_current_mode()` / `set_mode()`, toggled by the teleop bridge when a keyboard input
arrives).

### 6.2 Teleop bridge (`teleop_bridge.py`)

Thin adapter — **does not implement teleop itself** (that's external/upstream, per your setup). It
just:

- Accepts teleop events from whatever your existing keyboard-teleop system emits.
- On a teleop event arriving during what would've been a VL-driven segment, flips `SessionMode` to
  `HUMAN_INTERVENED` and logs the override (what direction VL _would have_ said, if you want that —
  optional "shadow mode" logging for direct comparison, see §6.4).
- Flips back to `AUTONOMOUS` after N seconds of no teleop input / on explicit "resume autonomy" event.

### 6.3 Metrics (`hci_metrics.py`)

Aggregates over an episode / batch of episodes, split by `SessionMode`:

- Success rate (goal reached) — autonomous-only episodes vs. intervened episodes.
- Steps/time-to-goal, split the same way.
- Intervention count & type per episode (how many CBF vs exploration vs uncertainty interventions).
- Uncertainty-mode trigger count and resolution (avg retries before goal found).
- Optional: VL confidence vs. human-override correlation (if shadow logging enabled).

### 6.4 Optional "shadow mode"

If you want a cleaner ablation signal: even during `HUMAN_INTERVENED` segments, still _call_ the VL
module (but don't act on it) and log what it would have said next to what the human actually chose.
This gives you a direct per-step VL-vs-human agreement metric, not just end-to-end success rate. Flag
this as `config.SHADOW_LOGGING_ENABLED` — off by default (extra inference cost), on for the study runs.

---

## 7. Logging (`episode_logger.py`)

One JSONL line per `query()` call, schema = `VLDirectiveResult` flattened + `session_mode`. This is
intentionally the same _style_ of logging you already use elsewhere in `sam_vla` (per-run JSON/JSONL)
but this logger is self-contained inside `vl_direction/` — it doesn't import or depend on the existing
`EpisodeLogger`, to keep the module agnostic/standalone. If you later want to merge log streams, that's
a join on `episode_id` + timestamp done outside this module, not a dependency from this module outward.

---

## 8. Config surface (`config.py`)

All externally-tunable, nothing hardcoded in logic files:

- `InternVL` model path/endpoint, backend type (hf/vllm/api).
- Per-mode `max_new_tokens`.
- Frame burst size per mode (e.g. CBF=1, exploration=1–5, uncertainty sweep=configurable).
- Covariance threshold (though the _computation_ of covariance is out of scope, the _threshold value_
  used to decide "ask uncertainty mode" is a config constant the caller reads — this module just
  receives whatever value crossed it, for logging).
- `max_units` default for uncertainty-directive traversal.
- Retry/reprompt limits for parse failures.
- `SHADOW_LOGGING_ENABLED` toggle.

---

## 9. Testing Strategy

- **Prompt unit tests:** given a fixed frame + bbox, assert the templated prompt string contains
  required slots correctly filled (no model call needed).
- **Parser tests:** feed a battery of realistic/messy raw model outputs ("I think it should go
  LEFT.", "left", "LEFT.", "Right side seems safer") and assert correct enum extraction + correct
  `parse_ok=False` on genuinely unparseable input.
- **Contract tests:** mock `InternVLClient.generate` to return canned strings, assert `query()`
  returns well-formed `VLDirectiveResult` for all three modes.
- **Uncertainty session tests:** simulate the request → submit → retry loop, assert attempt counter
  increments and payload shape is correct at each stage.
- These tests deliberately never require a live InternVL checkpoint — keeps CI fast and keeps the
  module's public contract the thing under test, not model quality (model quality is a separate,
  qualitative/study-level concern for your HCI paper, not a unit-test concern).

---

## 10. Integration Point (the only place this touches the rest of `sam_vla`)

One call site, wherever your orchestration loop currently decides "CBF avoidance / explore / stay
dormant." That decision logic (near-obstacle? goal-found? vague-hint received?) is **not** part of
this module — it's the caller's job to pick `mode` and build the right `Context`. This module answers
exactly one question per call: _given this mode and this context, what's the direction (or heading
request)?_ — nothing upstream or downstream of that changes.
