"""
All externally-tunable knobs for vl_direction, per next.md sec 8. Zero
imports so this module is safe to import from both sides of the future
client/server split (internvl_server.py runs in a separate, heavy-dependency
conda env; everything else in this package runs in the caller's env) --
mirrors sam_vla/vlm/qwen_config.py's role exactly.
"""

# --- InternVL backend selection ---
# "mock" returns canned text with no I/O (default -- keeps this module
# importable/testable without a live model or network). "hf"/"vllm"/"api"
# all route through InternVLSocketClient; the backend value only labels
# which serving stack internvl_server actually runs.
#
# The real checkpoint lives in the "vl" conda env (torch 2.11+cu128,
# transformers 5.x) -- see internvl_model_runner.py / internvl_server.py.
# INTERNVL_MODEL_PATH is InternVL3-8B: picked as the balanced choice for
# this task's short LEFT/RIGHT/FRONT/BACK + sweep-description calls -- good
# spatial/grounding accuracy for its size, and its ~16GB bf16 footprint
# leaves this GPU's 98GB with plenty of headroom for multi-frame bursts.
INTERNVL_BACKEND = "hf"
INTERNVL_MODEL_PATH = "OpenGVLab/InternVL3-8B"

INTERNVL_SERVER_HOST = "127.0.0.1"
INTERNVL_SERVER_PORT = 8766  # distinct from qwen_server's 8765

# --- Qwen backend (alternative to InternVL above, see qwen_model_runner.py /
# qwen_server.py) -- same generate(frames, prompt, max_new_tokens) protocol,
# just backed by Qwen2.5-VL-3B-Instruct via the "qwen_vlm" conda env instead
# of InternVL3-8B via "vl". Port is distinct from both InternVL's 8766 and
# sam_vla/vlm/qwen_server's 8765 so all three can run at once if needed. ---
QWEN_MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_SERVER_HOST = "127.0.0.1"
QWEN_SERVER_PORT = 8767

# --- per-mode generation knobs ---
MAX_NEW_TOKENS = {"cbf": 16, "exploration": 16, "uncertainty": 32, "ghost_mask": 96}
FRAME_BURST_SIZE = {
    "cbf": 1,
    "exploration": 3,
    "uncertainty": 1,
    "ghost_mask": 1,
}  # exploration configurable 1-5

# --- uncertainty sub-flow ---
# This module never compares covariance against this threshold itself -- the
# caller decides when to enter uncertainty mode and hands the crossed value
# in via UncertaintyContext.threshold_used, purely for HCI logging.
DEFAULT_COVARIANCE_THRESHOLD = 0.25
DEFAULT_MAX_UNITS = 5.0
UNCERTAINTY_ROVER_FRONT_REFERENCE_DEG = 0.0

# --- parse discipline ---
PARSE_RETRY_LIMIT = 1  # one corrective reprompt before parse_ok=False

# --- HCI / intervention ---
# Read by the ORCHESTRATOR, not enforced here: nothing in this module
# branches on it. "Shadow mode" (calling the VL module during a human
# intervened segment without acting on it) is entirely the caller's loop.
SHADOW_LOGGING_ENABLED = False
TELEOP_RESUME_AUTONOMY_TIMEOUT_S = 5.0

# --- logging ---
DEFAULT_LOG_ROOT = "vl_direction_logs"
