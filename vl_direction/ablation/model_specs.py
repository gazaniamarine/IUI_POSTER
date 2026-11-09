"""
Candidate checkpoints for the vl_direction model ablation (run_ablation.py).
Each spec names a HF repo id, which server family loads it, and a per-model
server-startup timeout -- larger/uncached checkpoints take longer to
download on first use and to load into GPU memory, so the timeout scales
roughly with parameter count rather than using one fixed value for all.
"""

from dataclasses import dataclass


@dataclass
class ModelSpec:
    name: str
    kind: str  # "internvl" (model.chat(), dynamic tiling) or "qwen" (chat-template, Qwen2_5_VLForConditionalGeneration)
    model_path: str
    startup_timeout_s: float


# Timeouts sized off a measured ~15MB/s real download throughput on this
# machine (an earlier curl test that suggested ~2KB/s was just measuring an
# unfollowed HF redirect response, not real bandwidth) -- e.g. InternVL3-14B's
# ~28GB of bf16 weights alone takes ~30 minutes to pull on first use, before
# any load time on top. These are first-run (cold cache) timeouts; a second
# run of the same model is far faster since the weights are already cached.
MODEL_SPECS = [
    # Baseline already in use by kb_teleop_vl.py / vl_direction/config.py; already cached locally.
    ModelSpec("InternVL3-8B", "internvl", "OpenGVLab/InternVL3-8B", 180.0),
    # Same size class, one InternVL generation back. ~16GB cold download.
    ModelSpec("InternVL2.5-8B", "internvl", "OpenGVLab/InternVL2_5-8B", 1500.0),
    # Larger InternVL3-generation model, upper-bound accuracy/latency reference. ~28GB cold download.
    ModelSpec("InternVL3-14B", "internvl", "OpenGVLab/InternVL3-14B", 2700.0),
    # Smaller InternVL3-generation models -- lower latency, accuracy tradeoff TBD.
    ModelSpec("InternVL3-2B", "internvl", "OpenGVLab/InternVL3-2B", 600.0),
    ModelSpec("InternVL3-1B", "internvl", "OpenGVLab/InternVL3-1B", 400.0),
    # Robotics-adjacent candidate: already driving sam_vla's VLA rollout
    # policy elsewhere in this repo, so a real alternative, not speculative. ~16GB cold download.
    ModelSpec("Qwen2.5-VL-7B-Instruct", "qwen", "Qwen/Qwen2.5-VL-7B-Instruct", 1500.0),
]
