# #!/usr/bin/env python3
# """Isolated Qwen homotopy service.

# Keeping Transformers/PyTorch CUDA in this process prevents native-library
# interaction with Habitat's EGL/OpenGL renderer in the rollout process.
# """

# from __future__ import annotations

# import argparse
# from dataclasses import asdict

# import numpy as np
# from flask import Flask, jsonify, request
# from PIL import Image

# from qwen_navdp_homotopy import VisualQwenHomotopySelector


# parser = argparse.ArgumentParser()
# parser.add_argument("--host", default="127.0.0.1")
# parser.add_argument("--port", type=int, default=8890)
# parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
# parser.add_argument("--device", default="auto")
# parser.add_argument("--minimum-obstacle-pixels", type=int, default=30)
# parser.add_argument("--release-clear-frames", type=int, default=8)
# parser.add_argument("--consistency-repeats", type=int, default=5)
# args = parser.parse_args()

# selector = VisualQwenHomotopySelector(
#     model_id=args.model_id,
#     device=args.device,
#     minimum_obstacle_pixels=args.minimum_obstacle_pixels,
#     release_clear_frames=args.release_clear_frames,
#     consistency_repeats=args.consistency_repeats,
# )
# app = Flask(__name__)


# @app.get("/health")
# def health():
#     return jsonify({"ready": True, "role": "obstacle_homotopy_only"})


# @app.post("/reset")
# def reset():
#     selector._latched_side = None
#     selector._latched_confidence = 0.0
#     selector._clear_frames = 0
#     return jsonify({"ok": True})


# @app.post("/select")
# def select():
#     try:
#         image = np.asarray(
#             Image.open(request.files["image"].stream).convert("RGB"),
#             dtype=np.uint8,
#         )
#         mask = np.asarray(
#             Image.open(request.files["obstacle_mask"].stream).convert("L"),
#             dtype=np.uint8,
#         )
#         if mask.shape != image.shape[:2]:
#             raise ValueError(
#                 f"mask/image shape mismatch: {mask.shape} vs {image.shape[:2]}"
#             )
#         decision = selector.step(image, (mask > 0).astype(np.uint8))
#         return jsonify(asdict(decision))
#     except (KeyError, ValueError) as error:
#         return jsonify({"error": str(error)}), 400
#     except Exception as error:
#         return jsonify({"error": f"Qwen homotopy inference failed: {error}"}), 500


# if __name__ == "__main__":
#     app.run(host=args.host, port=args.port, threaded=False)

#!/usr/bin/env python3
"""Isolated Qwen homotopy service.

Keeping Transformers/PyTorch CUDA in this process prevents native-library
interaction with Habitat's EGL/OpenGL renderer in the rollout process.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

from qwen_navdp_homotopy import VisualQwenHomotopySelector

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8890)
parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
parser.add_argument("--device", default="auto")
parser.add_argument("--minimum-obstacle-pixels", type=int, default=30)
parser.add_argument("--release-clear-frames", type=int, default=8)
parser.add_argument("--consistency-repeats", type=int, default=5)
args = parser.parse_args()

selector = VisualQwenHomotopySelector(
    model_id=args.model_id,
    device=args.device,
    minimum_obstacle_pixels=args.minimum_obstacle_pixels,
    release_clear_frames=args.release_clear_frames,
    consistency_repeats=args.consistency_repeats,
)
app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(
        {
            "ready": True,
            "roles": [
                "obstacle_homotopy",
                "return_or_stop_command",
                "mission_plan",
            ],
        }
    )


@app.post("/reset")
def reset():
    selector._latched_side = None
    selector._latched_confidence = 0.0
    selector._clear_frames = 0
    return jsonify({"ok": True})


@app.post("/select")
def select():
    try:
        image = np.asarray(
            Image.open(request.files["image"].stream).convert("RGB"),
            dtype=np.uint8,
        )
        mask = np.asarray(
            Image.open(request.files["obstacle_mask"].stream).convert("L"),
            dtype=np.uint8,
        )
        if mask.shape != image.shape[:2]:
            raise ValueError(
                f"mask/image shape mismatch: {mask.shape} vs {image.shape[:2]}"
            )
        decision = selector.step(image, (mask > 0).astype(np.uint8))
        return jsonify(asdict(decision))
    except (KeyError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        return jsonify({"error": f"Qwen homotopy inference failed: {error}"}), 500


@app.post("/command")
def command():
    try:
        image = np.asarray(
            Image.open(request.files["image"].stream).convert("RGB"),
            dtype=np.uint8,
        )
        user_command = str(request.form["command"])
        decision = selector.classify_command(image, user_command)
        return jsonify(asdict(decision))
    except (KeyError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        return jsonify({"error": f"Qwen command inference failed: {error}"}), 500


@app.post("/mission")
def mission():
    try:
        image = np.asarray(
            Image.open(request.files["image"].stream).convert("RGB"),
            dtype=np.uint8,
        )
        user_command = str(request.form["command"])
        decision = selector.classify_mission(image, user_command)
        payload = asdict(decision)
        payload["plan"] = list(decision.plan)
        return jsonify(payload)
    except (KeyError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        return jsonify({"error": f"Qwen mission inference failed: {error}"}), 500


if __name__ == "__main__":
    app.run(host=args.host, port=args.port, threaded=False)
