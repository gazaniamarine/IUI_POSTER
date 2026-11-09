"""
Spawns and supervises the qwen35_server subprocess, satisfying the lifecycle
Service protocol so it can be registered with ServiceRegistry. Sibling of
qwen_server_manager.py -- same subprocess-supervision logic, points at
qwen35_server/qwen35_config instead.
"""

import json
import os
import socket
import struct
import subprocess
import time

from sam_vla.vlm.qwen35_config import QWEN35_SERVER_PORT

_HEADER_SIZE = 4
_HEALTH_CHECK_RETRY_INTERVAL = 1.0
# Longer than qwen_server_manager.py's 30s: Qwen3.5-9B is a bigger checkpoint
# to load than the 3B model that timeout was tuned for. Still assumes the
# weights are already in the local HF cache -- a cold first-time download
# (~18GB) can exceed even this and should be done once up front via
# `python -m sam_vla.vlm.qwen35_model_runner <image>` (see qwen35_model_runner's
# own __main__), not discovered here as a spurious startup failure.
_START_TIMEOUT = 60.0
_STOP_TIMEOUT = 5.0

# qwen35_server needs transformers + the Qwen3.5 stack, which live in the
# qwen_vlm conda env, not necessarily the caller's own env (habitat). Resolve
# that env's interpreter directly rather than relying on "python" from PATH
# -- same helper qwen_server_manager.py uses, since it's the same env.
_QWEN_VLM_CONDA_ENV = "qwen_vlm"


def _resolve_qwen_vlm_python() -> str:
    override = os.environ.get("QWEN_VLM_PYTHON")
    if override:
        return override

    conda_info = subprocess.run(
        ["conda", "info", "--base"], capture_output=True, text=True, check=True
    ).stdout
    # Some conda installs print unrelated warnings (e.g. a broken
    # anaconda-anon-usage plugin) to stdout before the actual base path.
    conda_base = next(
        line.strip() for line in conda_info.splitlines() if line.startswith("/")
    )
    candidate = os.path.join(conda_base, "envs", _QWEN_VLM_CONDA_ENV, "bin", "python")
    if not os.path.exists(candidate):
        raise RuntimeError(
            f"could not find python for conda env '{_QWEN_VLM_CONDA_ENV}' at {candidate}; "
            "set QWEN_VLM_PYTHON to override"
        )
    return candidate


class Qwen35ServerManager:
    def __init__(
        self, port: int = None, model_id: str = None, prompt_style: str = None
    ):
        self.port = port if port is not None else QWEN35_SERVER_PORT
        # None keeps the Qwen3.5-9B / original-prompt defaults. Passed to the
        # spawned server via env (QWEN35_MODEL_ID / QWEN35_PROMPT_STYLE).
        self.model_id = model_id
        self.prompt_style = prompt_style
        self._process = None
        self._owns_process = False
        self.load_ms = None

    def _health_check(self, timeout: float = 1.0) -> bool:
        try:
            with socket.create_connection(
                ("127.0.0.1", self.port), timeout=timeout
            ) as conn:
                conn.settimeout(timeout)
                payload = json.dumps({"mode": "ping"}).encode("utf-8")
                conn.sendall(struct.pack(">I", len(payload)) + payload)

                header = conn.recv(_HEADER_SIZE)
                if len(header) < _HEADER_SIZE:
                    return False
                (payload_len,) = struct.unpack(">I", header)
                response = conn.recv(payload_len)
                message = json.loads(response.decode("utf-8"))
                return message.get("status") == "ok"
        except (OSError, socket.timeout, json.JSONDecodeError):
            return False

    def start(self) -> None:
        t0 = time.monotonic()
        if self._health_check():
            print(
                f"[Qwen35ServerManager] server already running on port {self.port}, not spawning"
            )
            self._owns_process = False
            self.load_ms = 0.0
            return

        print(
            f"[Qwen35ServerManager] no server on port {self.port}, spawning subprocess"
        )
        env = os.environ.copy()
        if self.model_id:
            env["QWEN35_MODEL_ID"] = self.model_id
        if self.prompt_style:
            env["QWEN35_PROMPT_STYLE"] = self.prompt_style
        self._process = subprocess.Popen(
            [
                _resolve_qwen_vlm_python(),
                "-m",
                "sam_vla.vlm.qwen35_server",
                "--port",
                str(self.port),
            ],
            cwd=os.getcwd(),
            env=env,
        )
        self._owns_process = True

        deadline = time.time() + _START_TIMEOUT
        while time.time() < deadline:
            if self._health_check():
                self.load_ms = (time.monotonic() - t0) * 1000.0
                print(f"[Qwen35ServerManager] server is up ({self.load_ms:.0f}ms)")
                return
            time.sleep(_HEALTH_CHECK_RETRY_INTERVAL)

        raise RuntimeError(
            f"qwen35_server did not respond to ping within {_START_TIMEOUT}s of spawning"
        )

    def stop(self) -> None:
        if not self._owns_process or self._process is None:
            print("[Qwen35ServerManager] not owned, skipping shutdown")
            return

        self._process.terminate()
        try:
            self._process.wait(timeout=_STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()

        self._process = None
        self._owns_process = False


if __name__ == "__main__":
    manager = Qwen35ServerManager()

    manager.start()
    print("health check after first start():", manager._health_check())

    owned_before = manager._owns_process
    manager.start()
    print(
        "owns_process changed on second start():", owned_before != manager._owns_process
    )

    manager.stop()
    print("health check after stop():", manager._health_check())
