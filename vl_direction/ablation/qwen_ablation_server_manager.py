"""
Spawns and supervises the qwen_ablation_server subprocess -- same shape as
vl_direction/internvl_server_manager.py's InternVLServerManager, kept as a
separate class (rather than generalizing that one) since the two servers
load fundamentally different model families; sharing a class would just be
an if/else on backend type wearing a trench coat. See model_specs.py for
how run_ablation.py picks between them.

Runs in the "vl" conda env, NOT sam_vla's "qwen_vlm" env: qwen_vlm doesn't
exist on this machine yet (CLAUDE.md notes it needs creating before
Qwen-based sam_vla rollouts will run), and creating it from scratch is out
of scope for an ablation run. "vl" already has transformers 4.57 with
Qwen2_5_VLForConditionalGeneration plus a working CUDA torch build
(verified before writing this), so it serves both InternVL and this
ablation's Qwen candidate without any new environment.
"""

import json
import os
import socket
import struct
import subprocess
import time

_HEADER_SIZE = 4
_HEALTH_CHECK_RETRY_INTERVAL = 1.0
_STOP_TIMEOUT = 5.0

_QWEN_ABLATION_CONDA_ENV = "vl"


def _resolve_qwen_ablation_python() -> str:
    override = os.environ.get("QWEN_ABLATION_PYTHON")
    if override:
        return override

    conda_info = subprocess.run(
        ["conda", "info", "--base"], capture_output=True, text=True, check=True
    ).stdout
    conda_base = next(
        line.strip() for line in conda_info.splitlines() if line.startswith("/")
    )
    candidate = os.path.join(
        conda_base, "envs", _QWEN_ABLATION_CONDA_ENV, "bin", "python"
    )
    if not os.path.exists(candidate):
        raise RuntimeError(
            f"could not find python for conda env '{_QWEN_ABLATION_CONDA_ENV}' at {candidate}; "
            "set QWEN_ABLATION_PYTHON to override"
        )
    return candidate


class QwenAblationServerManager:
    def __init__(
        self, port: int = None, model_path: str = None, startup_timeout: float = 60.0
    ):
        from vl_direction.ablation.qwen_ablation_server import _DEFAULT_PORT

        self.port = port if port is not None else _DEFAULT_PORT
        self.model_path = model_path
        self.startup_timeout = startup_timeout
        self._process = None
        self._owns_process = False

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
        if self._health_check():
            print(
                f"[QwenAblationServerManager] server already running on port {self.port}, not spawning"
            )
            self._owns_process = False
            return

        print(
            f"[QwenAblationServerManager] no server on port {self.port}, spawning subprocess"
        )
        env = os.environ.copy()
        env["QWEN_ABLATION_SERVER_PORT"] = str(self.port)
        if self.model_path is not None:
            env["QWEN_ABLATION_MODEL_PATH"] = self.model_path
        self._process = subprocess.Popen(
            [
                _resolve_qwen_ablation_python(),
                "-m",
                "vl_direction.ablation.qwen_ablation_server",
            ],
            cwd=os.getcwd(),
            env=env,
        )
        self._owns_process = True

        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self._health_check():
                print("[QwenAblationServerManager] server is up")
                return
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"qwen_ablation_server subprocess exited early with code {self._process.returncode} "
                    f"(model_path={self.model_path!r}) -- check its stderr above"
                )
            time.sleep(_HEALTH_CHECK_RETRY_INTERVAL)

        raise RuntimeError(
            f"qwen_ablation_server did not respond to ping within {self.startup_timeout}s of spawning "
            f"(model_path={self.model_path!r})"
        )

    def stop(self) -> None:
        if not self._owns_process or self._process is None:
            print("[QwenAblationServerManager] not owned, skipping shutdown")
            return

        self._process.terminate()
        try:
            self._process.wait(timeout=_STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()

        self._process = None
        self._owns_process = False
