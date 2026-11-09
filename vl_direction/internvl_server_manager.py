"""
Spawns and supervises the internvl_server subprocess, mirroring
sam_vla/vlm/qwen_server_manager.py's QwenServerManager exactly. Duck-types
the same start()/stop() shape as sam_vla.core.lifecycle.Service WITHOUT
importing it, keeping vl_direction fully standalone (next.md's stated
constraint: this module knows nothing about sam_vla internals).
"""

import json
import os
import socket
import struct
import subprocess
import time

from vl_direction.config import INTERNVL_SERVER_PORT

_HEADER_SIZE = 4
_HEALTH_CHECK_RETRY_INTERVAL = 1.0
_START_TIMEOUT = 30.0
_STOP_TIMEOUT = 5.0

# internvl_server needs transformers + InternVL's own stack, which live in
# the "vl" conda env (torch 2.11+cu128, transformers 5.x, InternVL3-8B) --
# not the env this manager runs in. Resolve that env's interpreter directly
# rather than relying on "python" from PATH.
_INTERNVL_VLM_CONDA_ENV = "vl"


def _resolve_internvl_vlm_python() -> str:
    override = os.environ.get("INTERNVL_VLM_PYTHON")
    if override:
        return override

    conda_info = subprocess.run(
        ["conda", "info", "--base"], capture_output=True, text=True, check=True
    ).stdout
    # Some conda installs print unrelated warnings to stdout before the
    # actual base path (see qwen_server_manager.py's identical handling).
    conda_base = next(
        line.strip() for line in conda_info.splitlines() if line.startswith("/")
    )
    candidate = os.path.join(
        conda_base, "envs", _INTERNVL_VLM_CONDA_ENV, "bin", "python"
    )
    if not os.path.exists(candidate):
        raise RuntimeError(
            f"could not find python for conda env '{_INTERNVL_VLM_CONDA_ENV}' at {candidate}; "
            "set INTERNVL_VLM_PYTHON to override"
        )
    return candidate


class InternVLServerManager:
    def __init__(
        self, port: int = None, model_path: str = None, startup_timeout: float = None
    ):
        self.port = port if port is not None else INTERNVL_SERVER_PORT
        self.model_path = model_path
        self.startup_timeout = (
            startup_timeout if startup_timeout is not None else _START_TIMEOUT
        )
        self._process = None
        self._owns_process = False
        # Study 1 (next.md) model_load_ms bucket: elapsed wall time this call to
        # start() spent waiting for the subprocess to come up. 0.0 if a server was
        # already running (nothing to attribute to this process's load).
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
                f"[InternVLServerManager] server already running on port {self.port}, not spawning"
            )
            self._owns_process = False
            self.load_ms = 0.0
            return

        print(
            f"[InternVLServerManager] no server on port {self.port}, spawning subprocess"
        )
        env = os.environ.copy()
        env["INTERNVL_SERVER_PORT"] = str(self.port)
        if self.model_path is not None:
            env["INTERNVL_MODEL_PATH"] = self.model_path
        self._process = subprocess.Popen(
            [_resolve_internvl_vlm_python(), "-m", "vl_direction.internvl_server"],
            cwd=os.getcwd(),
            env=env,
        )
        self._owns_process = True

        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self._health_check():
                self.load_ms = (time.monotonic() - t0) * 1000.0
                print(f"[InternVLServerManager] server is up ({self.load_ms:.0f}ms)")
                return
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"internvl_server subprocess exited early with code {self._process.returncode} "
                    f"(model_path={self.model_path!r}) -- check its stderr above"
                )
            time.sleep(_HEALTH_CHECK_RETRY_INTERVAL)

        raise RuntimeError(
            f"internvl_server did not respond to ping within {self.startup_timeout}s of spawning "
            f"(model_path={self.model_path!r})"
        )

    def stop(self) -> None:
        if not self._owns_process or self._process is None:
            print("[InternVLServerManager] not owned, skipping shutdown")
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
    manager = InternVLServerManager()
    print(
        "InternVLServerManager scaffolding is in place, but internvl_server.py "
        "cannot actually load a model yet (see internvl_model_runner.py). "
        "manager.start() would spawn the subprocess but the health check will "
        "never succeed until that's implemented."
    )
