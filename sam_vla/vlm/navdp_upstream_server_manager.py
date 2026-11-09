"""Spawns and supervises the *upstream* NavDP server subprocess -- either
InternRobotics/NavDP's own baselines/navdp/navdp_server.py (the real,
published NavDP diffusion-policy-plus-critic model, not this repo's own
navdp/ package -- see next.md's "Integration project" section: NavdpPolicy
and navdp/ stay untouched; this is a new, parallel policy backend) or this
project's own baselines/navdp/navdp_s2diff_server.py (same NavDP_Agent/
checkpoint, wrapped with S2Diff obstacle-guided sampling -- see that file's
S2DiffPointGoalAgent). `server_variant` ("navdp" | "s2diff") picks which one
gets spawned; both live in the same vendored checkout, so this manager stays
"free to use between the official navdp vs a custom navdp server" rather
than being hardcoded to one.

Modeled on qwen_server_manager.QwenServerManager's spawn/poll/load_ms shape,
but upstream's server speaks plain HTTP/JSON (confirmed by reading
baselines/navdp/navdp_server.py and navdp_s2diff_server.py from
github.com/InternRobotics/NavDP@master and this repo's own fork of it),
not the length-prefixed local socket protocol qwen_server/internvl_server
use, and it has no cheap ping route -- /navigator_reset IS the health check,
same call next.md's Phase 1 plan says to use.

The two variants' /navigator_reset responses differ (confirmed by reading
both files directly, not guessed): navdp_server.py always returns
{"algo": "navdp"}; navdp_s2diff_server.py returns {"algo": planner_name()},
which depends on --planner-mode/--remove-critic (e.g.
"navdp-hlc-s2diff-no-critic" by default) and is never the literal string
"navdp" -- so the health check must know which variant it's polling and
compute the *expected* algo string per variant (_expected_algo below),
mirroring navdp_s2diff_server.py's own planner_name() function exactly,
rather than hardcoding "navdp" for both (that mismatch previously made
start() time out and raise even when the s2diff server had booted fine and
answered 200 -- confirmed by reading a real run's server log, where
/navigator_reset returned 200 seconds before this manager's own timeout
fired).

Two-phase start() (WHY, not obvious from a first read of navdp_server.py):
Flask's dev server (`app.run(...)`, no `threaded=True`) handles one request
at a time. The first /navigator_reset after boot also lazily constructs
NavDP_Agent and loads the checkpoint onto the GPU, which can take a long
time. If we polled by repeatedly firing short-timeout /navigator_reset calls
the way the socket-based managers poll a ping route, a slow model load would
strand several reset calls queued behind the one that's actually loading the
model -- each one reopens navdp_server's per-reset mp4 writer for nothing.
So: poll a raw TCP connect (cheap, tells us Flask itself is up) in a loop,
then make exactly ONE real /navigator_reset call with a long timeout once the
port is open, and let *that* call double as both the health check and the
episode's real reset (a fresh NavdpUpstreamServerManager is constructed per
rollout episode, so there is no separate "real" reset to make later).
"""

from __future__ import annotations

import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import requests

from sam_vla.core.goal_geometry import intrinsics_from_hfov

_PORT_POLL_INTERVAL = 1.0
_START_TIMEOUT = 180.0  # loading the diffusion model + critic onto GPU can be slow
_RESET_CALL_TIMEOUT = 120.0
_STOP_TIMEOUT = 5.0
_DEFAULT_PORT = 8766  # distinct from QWEN_SERVER_PORT (8765)

# navdp_server.py needs Flask + upstream's own torch pin, in a dedicated
# "navdp" conda env per NavDP's README (`conda create -n navdp python=3.10 &&
# pip install -r requirements.txt` from the vendored checkout) -- distinct
# from every env in this repo's own table (none of habitat/sam2/sam3/vl/
# qwen_vlm have Flask + this exact pin).
_NAVDP_UPSTREAM_CONDA_ENV = "navdp"

# Filenames are relative to <navdp_upstream_root>/baselines/navdp -- see this
# module's docstring for how the two variants' /navigator_reset responses
# differ.
_SERVER_FILENAMES = {"navdp": "navdp_server.py", "s2diff": "navdp_s2diff_server.py"}


def _expected_algo(server_variant: str, planner_mode: str, remove_critic: bool) -> str:
    """Mirrors navdp_s2diff_server.py's own planner_name() exactly (read
    directly from that file, not guessed) so the health check knows what
    /navigator_reset will actually answer for the args we're about to pass
    it. navdp_server.py (the "navdp" variant) has no such logic -- it always
    answers {"algo": "navdp"}."""
    if server_variant == "navdp":
        return "navdp"
    if planner_mode == "pure-navdp":
        return "navdp-pure-critic"
    if planner_mode == "gradient":
        return "navdp-hlc-gradient-no-critic" if remove_critic else "navdp-hlc-gradient"
    return "navdp-hlc-s2diff-no-critic" if remove_critic else "navdp-hlc-s2diff"


def _resolve_navdp_upstream_python() -> str:
    override = os.environ.get("NAVDP_UPSTREAM_PYTHON")
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
    candidate = os.path.join(
        conda_base, "envs", _NAVDP_UPSTREAM_CONDA_ENV, "bin", "python"
    )
    if not os.path.exists(candidate):
        raise RuntimeError(
            f"could not find python for conda env '{_NAVDP_UPSTREAM_CONDA_ENV}' at "
            f"{candidate}; set NAVDP_UPSTREAM_PYTHON to override"
        )
    return candidate


def resolve_navdp_upstream_root(
    raw: Optional[str], server_variant: str = "navdp"
) -> Path:
    """Locates the vendored InternRobotics/NavDP checkout (git-cloned
    separately per next.md's Integration-project Phase 0 -- never vendored
    into this repo's own git history, same "external, read-from dependency"
    discipline this repo already applies to its own navdp/ and belief_exp's
    README). raw > $NAVDP_UPSTREAM_ROOT; unlike this repo's own navdp/, there
    is no in-repo fallback path since nothing here vendors it by default.
    Checks for whichever server file server_variant needs, so a checkout
    that only has one of the two variants' server scripts fails fast here
    rather than at spawn time."""
    filename = _SERVER_FILENAMES[server_variant]
    candidates = []
    if raw:
        candidates.append(Path(raw))
    env = os.environ.get("NAVDP_UPSTREAM_ROOT")
    if env:
        candidates.append(Path(env))
    for c in candidates:
        c = c.expanduser().resolve()
        if (c / "baselines" / "navdp" / filename).exists():
            return c
    raise FileNotFoundError(
        f"Could not find the vendored InternRobotics/NavDP checkout (expected "
        f"baselines/navdp/{filename} for server_variant={server_variant!r}). Pass "
        "navdp_upstream_root=/path/to/NavDP or set NAVDP_UPSTREAM_ROOT"
    )


def _reset_payload(
    image_height: int,
    image_width: int,
    hfov_deg: float,
    stop_threshold: float,
    batch_size: int,
) -> dict:
    """/navigator_reset's body: intrinsic is a 3x3 camera matrix (confirmed from
    policy_agent.py's NavDP_Agent -- indexed intrinsic[1][1]=fy, intrinsic[1][2]=cy,
    intrinsic[0][0]=fx, intrinsic[0][2]=cx), used only by project_trajectory's
    debug-video overlay, never by the policy/critic networks themselves, so an
    approximate HFOV-derived intrinsic (this repo's own intrinsics_from_hfov,
    shared with the CBF/belief-tracking pinhole model) is fine here."""
    intr = intrinsics_from_hfov(image_height, image_width, hfov_deg)
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    return {
        "intrinsic": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "stop_threshold": float(stop_threshold),
        "batch_size": int(batch_size),
    }


class NavdpUpstreamServerManager:
    """Spawn/poll/own-or-adopt lifecycle for navdp_server.py, satisfying the
    same Service protocol (start()/stop(), sam_vla.core.lifecycle) and
    load_ms convention (Study 1, next.md) as QwenServerManager /
    InternVLServerManager, so it can be registered with ServiceRegistry the
    same way."""

    def __init__(
        self,
        checkpoint_path: str,
        navdp_upstream_root: Optional[str] = None,
        port: Optional[int] = None,
        stop_threshold: float = 0.0,
        batch_size: int = 1,
        image_hw: tuple[int, int] = (480, 640),
        hfov_deg: float = 90.0,
        start_timeout: float = _START_TIMEOUT,
        server_variant: str = "navdp",
        device: str = "cuda:0",
        planner_mode: str = "s2diff",
        remove_critic: bool = True,
        s2diff_extra_args: Optional[dict] = None,
    ):
        if server_variant not in _SERVER_FILENAMES:
            raise ValueError(
                f"server_variant must be one of {sorted(_SERVER_FILENAMES)}, "
                f"got {server_variant!r}"
            )
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.server_variant = server_variant
        self.navdp_upstream_root = resolve_navdp_upstream_root(
            navdp_upstream_root, server_variant=server_variant
        )
        self.port = port if port is not None else _DEFAULT_PORT
        self.stop_threshold = float(stop_threshold)
        self.batch_size = int(batch_size)
        self.image_hw = image_hw
        self.hfov_deg = float(hfov_deg)
        self.start_timeout = float(start_timeout)
        # s2diff-only launch knobs (see navdp_s2diff_server.py's argparse for
        # what these gate); s2diff_extra_args passes through anything else
        # (e.g. guidance-strength, safe-distance) without this constructor
        # having to mirror every one of that file's ~30 CLI flags.
        self.device = str(device)
        self.planner_mode = str(planner_mode)
        self.remove_critic = bool(remove_critic)
        self.s2diff_extra_args = dict(s2diff_extra_args or {})
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._process: Optional[subprocess.Popen] = None
        self._owns_process = False
        self._log_file = None
        # Study 1 (next.md) model_load_ms bucket -- see this module's docstring
        # for why this is charged to the one real /navigator_reset call rather
        # than to repeated health-check polling.
        self.load_ms: Optional[float] = None
        self.log_path = os.path.join(
            tempfile.gettempdir(), f"navdp_upstream_server_{self.port}.log"
        )

    def _port_open(self, timeout: float = 1.0) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=timeout):
                return True
        except OSError:
            return False

    def _reset(self, timeout: float) -> bool:
        height, width = self.image_hw
        expected_algo = _expected_algo(
            self.server_variant, self.planner_mode, self.remove_critic
        )
        try:
            resp = requests.post(
                f"{self.base_url}/navigator_reset",
                json=_reset_payload(
                    height, width, self.hfov_deg, self.stop_threshold, self.batch_size
                ),
                timeout=timeout,
            )
            return resp.ok and resp.json().get("algo") == expected_algo
        except (requests.RequestException, ValueError):
            return False

    def _argv(self) -> list:
        filename = _SERVER_FILENAMES[self.server_variant]
        argv = [
            _resolve_navdp_upstream_python(),
            filename,
            "--port",
            str(self.port),
            "--checkpoint",
            self.checkpoint_path,
        ]
        if self.server_variant == "s2diff":
            argv += [
                "--device",
                self.device,
                "--planner-mode",
                self.planner_mode,
                "--remove-critic" if self.remove_critic else "--no-remove-critic",
            ]
            for flag, value in self.s2diff_extra_args.items():
                flag = f"--{flag.lstrip('-')}"
                if isinstance(value, bool):
                    argv.append(flag if value else flag.replace("--", "--no-", 1))
                else:
                    argv += [flag, str(value)]
        return argv

    def start(self) -> None:
        t0 = time.monotonic()
        if self._port_open() and self._reset(timeout=_RESET_CALL_TIMEOUT):
            print(
                f"[NavdpUpstreamServerManager] server already running on port "
                f"{self.port}, not spawning (reset for this episode done above)"
            )
            self._owns_process = False
            self.load_ms = 0.0
            return

        server_dir = self.navdp_upstream_root / "baselines" / "navdp"
        print(
            f"[NavdpUpstreamServerManager] no server on port {self.port}, "
            f"spawning subprocess from {server_dir} (log: {self.log_path})"
        )
        # Redirect rather than inherit stdout/stderr: an inherited fd ties this
        # subprocess's lifetime to whatever the caller's own stdout happens to
        # be connected to (e.g. a caller piped through `| tail`, which buffers
        # until EOF) -- if the caller exits/is killed without this process also
        # exiting, that pipe never closes and downstream readers hang forever
        # waiting for output that already happened. Verified live: this is
        # exactly what stranded an earlier smoke-test run for hours.
        self._log_file = open(self.log_path, "ab")
        self._process = subprocess.Popen(
            self._argv(),
            cwd=str(server_dir),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        self._owns_process = True

        deadline = time.time() + self.start_timeout
        while time.time() < deadline:
            if self._port_open():
                break
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"navdp_server subprocess exited early with code "
                    f"{self._process.returncode} (checkpoint={self.checkpoint_path!r}) "
                    f"-- check {self.log_path}"
                )
            time.sleep(_PORT_POLL_INTERVAL)
        else:
            raise RuntimeError(
                f"navdp_server's port {self.port} never opened within "
                f"{self.start_timeout}s of spawning -- check {self.log_path}"
            )

        remaining = max(deadline - time.time(), _RESET_CALL_TIMEOUT)
        if not self._reset(timeout=remaining):
            expected_algo = _expected_algo(
                self.server_variant, self.planner_mode, self.remove_critic
            )
            raise RuntimeError(
                "navdp_server's port opened but /navigator_reset did not return "
                f"{{'algo': {expected_algo!r}}} within {remaining:.0f}s (server_variant="
                f"{self.server_variant!r}) -- checkpoint load may have failed "
                f"(checkpoint={self.checkpoint_path!r}); check {self.log_path}"
            )
        self.load_ms = (time.monotonic() - t0) * 1000.0
        print(f"[NavdpUpstreamServerManager] server is up ({self.load_ms:.0f}ms)")

    def stop(self) -> None:
        if not self._owns_process or self._process is None:
            print("[NavdpUpstreamServerManager] not owned, skipping shutdown")
            return

        self._process.terminate()
        try:
            self._process.wait(timeout=_STOP_TIMEOUT)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()

        self._process = None
        self._owns_process = False
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Smoke-start a vendored upstream navdp server subprocess "
        "(either variant)."
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--navdp-upstream-root", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument(
        "--server-variant", choices=sorted(_SERVER_FILENAMES), default="navdp"
    )
    ap.add_argument("--planner-mode", choices=["pure-navdp", "s2diff", "gradient"], default="s2diff")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    manager = NavdpUpstreamServerManager(
        checkpoint_path=args.checkpoint,
        navdp_upstream_root=args.navdp_upstream_root,
        port=args.port,
        server_variant=args.server_variant,
        planner_mode=args.planner_mode,
        device=args.device,
    )
    manager.start()
    print(f"load_ms={manager.load_ms:.0f}")
    manager.stop()
