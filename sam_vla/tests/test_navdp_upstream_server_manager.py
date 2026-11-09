"""Integration project (next.md) Phase 1: exercises
NavdpUpstreamServerManager's path resolution and start()/stop() state
machine against mocked subprocess/socket/HTTP calls -- no real navdp_server
subprocess, checkpoint, or vendored checkout needed. See this module's
docstring for why start() is two-phase (poll a raw TCP connect, then exactly
one long-timeout /navigator_reset call) rather than repeatedly polling
/navigator_reset the way the socket-based server managers poll a ping route.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sam_vla.vlm.navdp_upstream_server_manager import (
    NavdpUpstreamServerManager,
    _expected_algo,
    resolve_navdp_upstream_root,
)


def test_resolve_navdp_upstream_root_raises_when_not_found(tmp_path, monkeypatch):
    monkeypatch.delenv("NAVDP_UPSTREAM_ROOT", raising=False)
    with pytest.raises(FileNotFoundError):
        resolve_navdp_upstream_root(str(tmp_path))


def test_resolve_navdp_upstream_root_finds_raw_path(tmp_path, monkeypatch):
    monkeypatch.delenv("NAVDP_UPSTREAM_ROOT", raising=False)
    server_dir = tmp_path / "baselines" / "navdp"
    server_dir.mkdir(parents=True)
    (server_dir / "navdp_server.py").write_text("# stub")

    resolved = resolve_navdp_upstream_root(str(tmp_path))
    assert resolved == tmp_path.resolve()


def test_resolve_navdp_upstream_root_finds_env_var(tmp_path, monkeypatch):
    server_dir = tmp_path / "baselines" / "navdp"
    server_dir.mkdir(parents=True)
    (server_dir / "navdp_server.py").write_text("# stub")
    monkeypatch.setenv("NAVDP_UPSTREAM_ROOT", str(tmp_path))

    resolved = resolve_navdp_upstream_root(None)
    assert resolved == tmp_path.resolve()


def test_resolve_navdp_upstream_root_s2diff_variant_needs_its_own_file(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("NAVDP_UPSTREAM_ROOT", raising=False)
    server_dir = tmp_path / "baselines" / "navdp"
    server_dir.mkdir(parents=True)
    (server_dir / "navdp_server.py").write_text("# stub")

    # Checkout only has the official server -- s2diff variant should not
    # silently resolve against it.
    with pytest.raises(FileNotFoundError):
        resolve_navdp_upstream_root(str(tmp_path), server_variant="s2diff")

    (server_dir / "navdp_s2diff_server.py").write_text("# stub")
    resolved = resolve_navdp_upstream_root(str(tmp_path), server_variant="s2diff")
    assert resolved == tmp_path.resolve()


def test_expected_algo_navdp_variant_is_always_navdp():
    assert _expected_algo("navdp", planner_mode="s2diff", remove_critic=True) == "navdp"
    assert _expected_algo("navdp", planner_mode="pure-navdp", remove_critic=False) == "navdp"


def test_expected_algo_s2diff_variant_depends_on_planner_mode_and_critic():
    assert (
        _expected_algo("s2diff", planner_mode="s2diff", remove_critic=True)
        == "navdp-hlc-s2diff-no-critic"
    )
    assert (
        _expected_algo("s2diff", planner_mode="s2diff", remove_critic=False)
        == "navdp-hlc-s2diff"
    )
    assert (
        _expected_algo("s2diff", planner_mode="gradient", remove_critic=True)
        == "navdp-hlc-gradient-no-critic"
    )
    assert (
        _expected_algo("s2diff", planner_mode="pure-navdp", remove_critic=True)
        == "navdp-pure-critic"
    )


def _make_manager(**kwargs) -> NavdpUpstreamServerManager:
    with patch(
        "sam_vla.vlm.navdp_upstream_server_manager.resolve_navdp_upstream_root",
        return_value=Path("/fake/navdp_upstream"),
    ):
        return NavdpUpstreamServerManager(checkpoint_path="/fake/ckpt.ckpt", **kwargs)


def test_start_adopts_already_running_server_without_spawning():
    manager = _make_manager()
    manager._port_open = MagicMock(return_value=True)
    manager._reset = MagicMock(return_value=True)

    with patch("sam_vla.vlm.navdp_upstream_server_manager.subprocess.Popen") as popen:
        manager.start()

    popen.assert_not_called()
    assert manager._owns_process is False
    assert manager.load_ms == 0.0


def test_start_spawns_and_waits_for_port_then_reset(monkeypatch):
    manager = _make_manager(start_timeout=5.0)

    # First health check (adopt-existing path) fails on _port_open alone --
    # short-circuits before _reset is ever called -- so we must spawn.
    port_open_results = iter([False, False, False, True])
    manager._port_open = MagicMock(side_effect=lambda timeout=1.0: next(port_open_results))
    # Only called once for real: the post-spawn /navigator_reset once the port opens.
    manager._reset = MagicMock(return_value=True)

    fake_process = MagicMock()
    fake_process.poll.return_value = None  # still running
    monkeypatch.setattr(
        "sam_vla.vlm.navdp_upstream_server_manager.subprocess.Popen",
        MagicMock(return_value=fake_process),
    )
    monkeypatch.setattr(
        "sam_vla.vlm.navdp_upstream_server_manager._resolve_navdp_upstream_python",
        lambda: "/fake/python",
    )
    monkeypatch.setattr("time.sleep", lambda _: None)

    manager.start()

    assert manager._owns_process is True
    assert manager.load_ms is not None and manager.load_ms >= 0.0


def test_start_raises_if_subprocess_exits_early(monkeypatch):
    manager = _make_manager(start_timeout=5.0)
    manager._port_open = MagicMock(return_value=False)
    manager._reset = MagicMock(return_value=False)

    fake_process = MagicMock()
    fake_process.poll.return_value = 1  # exited already
    fake_process.returncode = 1
    monkeypatch.setattr(
        "sam_vla.vlm.navdp_upstream_server_manager.subprocess.Popen",
        MagicMock(return_value=fake_process),
    )
    monkeypatch.setattr(
        "sam_vla.vlm.navdp_upstream_server_manager._resolve_navdp_upstream_python",
        lambda: "/fake/python",
    )
    monkeypatch.setattr("time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="exited early"):
        manager.start()


def test_stop_only_terminates_owned_process():
    manager = _make_manager()
    manager._owns_process = False
    manager._process = MagicMock()

    manager.stop()

    manager._process.terminate.assert_not_called()


def test_argv_navdp_variant_spawns_official_server_with_minimal_flags():
    manager = _make_manager()

    with patch(
        "sam_vla.vlm.navdp_upstream_server_manager._resolve_navdp_upstream_python",
        return_value="/fake/python",
    ):
        argv = manager._argv()

    assert argv == [
        "/fake/python",
        "navdp_server.py",
        "--port",
        str(manager.port),
        "--checkpoint",
        manager.checkpoint_path,
    ]


def test_argv_s2diff_variant_spawns_guided_server_with_its_own_flags():
    manager = _make_manager(
        server_variant="s2diff",
        planner_mode="gradient",
        device="cuda:1",
        remove_critic=False,
        s2diff_extra_args={"guidance-strength": 0.5, "particle-anchor": False},
    )

    with patch(
        "sam_vla.vlm.navdp_upstream_server_manager._resolve_navdp_upstream_python",
        return_value="/fake/python",
    ):
        argv = manager._argv()

    assert argv[:2] == ["/fake/python", "navdp_s2diff_server.py"]
    assert "--device" in argv and argv[argv.index("--device") + 1] == "cuda:1"
    assert (
        "--planner-mode" in argv
        and argv[argv.index("--planner-mode") + 1] == "gradient"
    )
    assert "--no-remove-critic" in argv
    assert "--guidance-strength" in argv
    assert argv[argv.index("--guidance-strength") + 1] == "0.5"
    assert "--no-particle-anchor" in argv
