"""Study 2 (next.md): exercises study2_noise_sweep's command construction,
forwarded-arg validation, seed stability, and manifest bookkeeping against a
mocked subprocess.run -- no real rollout, sim, or subprocess."""

import json
from unittest.mock import patch

import pytest

from sam_vla.study2_noise_sweep import (
    _episode_seed,
    build_command,
    check_forwarded_args,
    load_episodes,
    run_sweep,
)

EPISODES = [
    {"id": "ep00", "start_x": 0.0, "start_z": 8.0, "start_yaw": 0.0},
    {"id": "ep01", "start_x": 2.0, "start_z": 6.0, "start_yaw": 90.0},
]
NOISE_LEVELS = [0.0, 0.05, 0.1]


def test_load_episodes_round_trips(tmp_path):
    ep_file = tmp_path / "episodes.json"
    ep_file.write_text(json.dumps(EPISODES))
    assert load_episodes(ep_file) == EPISODES


def test_load_episodes_rejects_missing_key(tmp_path):
    ep_file = tmp_path / "episodes.json"
    ep_file.write_text(json.dumps([{"id": "ep00", "start_x": 0.0, "start_z": 8.0}]))
    with pytest.raises(ValueError, match="missing required key"):
        load_episodes(ep_file)


def test_load_episodes_rejects_duplicate_id(tmp_path):
    ep_file = tmp_path / "episodes.json"
    ep_file.write_text(json.dumps([EPISODES[0], EPISODES[0]]))
    with pytest.raises(ValueError, match="duplicate episode id"):
        load_episodes(ep_file)


def test_check_forwarded_args_rejects_driver_owned_flag():
    with pytest.raises(ValueError, match="drive-odom-noise-std"):
        check_forwarded_args(["--scene-path", "x.glb", "--drive-odom-noise-std", "0.1"])


def test_check_forwarded_args_allows_unrelated_flags():
    check_forwarded_args(["--scene-path", "x.glb", "--cbf", "--max-steps", "300"])


def test_episode_seed_is_stable_and_distinct_per_episode():
    assert _episode_seed(0) == _episode_seed(0)
    assert _episode_seed(0) != _episode_seed(1)


def test_build_command_shape(tmp_path):
    cmd = build_command(
        "python",
        EPISODES[0],
        0.1,
        12345,
        tmp_path / "ep00_noise0.1",
        ["--scene-path", "x.glb", "--ckpt", "c.pt"],
    )
    assert cmd[:3] == ["python", "-m", "sam_vla.run_navdp_rollout"]
    assert "--scene-path" in cmd and "x.glb" in cmd
    assert cmd[cmd.index("--start-x") + 1] == "0.0"
    assert cmd[cmd.index("--start-z") + 1] == "8.0"
    assert cmd[cmd.index("--start-yaw") + 1] == "0.0"
    assert cmd[cmd.index("--drive-odom-noise-std") + 1] == "0.1"
    assert cmd[cmd.index("--drive-odom-noise-seed") + 1] == "12345"
    assert cmd[cmd.index("--out-dir") + 1] == str(tmp_path / "ep00_noise0.1")


def _mock_completed(returncode=0):
    class _Result:
        pass

    r = _Result()
    r.returncode = returncode
    return r


def test_run_sweep_writes_manifest_for_all_runs(tmp_path):
    out_dir = tmp_path / "study2_run"
    with patch(
        "sam_vla.study2_noise_sweep.subprocess.run", return_value=_mock_completed(0)
    ) as mock_run:
        manifest = run_sweep(
            EPISODES, NOISE_LEVELS, out_dir, "python", ["--scene-path", "x.glb"]
        )

    # 2 episodes * 3 noise levels = 6 subprocess invocations
    assert mock_run.call_count == 6
    assert len(manifest["episodes"]) == 2
    for ep_record in manifest["episodes"]:
        assert set(ep_record["runs"]) == {"0.0", "0.05", "0.1"}
        for run_record in ep_record["runs"].values():
            assert run_record["returncode"] == 0

    manifest_path = out_dir / "sweep_manifest.json"
    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text()) == manifest


def test_run_sweep_reuses_same_seed_across_noise_levels_for_an_episode(tmp_path):
    out_dir = tmp_path / "study2_run"
    with patch(
        "sam_vla.study2_noise_sweep.subprocess.run", return_value=_mock_completed(0)
    ):
        manifest = run_sweep(
            EPISODES, NOISE_LEVELS, out_dir, "python", ["--scene-path", "x.glb"]
        )

    for ep_record in manifest["episodes"]:
        seeds_used = set()
        for run_record in ep_record["runs"].values():
            cmd = run_record["cmd"]
            seeds_used.add(cmd[cmd.index("--drive-odom-noise-seed") + 1])
        assert seeds_used == {str(ep_record["seed"])}


def test_run_sweep_stops_on_failure_by_default(tmp_path):
    out_dir = tmp_path / "study2_run"
    with patch(
        "sam_vla.study2_noise_sweep.subprocess.run",
        return_value=_mock_completed(1),
    ) as mock_run:
        with pytest.raises(RuntimeError, match="failed with returncode=1"):
            run_sweep(EPISODES, NOISE_LEVELS, out_dir, "python", ["--scene-path", "x.glb"])

    # stops after the very first (episode, noise_level) run fails
    assert mock_run.call_count == 1


def test_run_sweep_keep_going_runs_all(tmp_path):
    out_dir = tmp_path / "study2_run"
    with patch(
        "sam_vla.study2_noise_sweep.subprocess.run",
        return_value=_mock_completed(1),
    ) as mock_run:
        manifest = run_sweep(
            EPISODES,
            NOISE_LEVELS,
            out_dir,
            "python",
            ["--scene-path", "x.glb"],
            keep_going=True,
        )

    assert mock_run.call_count == 6
    for ep_record in manifest["episodes"]:
        for run_record in ep_record["runs"].values():
            assert run_record["returncode"] == 1


def test_run_sweep_dry_run_does_not_invoke_subprocess(tmp_path):
    out_dir = tmp_path / "study2_run"
    with patch("sam_vla.study2_noise_sweep.subprocess.run") as mock_run:
        manifest = run_sweep(
            EPISODES, NOISE_LEVELS, out_dir, "python", ["--scene-path", "x.glb"], dry_run=True
        )

    mock_run.assert_not_called()
    for ep_record in manifest["episodes"]:
        for run_record in ep_record["runs"].values():
            assert run_record["returncode"] is None
