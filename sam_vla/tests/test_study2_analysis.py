"""Study 2 (next.md): exercises study2_analysis's per-run metric extraction
and noise_level aggregation against synthetic manifest.json/rollout.npz
fixtures -- no real rollout, sim, or subprocess. Fixture shape mirrors
sam_vla/logging/rollout_logger.py's own round-trip self-test
(manifest["steps"][i]["timestamp"]/["vla_result"], rollout.npz's
distances_to_goal), following test_study1_analysis.py's pattern."""

import json
from pathlib import Path

import numpy as np
import pytest

from sam_vla.study2_analysis import (
    analyze_sweep,
    load_run_metrics,
    summarize_by_noise_level,
    write_csv,
)

TS = [
    "2026-08-06T00:00:00+00:00",
    "2026-08-06T00:00:05+00:00",
    "2026-08-06T00:00:10+00:00",
]


def _write_fixture_run(run_dir: Path, timestamps, distances, vla_results=None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    vla_results = vla_results or [{}] * len(timestamps)
    manifest = {
        "start_time": timestamps[0],
        "steps": [
            {"timestamp": ts, "vla_result": vr}
            for ts, vr in zip(timestamps, vla_results)
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    np.savez(run_dir / "rollout.npz", distances_to_goal=np.array(distances, dtype=np.float64))


def test_load_run_metrics_timing_and_success(tmp_path):
    run_dir = tmp_path / "ep00_noise0.0"
    _write_fixture_run(run_dir, TS, distances=[5.0, 2.0, 0.5])

    metrics = load_run_metrics(run_dir, success_radius=1.0)

    assert metrics["n_steps"] == 3
    assert metrics["total_episode_time_s"] == pytest.approx(10.0)
    assert metrics["success"] is True
    assert metrics["steps_to_goal"] == 2
    assert metrics["final_distance_to_goal"] == pytest.approx(0.5)
    assert metrics["cbf_blocked_steps"] == 0
    assert metrics["cbf_trigger_rate"] == pytest.approx(0.0)
    assert metrics["mean_uncertainty"] is None


def test_load_run_metrics_failure_when_never_close(tmp_path):
    run_dir = tmp_path / "ep00_noise0.1"
    _write_fixture_run(run_dir, TS, distances=[8.0, 6.0, 5.0])

    metrics = load_run_metrics(run_dir, success_radius=1.0)
    assert metrics["success"] is False
    assert metrics["steps_to_goal"] is None


def test_load_run_metrics_all_nan_distances_is_not_success(tmp_path):
    run_dir = tmp_path / "ep00_noise0.1"
    _write_fixture_run(run_dir, TS, distances=[float("nan")] * 3)

    metrics = load_run_metrics(run_dir, success_radius=1.0)
    assert metrics["success"] is False


def test_load_run_metrics_aggregates_cbf_and_uncertainty(tmp_path):
    run_dir = tmp_path / "ep00_noise0.15"
    vla_results = [
        {"blocked": True, "hard_gate_fired": True, "uncertainty_value": 0.1},
        {"blocked": True, "hard_gate_fired": False, "uncertainty_value": 0.3},
        {"blocked": False, "hard_gate_fired": False, "uncertainty_value": 0.2},
    ]
    _write_fixture_run(run_dir, TS, distances=[5.0, 2.0, 0.5], vla_results=vla_results)

    metrics = load_run_metrics(run_dir, success_radius=1.0)

    assert metrics["cbf_blocked_steps"] == 2
    assert metrics["cbf_hard_gate_steps"] == 1
    assert metrics["cbf_trigger_rate"] == pytest.approx(2 / 3)
    assert metrics["mean_uncertainty"] == pytest.approx(0.2)
    assert metrics["max_uncertainty"] == pytest.approx(0.3)


def _sweep_manifest_for(runs_by_level: dict) -> dict:
    return {
        "episodes": [
            {
                "id": "ep00",
                "seed": 20260807,
                "runs": {
                    level: {"out_dir": str(out_dir), "returncode": 0}
                    for level, out_dir in runs_by_level.items()
                },
            }
        ]
    }


def test_analyze_sweep_reads_noise_level_from_key(tmp_path):
    run_dir = tmp_path / "ep00_noise0.1"
    _write_fixture_run(run_dir, TS, distances=[5.0, 2.0, 0.5])
    manifest = _sweep_manifest_for({"0.1": run_dir})

    rows = analyze_sweep(manifest, success_radius=1.0)
    assert len(rows) == 1
    assert rows[0]["noise_level"] == pytest.approx(0.1)
    assert rows[0]["seed"] == 20260807
    assert rows[0]["episode_id"] == "ep00"


def test_analyze_sweep_skips_failed_run(tmp_path):
    ok_dir = tmp_path / "ep00_noise0.0"
    _write_fixture_run(ok_dir, TS, distances=[5.0, 2.0, 0.5])
    manifest = {
        "episodes": [
            {
                "id": "ep00",
                "seed": 1,
                "runs": {
                    "0.0": {"out_dir": str(ok_dir), "returncode": 0},
                    "0.1": {"out_dir": str(tmp_path / "missing"), "returncode": 1},
                },
            }
        ]
    }

    rows = analyze_sweep(manifest, success_radius=1.0)
    assert len(rows) == 1
    assert rows[0]["noise_level"] == pytest.approx(0.0)


def test_summarize_by_noise_level_groups_and_sorts(tmp_path):
    run_a1 = tmp_path / "epA_noise0.0"
    run_a2 = tmp_path / "epA_noise0.1"
    run_b1 = tmp_path / "epB_noise0.0"
    run_b2 = tmp_path / "epB_noise0.1"
    _write_fixture_run(run_a1, TS, distances=[5.0, 2.0, 0.5])
    _write_fixture_run(run_b1, TS, distances=[5.0, 2.0, 0.5])
    _write_fixture_run(run_a2, TS, distances=[8.0, 7.0, 6.0])  # fails
    _write_fixture_run(run_b2, TS, distances=[8.0, 7.0, 6.0])  # fails

    manifest = {
        "episodes": [
            {
                "id": "epA",
                "seed": 1,
                "runs": {
                    "0.1": {"out_dir": str(run_a2), "returncode": 0},
                    "0.0": {"out_dir": str(run_a1), "returncode": 0},
                },
            },
            {
                "id": "epB",
                "seed": 2,
                "runs": {
                    "0.1": {"out_dir": str(run_b2), "returncode": 0},
                    "0.0": {"out_dir": str(run_b1), "returncode": 0},
                },
            },
        ]
    }

    rows = analyze_sweep(manifest, success_radius=1.0)
    summary = summarize_by_noise_level(rows)

    assert [row["noise_level"] for row in summary] == [0.0, 0.1]
    zero_row = summary[0]
    assert zero_row["n_runs"] == 2
    assert zero_row["success_rate"] == pytest.approx(1.0)
    point1_row = summary[1]
    assert point1_row["success_rate"] == pytest.approx(0.0)


def test_write_csv_round_trips(tmp_path):
    run_dir = tmp_path / "ep00_noise0.0"
    _write_fixture_run(run_dir, TS, distances=[5.0, 2.0, 0.5])
    rows = analyze_sweep(_sweep_manifest_for({"0.0": run_dir}), success_radius=1.0)

    out_csv = tmp_path / "analysis.csv"
    write_csv(rows, out_csv)

    import csv as csv_module

    with open(out_csv) as f:
        read_rows = list(csv_module.DictReader(f))
    assert len(read_rows) == 1
    assert read_rows[0]["episode_id"] == "ep00"
