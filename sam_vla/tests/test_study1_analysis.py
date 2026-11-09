"""Study 1 (next.md) Phase 4: exercises study1_analysis's per-episode metric
extraction and the Condition A/B timing-subtraction comparison against
synthetic manifest.json/rollout.npz fixtures -- no real rollout, sim, or
subprocess. Fixture shape mirrors sam_vla/logging/rollout_logger.py's own
round-trip self-test (manifest["steps"][i]["timestamp"]/["uncertainty_event"],
rollout.npz's distances_to_goal)."""

import json
from pathlib import Path

import numpy as np
import pytest

from sam_vla.study1_analysis import (
    analyze_pairs,
    load_episode_metrics,
    summarize,
    write_csv,
)


def _write_fixture_run(
    run_dir: Path,
    timestamps,
    distances,
    uncertainty_events=None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    uncertainty_events = uncertainty_events or [None] * len(timestamps)
    manifest = {
        "start_time": timestamps[0],
        "steps": [
            {"timestamp": ts, "uncertainty_event": ev}
            for ts, ev in zip(timestamps, uncertainty_events)
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    np.savez(run_dir / "rollout.npz", distances_to_goal=np.array(distances, dtype=np.float64))


TS = [
    "2026-08-06T00:00:00+00:00",
    "2026-08-06T00:00:05+00:00",
    "2026-08-06T00:00:10+00:00",
]


def test_load_episode_metrics_timing_and_success(tmp_path):
    run_dir = tmp_path / "ep00_autonomous"
    _write_fixture_run(run_dir, TS, distances=[5.0, 2.0, 0.5])

    metrics = load_episode_metrics(run_dir, success_radius=1.0)

    assert metrics["n_steps"] == 3
    assert metrics["total_episode_time_s"] == pytest.approx(10.0)
    assert metrics["success"] is True
    assert metrics["n_uncertainty_triggers"] == 0
    assert metrics["sum_human_decision_ms"] == 0.0
    assert metrics["model_load_ms"] == 0.0


def test_load_episode_metrics_failure_when_never_close(tmp_path):
    run_dir = tmp_path / "ep00_autonomous"
    _write_fixture_run(run_dir, TS, distances=[8.0, 6.0, 5.0])

    metrics = load_episode_metrics(run_dir, success_radius=1.0)
    assert metrics["success"] is False


def test_load_episode_metrics_all_nan_distances_is_not_success(tmp_path):
    run_dir = tmp_path / "ep00_autonomous"
    _write_fixture_run(run_dir, TS, distances=[float("nan")] * 3)

    metrics = load_episode_metrics(run_dir, success_radius=1.0)
    assert metrics["success"] is False


def test_load_episode_metrics_aggregates_uncertainty_events(tmp_path):
    run_dir = tmp_path / "ep00_human"
    events = [
        None,
        {
            "vlm_inference_ms": 200.0,
            "human_decision_ms": 3000.0,
            "drive_ms": 500.0,
            "model_load_ms": 4000.0,
        },
        {
            "vlm_inference_ms": 150.0,
            "human_decision_ms": 2000.0,
            "drive_ms": 400.0,
            "model_load_ms": 0.0,
        },
    ]
    _write_fixture_run(run_dir, TS, distances=[5.0, 2.0, 0.5], uncertainty_events=events)

    metrics = load_episode_metrics(run_dir, success_radius=1.0)

    assert metrics["n_uncertainty_triggers"] == 2
    assert metrics["sum_vlm_inference_ms"] == pytest.approx(350.0)
    assert metrics["sum_human_decision_ms"] == pytest.approx(5000.0)
    assert metrics["sum_drive_ms"] == pytest.approx(900.0)
    assert metrics["model_load_ms"] == pytest.approx(4000.0)


def _pairs_manifest_for(tmp_path, autonomous_dir, human_dir):
    return {
        "episodes": [
            {
                "id": "ep00",
                "runs": {
                    "autonomous": {"out_dir": str(autonomous_dir), "returncode": 0},
                    "human": {"out_dir": str(human_dir), "returncode": 0},
                },
            }
        ]
    }


def test_analyze_pairs_computes_derived_time_for_human_condition(tmp_path):
    autonomous_dir = tmp_path / "ep00_autonomous"
    human_dir = tmp_path / "ep00_human"
    _write_fixture_run(autonomous_dir, TS, distances=[5.0, 2.0, 0.5])
    _write_fixture_run(
        human_dir,
        TS,
        distances=[5.0, 2.0, 0.5],
        uncertainty_events=[
            None,
            {
                "vlm_inference_ms": 200.0,
                "human_decision_ms": 2000.0,
                "drive_ms": 500.0,
                "model_load_ms": 3000.0,
            },
            None,
        ],
    )
    pairs_manifest = _pairs_manifest_for(tmp_path, autonomous_dir, human_dir)

    rows = analyze_pairs(pairs_manifest, success_radius=1.0)
    by_condition = {row["condition"]: row for row in rows}

    assert by_condition["autonomous"]["vlm_and_driving_only_time_s"] is None
    # total 10s - human_decision 2s - model_load 3s = 5s
    assert by_condition["human"]["vlm_and_driving_only_time_s"] == pytest.approx(5.0)


def test_analyze_pairs_skips_failed_run(tmp_path):
    autonomous_dir = tmp_path / "ep00_autonomous"
    _write_fixture_run(autonomous_dir, TS, distances=[5.0, 2.0, 0.5])
    pairs_manifest = {
        "episodes": [
            {
                "id": "ep00",
                "runs": {
                    "autonomous": {"out_dir": str(autonomous_dir), "returncode": 0},
                    "human": {"out_dir": str(tmp_path / "missing"), "returncode": 1},
                },
            }
        ]
    }

    rows = analyze_pairs(pairs_manifest, success_radius=1.0)
    assert len(rows) == 1
    assert rows[0]["condition"] == "autonomous"


def test_summarize_pairs_only_complete_episodes(tmp_path):
    autonomous_dir = tmp_path / "ep00_autonomous"
    human_dir = tmp_path / "ep00_human"
    _write_fixture_run(autonomous_dir, TS, distances=[5.0, 2.0, 0.5])
    _write_fixture_run(
        human_dir,
        TS,
        distances=[5.0, 2.0, 0.5],
        uncertainty_events=[
            None,
            {
                "vlm_inference_ms": 200.0,
                "human_decision_ms": 2000.0,
                "drive_ms": 500.0,
                "model_load_ms": 3000.0,
            },
            None,
        ],
    )
    rows = analyze_pairs(_pairs_manifest_for(tmp_path, autonomous_dir, human_dir), success_radius=1.0)

    summary = summarize(rows)
    assert summary["n_paired_episodes"] == 1
    assert summary["condition_b_mean_raw_time_s"] == pytest.approx(10.0)
    assert summary["condition_a_mean_derived_time_s"] == pytest.approx(5.0)
    assert summary["condition_a_success_rate"] == pytest.approx(1.0)
    assert summary["a_faster_than_b_count"] == 1


def test_write_csv_round_trips(tmp_path):
    autonomous_dir = tmp_path / "ep00_autonomous"
    human_dir = tmp_path / "ep00_human"
    _write_fixture_run(autonomous_dir, TS, distances=[5.0, 2.0, 0.5])
    _write_fixture_run(human_dir, TS, distances=[5.0, 2.0, 0.5])
    rows = analyze_pairs(_pairs_manifest_for(tmp_path, autonomous_dir, human_dir), success_radius=1.0)

    out_csv = tmp_path / "analysis.csv"
    write_csv(rows, out_csv)

    import csv as csv_module

    with open(out_csv) as f:
        read_rows = list(csv_module.DictReader(f))
    assert len(read_rows) == 2
    assert {r["condition"] for r in read_rows} == {"autonomous", "human"}
