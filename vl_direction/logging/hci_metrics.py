"""
Pure aggregation functions over already-loaded vl_direction directive
records (next.md sec 6.3), independently unit-testable since they take
plain dicts/lists rather than reading files themselves. Success rate and
steps/time-to-goal are supplied by the caller (an orchestrator/belief-system
fact this module never observes) rather than derived from directives.jsonl
alone -- that's an explicit non-goal of this module.
"""

import json
from typing import Any, Dict, Iterable, List


def load_directives_jsonl(path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def intervention_counts(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in records:
        if r.get("session_mode") == "human_intervened":
            token = r.get("identity_token", "unknown")
            counts[token] = counts.get(token, 0) + 1
    return counts


def uncertainty_trigger_stats(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    records = list(records)
    triggers = [
        r
        for r in records
        if r.get("uncertainty_payload")
        and r["uncertainty_payload"].get("status") == "NEEDS_HUMAN_INPUT"
    ]
    resolutions = [
        r
        for r in records
        if r.get("uncertainty_payload")
        and r["uncertainty_payload"].get("status") == "HEADING_DIRECTIVE"
    ]
    avg_retries = (
        sum(r["uncertainty_payload"].get("attempt", 0) for r in resolutions)
        / len(resolutions)
        if resolutions
        else None
    )
    return {"trigger_count": len(triggers), "avg_retries_to_resolution": avg_retries}


def success_rate_by_mode(
    episode_summaries: Iterable[Dict[str, Any]],
) -> Dict[str, float]:
    """episode_summaries: caller-supplied dicts with at least
    {"session_mode": "autonomous"|"human_intervened", "success": bool}."""
    by_mode: Dict[str, List[bool]] = {}
    for ep in episode_summaries:
        by_mode.setdefault(ep["session_mode"], []).append(bool(ep["success"]))
    return {mode: sum(v) / len(v) for mode, v in by_mode.items() if v}


def steps_to_goal_by_mode(
    episode_summaries: Iterable[Dict[str, Any]],
) -> Dict[str, float]:
    """episode_summaries: caller-supplied dicts with at least
    {"session_mode": ..., "steps_to_goal": float}, successful episodes only."""
    by_mode: Dict[str, List[float]] = {}
    for ep in episode_summaries:
        by_mode.setdefault(ep["session_mode"], []).append(ep["steps_to_goal"])
    return {mode: sum(v) / len(v) for mode, v in by_mode.items() if v}


def confidence_vs_override_correlation(
    records: Iterable[Dict[str, Any]], teleop_events: Iterable[Dict[str, Any]]
) -> Dict[str, Any]:
    """Only meaningful when config.SHADOW_LOGGING_ENABLED was on for the run
    (next.md sec 6.3/6.4): pairs each shadow-mode VL confidence with whether
    a human override happened at roughly the same time. teleop_events is a
    caller-supplied list of {"timestamp": iso8601} for human interventions."""
    override_timestamps = {e["timestamp"] for e in teleop_events}
    overridden = [
        r["confidence"] for r in records if r.get("timestamp") in override_timestamps
    ]
    not_overridden = [
        r["confidence"]
        for r in records
        if r.get("timestamp") not in override_timestamps
    ]
    return {
        "avg_confidence_when_overridden": (
            sum(overridden) / len(overridden) if overridden else None
        ),
        "avg_confidence_when_not_overridden": (
            sum(not_overridden) / len(not_overridden) if not_overridden else None
        ),
    }


if __name__ == "__main__":
    demo_records = [
        {
            "identity_token": "cbf",
            "session_mode": "human_intervened",
            "uncertainty_payload": None,
            "confidence": 0.9,
            "timestamp": "t1",
        },
        {
            "identity_token": "exploration",
            "session_mode": "autonomous",
            "uncertainty_payload": None,
            "confidence": 0.4,
            "timestamp": "t2",
        },
        {
            "identity_token": "uncertainty",
            "session_mode": "autonomous",
            "uncertainty_payload": {"status": "NEEDS_HUMAN_INPUT", "attempt": 0},
            "confidence": 1.0,
            "timestamp": "t3",
        },
        {
            "identity_token": "uncertainty",
            "session_mode": "autonomous",
            "uncertainty_payload": {"status": "HEADING_DIRECTIVE", "attempt": 1},
            "confidence": 1.0,
            "timestamp": "t4",
        },
    ]
    print("intervention_counts ->", intervention_counts(demo_records))
    print("uncertainty_trigger_stats ->", uncertainty_trigger_stats(demo_records))
