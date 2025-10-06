
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from dataclass import AgentSnapshot

AFFILIATION_SCALE: Dict[str, int] = {
    "far left": 2,
    "left": 1,
    "moderate": 0,
    "right": -1,
    "far right": -2,
}


def capture_affiliation_snapshot(agents: Dict[str, AgentSnapshot]) -> Dict[str, str]:
    return {
        agent_id: agent.profile.political_affiliation
        for agent_id, agent in agents.items()
    }


def _affiliation_to_score(affiliation: str) -> int:
    return AFFILIATION_SCALE.get(affiliation.lower(), 0) if affiliation else 0


def summarize_political_shift(
    baseline: Dict[str, str],
    agents: Dict[str, AgentSnapshot],
    round_id: int,
    topic: str | None = None,
) -> Dict[str, Any]:
    per_agent = []
    shift_left = 0
    shift_right = 0
    changed = 0
    total_delta = 0
    max_shift = (None, 0)

    transition_counts: Dict[str, int] = {}
    initial_bucket: Dict[str, int] = {}

    for agent_id, before in baseline.items():
        after = agents.get(agent_id)
        if after is None:
            continue
        after_aff = after.profile.political_affiliation

        before_score = _affiliation_to_score(before)
        after_score = _affiliation_to_score(after_aff)
        delta = after_score - before_score

        bucket_key = str(before_score)
        initial_bucket[bucket_key] = initial_bucket.get(bucket_key, 0) + 1
        transition_key = f"{before_score}->{after_score}"
        transition_counts[transition_key] = transition_counts.get(transition_key, 0) + 1

        if delta != 0:
            changed += 1
            total_delta += delta
            direction = "right" if delta > 0 else "left"
            if delta > 0:
                shift_right += 1
            else:
                shift_left += 1
            if abs(delta) > abs(max_shift[1]):
                max_shift = (agent_id, delta)
        else:
            direction = "stable"

        per_agent.append(
            {
                "agent_id": agent_id,
                "before": before,
                "after": after_aff,
                "delta": delta,
                "direction": direction,
            }
        )

    avg_delta = total_delta / changed if changed else 0.0
    aggregates = {
        "changed": changed,
        "shift_left": shift_left,
        "shift_right": shift_right,
        "average_delta": avg_delta,
        "max_shift": {
            "agent_id": max_shift[0],
            "delta": max_shift[1],
        },
    }

    return {
        "round": round_id,
        "scale": AFFILIATION_SCALE,
        "topic": topic,
        "per_agent": per_agent,
        "aggregates": aggregates,
        "transitions": transition_counts,
        "initial_counts": initial_bucket,
    }


def dump_round_report(round_dir: Path, summary: Dict[str, Any]) -> Path:
    round_id = summary["round"]
    report_path = round_dir / f"round_{round_id}_politics.json"
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    return report_path
