from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from parse import normalize_political_affiliation

_ROUND_DIR_PATTERN = re.compile(r"^round_(\d+)$")


def discover_round_dirs(run_dir: Path | str) -> Dict[int, Path]:
    base = Path(run_dir)
    if not base.exists():
        raise FileNotFoundError(f"Run directory not found: {base}")

    candidates: Dict[int, Path] = {}
    for item in base.iterdir():
        if not item.is_dir():
            continue
        match = _ROUND_DIR_PATTERN.match(item.name)
        if not match:
            continue
        candidates[int(match.group(1))] = item

    return dict(sorted(candidates.items()))


def _load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_affiliation(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = normalize_political_affiliation(value)
    if normalized:
        return normalized
    return value.strip().lower()


def load_round_profiles(run_dir: Path | str) -> Dict[int, Dict[str, Dict[str, Any]]]:
    profiles_by_round: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for round_idx, round_path in discover_round_dirs(run_dir).items():
        profile_path = round_path / f"round_{round_idx}_profile.json"
        snapshot = _load_json(profile_path)
        if snapshot is None:
            continue
        profiles_by_round[round_idx] = snapshot
    return profiles_by_round


def summarize_affiliations_by_round(run_dir: Path | str) -> Dict[int, Dict[str, int]]:
    summary: Dict[int, Dict[str, int]] = {}
    for round_idx, profiles in load_round_profiles(run_dir).items():
        counter: Counter[str] = Counter()
        for record in profiles.values():
            affiliation = _coerce_affiliation(record.get("political_affiliation"))
            if affiliation:
                counter[affiliation] += 1
        summary[round_idx] = dict(counter)
    return summary


def extract_affiliation_timeline(run_dir: Path | str) -> Dict[str, List[Tuple[int, Optional[str]]]]:
    timeline: Dict[str, List[Tuple[int, Optional[str]]]] = defaultdict(list)
    for round_idx, profiles in load_round_profiles(run_dir).items():
        for agent_id, record in profiles.items():
            timeline[agent_id].append((round_idx, _coerce_affiliation(record.get("political_affiliation"))))
    return dict(timeline)


def detect_affiliation_changes(run_dir: Path | str) -> List[Dict[str, Any]]:
    changes: List[Dict[str, Any]] = []
    for agent_id, entries in extract_affiliation_timeline(run_dir).items():
        previous_affiliation: Optional[str] = None
        previous_round: Optional[int] = None
        for round_idx, affiliation in sorted(entries, key=lambda item: item[0]):
            if previous_affiliation is None:
                previous_affiliation = affiliation
                previous_round = round_idx
                continue
            if affiliation != previous_affiliation:
                changes.append(
                    {
                        "agent_id": agent_id,
                        "from": previous_affiliation,
                        "to": affiliation,
                        "round": round_idx,
                        "previous_round": previous_round,
                        "delta_rounds": None if previous_round is None else round_idx - previous_round,
                    }
                )
                previous_affiliation = affiliation
                previous_round = round_idx
    return changes


def extract_propagation_edges(run_dir: Path | str) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    for round_idx, round_path in discover_round_dirs(run_dir).items():
        propagation_path = round_path / f"round_{round_idx}_propagation.json"
        payload = _load_json(propagation_path)
        if not isinstance(payload, dict):
            continue
        for layer in payload.get("layers", []):
            layer_idx = layer.get("layer")
            for node in layer.get("nodes", []):
                source_id = node.get("agent_id")
                deliveries = node.get("delivered", []) or []
                for delivery in deliveries:
                    target_id = delivery.get("follower_id")
                    for news_item in delivery.get("news", []) or []:
                        edges.append(
                            {
                                "round": round_idx,
                                "layer": layer_idx,
                                "source": source_id,
                                "target": target_id,
                                "news_id": news_item.get("news_id"),
                                "topic": news_item.get("topic"),
                                "sender": news_item.get("sender"),
                                "content": news_item.get("content"),
                            }
                        )
    return edges


def build_news_propagation_map(run_dir: Path | str) -> Dict[str, Dict[str, Any]]:
    news_map: Dict[str, Dict[str, Any]] = {}
    for edge in extract_propagation_edges(run_dir):
        news_id = edge.get("news_id")
        if not news_id:
            continue
        bucket = news_map.setdefault(
            news_id,
            {
                "topic": edge.get("topic"),
                "transmissions": [],
                "agents_involved": set(),
                "rounds": set(),
            },
        )
        bucket["transmissions"].append(edge)
        bucket["agents_involved"].update(filter(None, [edge.get("source"), edge.get("target")]))
        bucket["rounds"].add(edge["round"])
        if bucket.get("topic") is None and edge.get("topic"):
            bucket["topic"] = edge.get("topic")

    for news_id, bucket in news_map.items():
        bucket["agents_involved"] = sorted(bucket["agents_involved"])
        bucket["rounds"] = sorted(bucket["rounds"])

    return news_map


def compute_news_reach(news_map: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    reach: Dict[str, int] = {}
    for news_id, bucket in news_map.items():
        unique_targets = {edge.get("target") for edge in bucket.get("transmissions", []) if edge.get("target")}
        reach[news_id] = len(unique_targets)
    return reach


def iter_round_files(run_dir: Path | str, suffix: str) -> Iterable[Tuple[int, Path]]:
    for round_idx, round_path in discover_round_dirs(run_dir).items():
        candidate = round_path / f"round_{round_idx}_{suffix}"
        if candidate.exists():
            yield round_idx, candidate
