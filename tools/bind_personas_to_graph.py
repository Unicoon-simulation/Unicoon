#!/usr/bin/env python3
"""Bind refined personas to mapped graph IDs and emit simulation-ready background JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bind personas to mapped graph ids")
    parser.add_argument("--mapping", type=Path, required=True, help="Mapping JSON original->mapped")
    parser.add_argument("--personas", type=Path, required=True, help="Persona JSON with backgrounds")
    parser.add_argument(
        "--edges",
        type=Path,
        required=True,
        help="Mapped edge list used for validation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output agent background JSON",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Do not fail if some personas are missing; simply skip them",
    )
    return parser.parse_args()


def load_mapping(path: Path) -> Dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("mapping JSON must be an object")
    return {str(k): str(v) for k, v in data.items()}


def load_personas(path: Path) -> Dict[str, Dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or "backgrounds" not in payload:
        raise ValueError("persona file must contain a 'backgrounds' object")
    backgrounds = payload["backgrounds"]
    if not isinstance(backgrounds, Mapping):
        raise ValueError("'backgrounds' must be an object")
    return {str(k): dict(v) for k, v in backgrounds.items()}


def extract_nodes_from_edges(path: Path) -> List[str]:
    nodes: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            nodes.add(parts[0])
            nodes.add(parts[1])
    return sorted(nodes, key=lambda x: int(x) if x.isdigit() else x)


def bind(
    mapping: Mapping[str, str],
    personas: Mapping[str, Dict[str, object]],
    required_ids: Iterable[str],
    *,
    allow_missing: bool,
) -> Tuple[Dict[str, Dict[str, object]], List[str], List[str]]:
    selected_ids = sorted({str(r) for r in required_ids}, key=lambda x: int(x) if x.isdigit() else x)
    missing: List[str] = []
    bound: Dict[str, Dict[str, object]] = {}

    for mapped_id in selected_ids:
        persona = personas.get(mapped_id)
        if persona is None:
            missing.append(mapped_id)
            if allow_missing:
                continue
            raise ValueError(f"Persona for agent {mapped_id} is missing")
        persona = dict(persona)
        persona.setdefault("agent_id", mapped_id)
        bound[mapped_id] = persona

    extras = [pid for pid in personas.keys() if pid not in bound]
    return bound, missing, extras


def main() -> None:
    args = parse_args()
    mapping = load_mapping(args.mapping)
    personas = load_personas(args.personas)
    nodes_from_edges = extract_nodes_from_edges(args.edges)

    mapped_values = {str(v) for v in mapping.values()}
    required_ids = sorted({nid for nid in nodes_from_edges}, key=lambda x: int(x) if x.isdigit() else x)

    missing_in_mapping = [rid for rid in required_ids if rid not in mapped_values]
    if missing_in_mapping:
        raise ValueError(
            "Edge list references node ids that are absent from mapping: "
            + ", ".join(missing_in_mapping[:10])
        )

    bound, missing, extras = bind(
        mapping=mapping,
        personas=personas,
        required_ids=required_ids,
        allow_missing=args.allow_missing,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({"backgrounds": bound}, fh, ensure_ascii=False, indent=2)

    summary = {
        "mapped_personas": len(bound),
        "required_nodes": len(required_ids),
        "extras_available": len(extras),
        "missing": missing,
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
