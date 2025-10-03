#!/usr/bin/env python3
"""Map arbitrary graph node IDs to a contiguous index range for persona generation.

Usage example:
    python tools/map_edges.py \
        --input data/12831.edges \
        --edges-out data/12831_mapped_236.edges \
        --mapping-out data/node_mapping_12831_to_236.json \
        --target-count 236
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map graph node ids to contiguous indices")
    parser.add_argument("--input", type=Path, required=True, help="Source edge list file")
    parser.add_argument("--edges-out", type=Path, required=True, help="Path for mapped edge list")
    parser.add_argument(
        "--mapping-out", type=Path, required=True, help="Output JSON mapping original->mapped"
    )
    parser.add_argument(
        "--target-count",
        type=int,
        help="Optional cap on number of nodes to keep; highest degree nodes are retained",
    )
    parser.add_argument(
        "--skip-self-loop",
        action="store_true",
        help="Drop edges where source == target after mapping (default keeps them)",
    )
    return parser.parse_args()


def load_edges(path: Path) -> List[Tuple[str, str, float, float]]:
    edges: List[Tuple[str, str, float, float]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            src, dst = parts[0], parts[1]
            trust = float(parts[2]) if len(parts) >= 3 else 0.5
            interest = float(parts[3]) if len(parts) >= 4 else 0.5
            edges.append((src, dst, trust, interest))
    return edges


def select_nodes(
    edges: Iterable[Tuple[str, str, float, float]], target_count: int | None
) -> List[str]:
    counter: Counter[str] = Counter()
    for src, dst, *_ in edges:
        counter[src] += 1
        counter[dst] += 1

    def base_order(node: str) -> tuple[int, int | str]:
        if node.isdigit():
            return (0, int(node))
        return (1, node)

    if target_count is None or target_count >= len(counter):
        return sorted(counter.keys(), key=base_order)

    # sort by degree desc, tie break by original value (numeric if possible)
    def sort_key(node: str) -> Tuple[int, str]:
        numeric = node.isdigit()
        norm = f"{int(node):020d}" if numeric else node
        return (-counter[node], norm)

    ordered = sorted(counter.keys(), key=sort_key)
    selected = ordered[:target_count]

    # maintain deterministic order when assigning new ids
    selected_sorted = sorted(selected, key=base_order)
    return selected_sorted


def build_mapping(nodes: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for idx, node in enumerate(nodes):
        mapping[node] = str(idx)
    return mapping


def remap_edges(
    edges: Iterable[Tuple[str, str, float, float]],
    mapping: Dict[str, str],
    *,
    skip_self_loop: bool,
) -> List[Tuple[str, str, float, float]]:
    output: List[Tuple[str, str, float, float]] = []
    allowed = set(mapping.keys())
    for src, dst, trust, interest in edges:
        if src not in allowed or dst not in allowed:
            continue
        mapped_src = mapping[src]
        mapped_dst = mapping[dst]
        if skip_self_loop and mapped_src == mapped_dst:
            continue
        output.append((mapped_src, mapped_dst, trust, interest))
    return output


def dump_edges(path: Path, edges: Iterable[Tuple[str, str, float, float]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for src, dst, trust, interest in edges:
            fh.write(f"{src} {dst} {trust:.6f} {interest:.6f}\n")


def main() -> None:
    args = parse_args()
    edges = load_edges(args.input)
    nodes = select_nodes(edges, args.target_count)
    mapping = build_mapping(nodes)
    mapped_edges = remap_edges(edges, mapping, skip_self_loop=args.skip_self_loop)

    args.edges_out.parent.mkdir(parents=True, exist_ok=True)
    args.mapping_out.parent.mkdir(parents=True, exist_ok=True)

    dump_edges(args.edges_out, mapped_edges)
    with args.mapping_out.open("w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2, ensure_ascii=False)

    coverage = len(mapped_edges) / max(1, len(edges))
    print(
        json.dumps(
            {
                "input_edges": len(edges),
                "output_edges": len(mapped_edges),
                "original_nodes": len({n for edge in edges for n in edge[:2]}),
                "mapped_nodes": len(mapping),
                "edge_coverage": round(coverage, 4),
            }
        )
    )


if __name__ == "__main__":
    main()
