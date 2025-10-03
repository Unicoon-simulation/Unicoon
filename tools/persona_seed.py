#!/usr/bin/env python3
"""Generate draft personas aligned with mapped graph nodes.

This stage avoids any LLM dependency. The output can be refined later by
`data/refine_profiles.py` or other tooling.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping

AGENT_PROMPT_TEMPLATE = (
    "Imagine you are a human. Your name is {name}, and your gender is {gender}. "
    "You are {age} years old. Your personality is shaped by these specific traits: {traits_str}. "
    "Your educational background is at the level of {education_level}. "
    "Your political affiliation is {political_affiliation}. "
    "Act according to this human identity, letting these details fully define your thoughts, "
    "responses, interactions, and decisions."
)

DEFAULT_POLITICAL_DISTRIBUTION: Dict[str, float] = {
    "far_left": 0.15,
    "left": 0.25,
    "moderate": 0.20,
    "right": 0.25,
    "far_right": 0.15,
}

EDUCATION_LEVELS = ["High School", "Bachelor's Degree", "Master's Degree", "PhD"]
GENDERS = ["male", "female", "nonbinary"]
TRAIT_POOL = [
    "Curiosity",
    "Empathy",
    "Resilience",
    "Pragmatism",
    "Idealism",
    "Assertiveness",
    "Patience",
    "Creativity",
    "Discipline",
    "Humor",
    "Introspection",
    "Optimism",
    "Analytical thinking",
    "Community focus",
    "Environmental concern",
    "Traditionalism",
    "Skepticism",
    "Entrepreneurial mindset",
    "Data-driven",
    "Storytelling",
]

POLITICAL_SELF_AWARENESS: Dict[str, List[str]] = {
    "far_left": [
        "I organise online actions to push bold reforms and keep pressure on institutions.",
        "I amplify grassroots campaigns and challenge corporate narratives whenever I post.",
    ],
    "left": [
        "I promote inclusive policies and highlight stories about social justice progress.",
        "I fact-check before debating and try to humanise policy impacts in every thread.",
    ],
    "moderate": [
        "I look for balanced evidence and try to cool down heated comment sections.",
        "I compare multiple sources before weighing in and encourage compromise.",
    ],
    "right": [
        "I emphasise individual responsibility and share data about fiscal discipline online.",
        "I defend traditional institutions while engaging respectfully with differing views.",
    ],
    "far_right": [
        "I rally supporters around sovereignty, cultural heritage, and law-and-order priorities.",
        "I track news for threats to national strength and call out globalist rhetoric quickly.",
    ],
}

AGE_BUCKETS = {
    "young": (21, 34),
    "mid": (35, 54),
    "senior": (55, 72),
}

RNG = random.Random()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate seed personas for mapped nodes")
    parser.add_argument("--mapping", type=Path, help="Path to node mapping JSON (original->mapped)")
    parser.add_argument("--count", type=int, help="Fallback persona count if mapping is omitted")
    parser.add_argument("--output", type=Path, required=True, help="Output persona JSON file")
    parser.add_argument(
        "--political-config",
        type=Path,
        help="Optional JSON file with political distribution, e.g. {\"left\":0.3,...}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for deterministic generation",
    )
    parser.add_argument(
        "--name-prefix",
        default="Agent",
        help="Prefix used when synthesising persona names (default: Agent)",
    )
    return parser.parse_args()


def _mapping_sort_key(value: str) -> tuple[int, int | str]:
    if value.isdigit():
        return (0, int(value))
    return (1, value)


def load_mapping(mapping_path: Path | None) -> List[str]:
    if mapping_path is None:
        return []
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("mapping JSON must be an object")
    mapped_ids = sorted({str(v) for v in data.values()}, key=_mapping_sort_key)
    return mapped_ids


def load_distribution(path: Path | None) -> Dict[str, float]:
    if path is None:
        return DEFAULT_POLITICAL_DISTRIBUTION.copy()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("political-config must be a JSON object")
    dist: Dict[str, float] = {}
    total = 0.0
    for key, value in payload.items():
        if key not in DEFAULT_POLITICAL_DISTRIBUTION:
            raise ValueError(f"Unknown political affiliation: {key}")
        try:
            weight = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid weight for {key}: {value}") from exc
        if weight <= 0:
            continue
        dist[key] = weight
        total += weight
    if not dist:
        raise ValueError("political-config yielded no positive weights")
    for key in list(dist.keys()):
        dist[key] /= total
    return dist


def weighted_choices(distribution: Mapping[str, float], count: int) -> List[str]:
    labels = list(distribution.keys())
    weights = [distribution[label] for label in labels]
    return RNG.choices(labels, weights=weights, k=count)


def pick_traits() -> List[str]:
    traits = RNG.sample(TRAIT_POOL, k=5)
    return traits


def pick_age() -> int:
    bucket = RNG.choice(list(AGE_BUCKETS.values()))
    return RNG.randint(bucket[0], bucket[1])


def build_persona(agent_id: str, prefix: str, affiliation: str) -> Dict[str, object]:
    name = f"{prefix}_{agent_id}"
    gender = RNG.choice(GENDERS)
    education = RNG.choice(EDUCATION_LEVELS)
    traits = pick_traits()
    age = pick_age()
    self_awareness_options = POLITICAL_SELF_AWARENESS.get(affiliation, [
        "I keep up with current events and share sources that match my outlook.",
        "I discuss politics online but stay respectful of different perspectives.",
    ])
    self_awareness = RNG.choice(self_awareness_options)

    persona = {
        "name": name,
        "age": age,
        "education level": education,
        "traits": traits,
        "gender": gender,
        "political_affiliation": affiliation,
        "self-awareness": self_awareness,
    }
    persona["system_prompt"] = AGENT_PROMPT_TEMPLATE.format(
        name=name,
        gender=gender,
        age=age,
        traits_str=", ".join(traits),
        education_level=education,
        political_affiliation=affiliation,
    )
    return persona


def generate_personas(agent_ids: Iterable[str], distribution: Mapping[str, float], prefix: str) -> Dict[str, Dict[str, object]]:
    agent_list = list(agent_ids)
    assignments = weighted_choices(distribution, len(agent_list))
    result: Dict[str, Dict[str, object]] = {}
    for agent_id, affiliation in zip(agent_list, assignments):
        result[agent_id] = build_persona(agent_id, prefix, affiliation)
    return result


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        RNG.seed(args.seed)

    mapped_ids = load_mapping(args.mapping)
    if mapped_ids:
        agent_ids = mapped_ids
    else:
        if args.count is None or args.count <= 0:
            raise ValueError("--count must be provided when --mapping is omitted")
        agent_ids = [str(i) for i in range(args.count)]

    distribution = load_distribution(args.political_config)
    personas = generate_personas(agent_ids, distribution, args.name_prefix)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump({"backgrounds": personas}, fh, ensure_ascii=False, indent=2)

    print(json.dumps({"personas": len(personas), "political": distribution}))


if __name__ == "__main__":
    main()
