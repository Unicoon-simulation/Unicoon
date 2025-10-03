#!/usr/bin/env python3
"""Classify news items into policy issue buckets using a local LLM."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, Sequence, TextIO

import httpx

# 将项目根目录加入到 sys.path，便于复用现有模块
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from servellm import call_openai_api, extract_json_from_response, format_messages

DEFAULT_TOPICS: Sequence[str] = (
    "Climate change and environmental policy",
    "Healthcare access and public health",
    "Economic inequality and taxation",
    "Immigration and border policy",
    "Education reform and student debt",
    "Technology regulation and data privacy",
    "National security and foreign policy",
    "Criminal justice and policing",
    "LGBTQ+ rights and civil liberties",
)

SYSTEM_PROMPT = (
    "You are a meticulous policy analyst. Classify short news briefs into public policy issue "
    "areas. Always return strict JSON without commentary."
)

RESPONSE_SCHEMA = (
    "Return JSON matching this schema:\n"
    "{\n"
    "  \"classifications\": [\n"
    "    {\n"
    "      \"news_id\": \"<string>\",\n"
    "      \"issue_category\": \"<one of the provided issue areas or None>\",\n"
    "      \"confidence\": <float between 0 and 1>\n"
    "    }\n"
    "  ]\n"
    "}\n"
    "Never invent identifiers."
)

ISSUE_FIELD = "issue_category"


def parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Filter combined news into predefined policy topics using a local LLM."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/combined_news.json"),
        help="Input JSON file containing a list of news items (default: data/combined_news.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/policy_topics"),
        help="Directory where per-topic JSON files will be stored (default: data/policy_topics).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of news items per LLM request (default: 8).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the classification model (default: 0.0).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Max tokens allowed in each response (default: 1024).",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BASE_URL", "http://localhost:8001/v1"),
        help="OpenAI-compatible base URL (default: env LLM_BASE_URL or http://localhost:8001/v1).",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LLM_API_KEY", "EMPTY"),
        help="API key for the local model (default: env LLM_API_KEY or EMPTY).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL"),
        help="Model identifier exposed by the local endpoint; auto-detected if omitted.",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        help="Explicit policy topics. If omitted, the built-in nine-issue list is used.",
    )
    parser.add_argument(
        "--topics-file",
        type=Path,
        help="Path to a JSON file containing a list of policy topics.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing topic files instead of aborting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first batch prompt without calling the LLM.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def resolve_topics(args: argparse.Namespace) -> list[str]:
    """Resolve the active issue list, falling back to defaults."""
    if args.topics:
        topics = [topic.strip() for topic in args.topics if topic.strip()]
        if not topics:
            raise ValueError("--topics provided but empty after stripping.")
        return topics

    if args.topics_file:
        raw = args.topics_file.read_text(encoding="utf-8")
        payload = json.loads(raw)
        if not isinstance(payload, list):
            raise ValueError("--topics-file must point to a JSON list.")
        topics = [str(topic).strip() for topic in payload if str(topic).strip()]
        if not topics:
            raise ValueError("--topics-file did not contain usable topic names.")
        return topics

    return list(DEFAULT_TOPICS)


def resolve_model_name(args: argparse.Namespace, logger: logging.Logger) -> str:
    """Resolve the model identifier, querying the LLM endpoint when needed."""
    if args.model:
        model_id = str(args.model).strip()
        if model_id:
            logger.debug("Using model from CLI or environment: %s", model_id)
            return model_id

    url = f"{args.base_url.rstrip('/')}/models"
    logger.debug("Fetching model list from %s", url)

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.RequestError as exc:  # noqa: BLE001
        raise ValueError(f"Failed to fetch models from {url}: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:  # noqa: BLE001
        raise ValueError("LLM /models endpoint did not return JSON.") from exc

    models = payload.get("data")
    if not isinstance(models, list) or not models:
        raise ValueError("LLM /models endpoint returned no models.")

    first = models[0]
    if not isinstance(first, dict) or "id" not in first:
        raise ValueError("Unexpected /models payload structure: missing 'id'.")

    model_id = str(first["id"]).strip()
    if not model_id:
        raise ValueError("Retrieved model id is empty.")

    logger.info("Auto-selected model: %s", model_id)
    return model_id


def slugify(text: str) -> str:
    """Convert a category name into a filesystem-friendly slug."""
    lower = text.lower()
    lower = re.sub(r"[^a-z0-9]+", "_", lower)
    return lower.strip("_") or "topic"


def trim_content(text: str, limit: int = 800) -> str:
    """Trim overly long news content for prompt readability."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit]}..."


class JsonArraySink:
    """Stream JSON array output without buffering all records."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: TextIO | None = None
        self._count = 0

    def write(self, item: Dict[str, str]) -> None:
        handle = self._handle
        if handle is None:
            handle = self.path.open("w", encoding="utf-8")
            handle.write("[\n")
            self._handle = handle
        else:
            handle.write(",\n")

        handle.write("  ")
        handle.write(json.dumps(item, ensure_ascii=True))
        self._count += 1

    def close(self) -> int:
        if self._handle is None:
            self.path.write_text("[]\n", encoding="utf-8")
            return 0

        self._handle.write("\n]\n")
        self._handle.close()
        count = self._count
        self._handle = None
        return count


def build_batch_prompt(batch: Sequence[Dict[str, str]], topics: Sequence[str]) -> str:
    """Assemble the classification prompt for the current batch."""
    topic_lines = "\n".join(f"- {topic}" for topic in topics)
    article_chunks = []
    for idx, item in enumerate(batch, start=1):
        article_chunks.append(
            (
                f"{idx}. news_id: {item['news_id']}\n"
                f"   topic_label: {item.get('topic', 'UNKNOWN')}\n"
                f"   content: {trim_content(item['content'])}"
            )
        )
    articles = "\n\n".join(article_chunks)
    return (
        "Classify each article into exactly one of the listed policy issue areas. "
        "Respond with None only when no issue fits."
        f"\n\nIssue areas:\n{topic_lines}\n\nArticles:\n{articles}\n\n"
        f"{RESPONSE_SCHEMA}"
        " Keep answers terse."
    )


async def classify_batch(
    batch: Sequence[Dict[str, str]],
    topics: Sequence[str],
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, str]:
    """Invoke the LLM for one batch and return news_id to category mapping."""
    messages = format_messages(
        build_batch_prompt(batch, topics),
        system_prompt=SYSTEM_PROMPT,
    )
    response = await call_openai_api(
        messages=messages,
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if not response:
        return {}

    logger = logging.getLogger("policy_filter")

    payload = extract_json_from_response(response)
    if payload is None:
        return {}

    if isinstance(payload, list):
        classifications = payload
    elif isinstance(payload, dict):
        if ISSUE_FIELD in payload and "news_id" in payload:
            classifications = [payload]
        else:
            classifications = payload.get("classifications")
    else:
        logger.debug("Unexpected payload type from LLM: %s", type(payload).__name__)
        return {}

    if not isinstance(classifications, list):
        logger.debug("LLM payload did not include a classifications list")
        return {}

    result: Dict[str, str] = {}
    for entry in classifications:
        if not isinstance(entry, dict):
            continue
        news_id = str(entry.get("news_id", "")).strip()
        category = str(entry.get(ISSUE_FIELD, "")).strip()
        if not news_id or not category:
            continue
        result[news_id] = category
    return result


def enrich_news(
    news_item: Dict[str, str],
    category: str,
) -> Dict[str, str]:
    """Attach the policy issue metadata to a news item."""
    enriched = dict(news_item)
    enriched[ISSUE_FIELD] = category
    return enriched


async def run(args: argparse.Namespace) -> None:
    """Main async entry for the classification workflow."""
    logger = logging.getLogger("policy_filter")

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with args.input.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of news objects.")

    topics = resolve_topics(args)
    topic_set = {topic.lower(): topic for topic in topics}

    model_name = resolve_model_name(args, logger)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {topic: args.output_dir / f"{slugify(topic)}.json" for topic in topics}
    if not args.overwrite:
        clashes = [str(path) for path in output_paths.values() if path.exists()]
        if clashes:
            raise FileExistsError(
                "Output files exist: " + ", ".join(clashes) + ". Pass --overwrite to replace them."
            )

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    total_items = len(data)
    total_batches = (total_items + args.batch_size - 1) // args.batch_size
    logger.info("Loaded %d news items (%d batches)", total_items, total_batches)

    if args.dry_run and total_items:
        preview = build_batch_prompt(data[: args.batch_size], topics)
        print(preview)
        return

    sinks = {topic: JsonArraySink(path) for topic, path in output_paths.items()}

    try:
        for index, start in enumerate(range(0, total_items, args.batch_size), start=1):
            batch = data[start : start + args.batch_size]
            mapping = await classify_batch(
                batch,
                topics,
                base_url=args.base_url,
                api_key=args.api_key,
                model=model_name,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            logger.debug("Batch %d produced %d labels", index, len(mapping))

            for item in batch:
                predicted = mapping.get(item["news_id"])
                if not predicted:
                    continue
                normalized = topic_set.get(predicted.lower())
                if not normalized:
                    continue
                sinks[normalized].write(enrich_news(item, normalized))

            if index % 10 == 0 or index == total_batches:
                logger.info("Processed %d/%d batches", index, total_batches)
    finally:
        for topic, sink in sinks.items():
            count = sink.close()
            logger.info("Wrote %d items to %s", count, sink.path)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
