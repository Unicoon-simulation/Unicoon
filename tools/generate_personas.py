#!/usr/bin/env python3
"""可配置讨论议题的智能体角色合成工具。"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx

# 将项目根目录加入到 sys.path，便于复用现有模块
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from servellm import call_openai_api, extract_json_from_response, format_messages

# 默认议题列表，来源于 Social-Bots-Meet-LLM 的常见公共议题
DEFAULT_TOPICS: List[str] = [
    "Climate change and environmental policy",
    "Healthcare access and public health",
    "Economic inequality and taxation",
    "Immigration and border policy",
    "Education reform and student debt",
    "Technology regulation and data privacy",
    "National security and foreign policy",
    "Criminal justice and policing",
    "LGBTQ+ rights and civil liberties",
]

# 生成 agent 提示的模板，保持与现有 dataclass.AgentProfile 一致
AGENT_PROMPT_TEMPLATE = (
    "Imagine you are a human. Your name is {name}, and your gender is {gender}. "
    "You are {age} years old. Your personality is shaped by these specific traits: {traits_str}. "
    "Your educational background is at the level of {education_level}. "
    "Your political affiliation is {political_affiliation}. "
    "Act according to this human identity, letting these details fully define your thoughts, "
    "responses, interactions, and decisions."
)

# 系统提示，约束模型只返回 JSON
SYSTEM_PROMPT = (
    "You are an expert sociologist who specialises in building diverse personas for social "
    "simulation. You must always return compact, strictly valid JSON and never add extra prose."
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Generate agent persona backgrounds via an OpenAI-compatible API."
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of personas to generate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated_profiles.json"),
        help="Output JSON file path (default: data/generated_profiles.json).",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        help="Explicit discussion topics. Overrides --topics-file if both are provided.",
    )
    parser.add_argument(
        "--topics-file",
        type=Path,
        help="Path to a JSON file containing a list of discussion topics.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of personas to request per API call (default: 10).",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BASE_URL", "http://localhost:8001/v1"),
        help="OpenAI-compatible base URL (default: env LLM_BASE_URL or http://localhost:8001/v1).",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="API key (default: env OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL"),
        help="Model name for the API request. If omitted, the script will query the base URL "
        "and pick the first available model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for persona generation (default: 0.7).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens for each API response (default: 2048).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved prompt without calling the API.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def resolve_topics(args: argparse.Namespace) -> List[str]:
    """根据命令行输入返回有序的议题列表。"""
    if args.topics:
        topics = [topic.strip() for topic in args.topics if topic.strip()]
        if not topics:
            raise ValueError("--topics provided but no valid topic strings found.")
        return topics

    if args.topics_file:
        text = args.topics_file.read_text(encoding="utf-8")
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("--topics-file must contain a JSON list of strings.")
        topics = [str(topic).strip() for topic in data if str(topic).strip()]
        if not topics:
            raise ValueError("--topics-file did not provide any usable topics.")
        return topics

    return DEFAULT_TOPICS


def ensure_ascii(text: str) -> str:
    """通过替换不支持的字符将文本强制转换为ASCII。"""
    return text.encode("ascii", errors="ignore").decode("ascii")


def resolve_model_name(base_url: str, model_arg: Optional[str]) -> str:
    """解析模型名称，可选地查询LLM服务器。"""
    if model_arg:
        return model_arg

    logger = logging.getLogger("persona")
    url = f"{base_url.rstrip('/')}/models"
    logger.debug("Fetching model list from %s", url)

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.RequestError as exc:
        raise ValueError(f"Failed to fetch models from {url}: {exc}") from exc

    try:
        payload: Dict[str, Any] = response.json()
    except ValueError as exc:
        raise ValueError("LLM /models endpoint did not return JSON.") from exc

    models = payload.get("data")
    if not isinstance(models, list) or not models:
        raise ValueError("No models available from the LLM server.")

    first_model = models[0]
    if not isinstance(first_model, dict) or "id" not in first_model:
        raise ValueError("Unexpected /models payload structure: missing 'id'.")

    model_id = str(first_model["id"]).strip()
    if not model_id:
        raise ValueError("Retrieved model id is empty.")

    logger.info("Auto-selected model: %s", model_id)
    return model_id


def build_persona_prompt(
    count: int,
    topics: Iterable[str],
    start_index: int,
) -> str:
    """构建角色生成提示内容。"""
    topic_lines = "\n".join(f"- {topic}" for topic in topics)
    index_preview = ", ".join(str(start_index + idx) for idx in range(count))

    schema = (
        "{\n"
        "  \"backgrounds\": {\n"
        "    \"<agent_id>\": {\n"
        "      \"name\": \"<unique ASCII name>\",\n"
        "      \"age\": <integer 18-50>,\n"
        "      \"education level\": \"<education level>\",\n"
        "      \"traits\": [\"personality trait\", ...],\n"
        "      \"gender\": \"male|female|nonbinary\",\n"
        "      \"political_affiliation\": \"far_left|left|moderate|right|far_right\",\n"
        "      \"discussion_topics\": [\"topic from provided list\", ...],\n"
        "      \"topic_positions\": {\"topic\": \"concise stance\", ...},\n"
        "      \"communication_style\": \"short summary of how they debate online\",\n"
        "      \"self-awareness\": \"first-person sentence about their posting habit\",\n"
        "      \"system_prompt\": \"instructions persona will follow\"\n"
        "    }\n"
        "  }\n"
        "}\n"
    )

    instructions = (
        f"Create {count} distinct personas for a multi-agent political discussion. "
        f"Use the agent IDs {index_preview} as strings. Each agent's discussion_topics must be "
        "a non-empty subset of the provided topics, with 2-4 entries. topic_positions must only "
        "contain keys drawn from those discussion_topics. Traits should be diverse mixtures of "
        "Big Five descriptors and civic attitudes. Ensure ages, genders, education levels, and "
        "political affiliations are varied across the batch. Keep every string strictly ASCII."
    )

    return (
        f"{instructions}\n\n"
        f"Provided topics:\n{topic_lines}\n\n"
        "Return exactly one JSON object that matches this schema (field order does not matter):\n"
        f"{schema}\n"
        "Fill system_prompt with a concise instruction set the persona would follow in English."
    )


def merge_backgrounds(
    accumulator: Dict[str, Dict[str, Any]],
    payload: Dict[str, Any],
) -> None:
    """将载荷中的角色记录合并到累加器中，并进行偏移调整。"""
    backgrounds = payload.get("backgrounds")
    if not isinstance(backgrounds, dict):
        raise ValueError("LLM response missing 'backgrounds' dictionary.")

    for raw_key, persona in backgrounds.items():
        if not isinstance(persona, dict):
            raise ValueError("Persona entry must be a JSON object.")

        try:
            idx = int(raw_key)
        except (TypeError, ValueError) as exc:  # noqa: PERF203
            raise ValueError(f"Persona key '{raw_key}' is not an integer string.") from exc

        final_key = str(idx)
        if final_key in accumulator:
            raise ValueError(f"Duplicate persona id detected: {final_key}")

        cleaned_persona = normalize_persona(persona)
        accumulator[final_key] = cleaned_persona


def normalize_persona(persona: Dict[str, Any]) -> Dict[str, Any]:
    """清理和后处理角色字段以满足模拟器要求。"""
    required_fields = [
        "name",
        "age",
        "education level",
        "traits",
        "gender",
        "political_affiliation",
        "self-awareness",
    ]

    for field in required_fields:
        if field not in persona:
            raise ValueError(f"Persona missing required field: {field}")

    persona["name"] = ensure_ascii(str(persona["name"]).strip())
    persona["gender"] = ensure_ascii(str(persona["gender"]).strip().lower())
    persona["political_affiliation"] = ensure_ascii(
        str(persona["political_affiliation"]).strip().lower()
    )
    persona["education level"] = ensure_ascii(str(persona["education level"]).strip())
    persona["self-awareness"] = ensure_ascii(str(persona["self-awareness"]).strip())

    # traits 字段统一为字符串列表，移除空内容
    traits = persona.get("traits", [])
    if isinstance(traits, str):
        traits = [traits]
    if not isinstance(traits, list):
        raise ValueError("Persona 'traits' must be a list or string.")
    persona["traits"] = [ensure_ascii(str(trait).strip()) for trait in traits if str(trait).strip()]
    if not persona["traits"]:
        raise ValueError("Persona must include at least one personality trait.")

    # 讨论议题最少一个
    discussion_topics = persona.get("discussion_topics", [])
    if isinstance(discussion_topics, str):
        discussion_topics = [discussion_topics]
    persona["discussion_topics"] = [
        ensure_ascii(str(topic).strip())
        for topic in discussion_topics
        if str(topic).strip()
    ]

    if not persona["discussion_topics"]:
        raise ValueError("Persona must list at least one discussion topic.")

    # topic_positions 仅保留 discussion_topics 内的键
    topic_positions = persona.get("topic_positions", {})
    if isinstance(topic_positions, list):
        topic_positions = {
            ensure_ascii(str(item.get("topic", ""))): ensure_ascii(str(item.get("stance", "")))
            for item in topic_positions
            if isinstance(item, dict)
        }
    if not isinstance(topic_positions, dict):
        topic_positions = {}
    persona["topic_positions"] = {
        ensure_ascii(str(topic)): ensure_ascii(str(topic_positions.get(topic, "")).strip())
        for topic in persona["discussion_topics"]
        if str(topic_positions.get(topic, "")).strip()
    }

    # 若缺少 communication_style 则补充默认描述
    if not persona.get("communication_style"):
        persona["communication_style"] = "Prefers balanced, source-backed arguments."
    else:
        persona["communication_style"] = ensure_ascii(str(persona["communication_style"]).strip())

    # system_prompt 若缺失则按模板生成
    if not persona.get("system_prompt"):
        persona["system_prompt"] = AGENT_PROMPT_TEMPLATE.format(
            name=persona["name"],
            gender=persona["gender"],
            age=persona["age"],
            traits_str=", ".join(persona["traits"]),
            education_level=persona["education level"],
            political_affiliation=persona["political_affiliation"],
        )
    else:
        persona["system_prompt"] = ensure_ascii(str(persona["system_prompt"]).strip())

    return persona


async def request_batch(
    count: int,
    topics: List[str],
    start_index: int,
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """调用LLM一次合成一批角色。"""
    prompt = build_persona_prompt(count=count, topics=topics, start_index=start_index)
    messages = format_messages(prompt, role="user", system_prompt=SYSTEM_PROMPT)

    response = await call_openai_api(
        messages=messages,
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if response is None:
        raise RuntimeError("LLM returned no response.")

    payload = extract_json_from_response(response)
    if payload is None:
        raise ValueError("Failed to parse JSON from LLM response.")

    return payload


async def generate_personas(
    args: argparse.Namespace,
    model_name: str,
) -> Dict[str, Dict[str, Any]]:
    """批量生成角色并返回合并的字典。"""
    topics = resolve_topics(args)
    batch_size = max(1, min(args.batch_size, args.count))

    logging.getLogger("persona").info(
        "Using %d topics, requesting %d personas (%d per batch) via model %s",
        len(topics),
        args.count,
        batch_size,
        model_name,
    )

    if args.dry_run:
        preview_prompt = build_persona_prompt(batch_size, topics, 0)
        print(preview_prompt)
        return {}

    if not args.api_key:
        raise ValueError("API key is empty. Set --api-key or OPENAI_API_KEY.")

    merged_backgrounds: Dict[str, Dict[str, Any]] = {}

    remaining = args.count
    start_idx = 0
    while remaining > 0:
        current_batch = min(batch_size, remaining)
        payload = await request_batch(
            count=current_batch,
            topics=topics,
            start_index=start_idx,
            base_url=args.base_url,
            api_key=args.api_key,
            model=model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        merge_backgrounds(merged_backgrounds, payload)

        start_idx += current_batch
        remaining -= current_batch

    if len(merged_backgrounds) != args.count:
        raise ValueError(
            f"Expected {args.count} personas, but received {len(merged_backgrounds)}."
        )

    return merged_backgrounds


def write_output(path: Path, backgrounds: Dict[str, Dict[str, Any]], overwrite: bool) -> None:
    """将角色字典持久化到JSON文件。"""
    if backgrounds and path.exists() and not overwrite:
        raise FileExistsError(f"Output file {path} already exists. Use --overwrite to replace it.")

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"backgrounds": backgrounds}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="ascii")


def configure_logging(verbose: bool) -> None:
    """根据详细程度标志配置日志。"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def main() -> None:
    """命令行工具的入口点。"""
    args = parse_args()
    configure_logging(args.verbose)

    try:
        model_name = resolve_model_name(args.base_url, args.model)
    except ValueError as exc:
        logging.getLogger("persona").error("Model resolution failed: %s", exc)
        raise SystemExit(1) from exc

    try:
        backgrounds = asyncio.run(generate_personas(args, model_name))
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("persona").error("Generation failed: %s", exc)
        raise SystemExit(1) from exc

    if args.dry_run:
        return

    write_output(args.output, backgrounds, args.overwrite)
    logging.getLogger("persona").info(
        "Wrote %d personas to %s", len(backgrounds), args.output.as_posix()
    )


if __name__ == "__main__":
    main()
