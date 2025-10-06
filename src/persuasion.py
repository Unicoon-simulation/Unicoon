
from __future__ import annotations

import logging
from typing import Dict, Any, Sequence

from dataclass import AgentSnapshot
from build_prompt import build_persuasion_prompt
from parse import parse_persuasion_response
from servellm import submit, format_messages

logger = logging.getLogger("persuasion")


def _summarize_context(snippets: Sequence[str], limit: int = 3) -> str:
    filtered = [snippet.strip() for snippet in snippets if snippet]
    if not filtered:
        return ""
    filtered = filtered[-limit:]
    return " \n".join(filtered)


async def generate_persuasion_decision(
    agent: AgentSnapshot,
    target: AgentSnapshot,
    topic: str,
    context_snippets: Sequence[str],
    executor,
) -> Dict[str, Any]:
    context_messages = _summarize_context(context_snippets)
    prompt = build_persuasion_prompt(
        agent=agent,
        target=target,
        topic=topic,
        context_messages=context_messages,
    )

    messages = format_messages(prompt, system_prompt=agent.profile.agent_prompt)
    response = await submit(messages, executor, parse_type="json")

    if not response:
        logger.warning(
            "劝说消息生成失败: LLM 无响应 (agent=%s target=%s)",
            agent.agent_id,
            target.agent_id,
        )
        return {"will": "no", "message": ""}

    parsed = parse_persuasion_response(response)
    if parsed.get("status") != "success":
        logger.warning(
            "劝说消息解析失败 (agent=%s target=%s) raw=%s",
            agent.agent_id,
            target.agent_id,
            response,
        )
        return {"will": "no", "message": ""}

    return {
        "will": parsed.get("will", "no"),
        "message": parsed.get("message", ""),
        "raw_response": response,
    }

