import asyncio
import json
import logging
import os
import random
from typing import List, Dict, Any, Optional

import httpx

# 请求总超时时间（秒）
REQUEST_TIMEOUT_SECONDS = 1000.0
MAX_RETRIES = 3
BACKOFF_BASE = 1.5
BACKOFF_CAP = 10.0
JITTER_RANGE = (0.0, 0.5)

REQUEST_TIMEOUT = httpx.Timeout(REQUEST_TIMEOUT_SECONDS)


def _compute_backoff(attempt: int) -> float:
    """计算带抖动的退避等待时间。"""
    wait = min(BACKOFF_BASE ** attempt, BACKOFF_CAP)
    return wait + random.uniform(*JITTER_RANGE)


async def submit(
    messages: List[Dict[str, str]],
    executor,
    parse_type: str = "json",
    **kwargs
) -> Any:
    """
    提交消息到LLM服务，生成失败时自动重试。
    Args:
        messages: 待提交的消息列表，格式为[{"role": "user", "content": "..."}]
        executor: LLM执行器实例
        parse_type: 解析类型，用于调用不同的解析方法
        **kwargs: 其他参数
    Returns:
        LLM服务的响应结果
    """
    logger = logging.getLogger("servellm")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await executor.execute(messages, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM call failed, attempt %d: %s", attempt, exc)
            response = None

        if response:
            logger.debug("LLM call succeeded after %d attempt(s)", attempt)
            return response

        if attempt < MAX_RETRIES:
            await asyncio.sleep(_compute_backoff(attempt))

    logger.error("LLM call failed after max retries %d", MAX_RETRIES)
    return None


async def call_openai_api(
    messages: List[Dict[str, str]],
    base_url: str,
    api_key: str,
    model: str,
    **kwargs
) -> Optional[str]:
    """
    直接调用OpenAI兼容API。
    Args:
        messages: 消息列表
        base_url: API基础URL
        api_key: API密钥
        model: 模型名称
        **kwargs: 其他参数
    Returns:
        API响应内容，失败时返回None
    """
    logger = logging.getLogger("servellm")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": messages,
        "temperature": kwargs.get("temperature", 0.7),
        "max_tokens": kwargs.get("max_tokens", 1000)
    }

    # 指定OpenRouter供应商顺序，确保路由到指定供应商
    provider_slug = os.getenv("OPENROUTER_PROVIDER")
    if provider_slug:
        providers = [p.strip() for p in provider_slug.split(',') if p.strip()]
        if providers:
            data["provider"] = {
                "order": providers,
                "allow_fallbacks": len(providers) > 1,
            }

    # Qwen3 特定参数：关闭 think 模式
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.post(
                f"{base_url.rstrip('/')}"
                f"/chat/completions",
                headers=headers,
                json={**data, **extra_body}
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
    except httpx.TimeoutException as exc:
        logger.error("LLM request timed out: %s", exc)
    except httpx.HTTPStatusError as exc:
        logger.error("API call failed: %s - %s", exc.response.status_code, exc.response.text)
    except httpx.RequestError as exc:
        logger.error("HTTP request error: %s", exc)
    except (KeyError, json.JSONDecodeError) as exc:
        logger.error("Failed to parse LLM response: %s", exc)

    return None


def format_messages(
    content: str,
    role: str = "user",
    system_prompt: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    格式化消息为OpenAI格式。
    Args:
        content: 消息内容
        role: 消息角色，默认为"user"
        system_prompt: 系统提示词，可选
    Returns:
        格式化后的消息列表
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": role, "content": content})

    return messages


def extract_json_from_response(response: str) -> Optional[Dict]:
    """
    从响应中提取JSON数据。
    Args:
        response: LLM响应字符串
    Returns:
        提取的JSON对象，失败时返回None
    """
    response = response.strip()

    try:
        if response.startswith('{') and response.endswith('}'):
            return json.loads(response)

        if '```json' in response:
            start = response.find('```json') + 7
            end = response.find('```', start)
            if end != -1:
                json_str = response[start:end].strip()
                return json.loads(json_str)

        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end != -1 and start < end:
            json_str = response[start:end + 1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        logging.getLogger("servellm").debug("extract_json_from_response failed to parse JSON")

    return None
