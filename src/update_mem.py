import asyncio
import logging
import os
from typing import List, Dict, Any

from dataclass import AgentSnapshot, Score
from build_prompt import build_update_agent_prompt
from parse import parse_update_mem_response, parse_mem_updatenew_response
from servellm import submit, format_messages
from memory_pipeline import process_agent_memory, RELATIONSHIP_REMOVE_THRESHOLD
from traits_utils import normalize_traits_list

DEFAULT_SCORE_VALUE = 0.5
PARSE_RETRY_LIMIT = 2  # 解析失败时的最大重试次数


def _coerce_float(value: Any, default: float) -> float:
    """将值转换为float，转换失败时退回默认值"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_score(value: float) -> float:
    """保持评分值在[0, 1]范围内"""
    return max(0.0, min(1.0, value))


def _sanitize_score_value(raw_value: Any, fallback: float) -> float:
    """将原始评分标准化为限制范围内的浮点数"""
    return _clamp_score(_coerce_float(raw_value, fallback))


def _sanitize_score(score: Score) -> Score:
    """确保 Score 实例存储数值类型的信任度/兴趣度"""
    score.trust = _sanitize_score_value(getattr(score, "trust", DEFAULT_SCORE_VALUE), DEFAULT_SCORE_VALUE)
    score.interest = _sanitize_score_value(getattr(score, "interest", DEFAULT_SCORE_VALUE), DEFAULT_SCORE_VALUE)
    return score


VALID_AFFILIATIONS = {"far_left", "left", "moderate", "right", "far_right"}


def _append_short_term_memory(agent: AgentSnapshot, payload: Any) -> None:
    if isinstance(payload, str):
        value = payload.strip()
        if value:
            agent.memory.short_term_memory.append(value)
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                value = item.strip()
                if value:
                    agent.memory.short_term_memory.append(value)


def _apply_score_updates(
    agent: AgentSnapshot,
    score_updates: List[Dict[str, Any]] | None,
    logger: logging.Logger,
    *,
    log_changes: bool = False,
) -> None:
    if not score_updates:
        return

    for score_update in score_updates:
        target_agent_id = score_update.get("agent_id")
        if not target_agent_id or target_agent_id not in agent.memory.following_list:
            continue

        score_obj = _sanitize_score(agent.memory.following_list[target_agent_id])
        old_trust = score_obj.trust
        old_interest = score_obj.interest
        new_trust = _sanitize_score_value(score_update.get("trust"), old_trust)
        new_interest = _sanitize_score_value(score_update.get("interest"), old_interest)

        score_obj.trust = new_trust
        score_obj.interest = new_interest

        removed = (
            new_trust < RELATIONSHIP_REMOVE_THRESHOLD
            or new_interest < RELATIONSHIP_REMOVE_THRESHOLD
        )
        if removed:
            del agent.memory.following_list[target_agent_id]
            logger.info(
                "Agent %s removed the follow relationship with %s: trust %.2f→%.2f, interest %.2f→%.2f",
                agent.agent_id,
                target_agent_id,
                old_trust,
                new_trust,
                old_interest,
                new_interest,
            )
            continue

        if log_changes and (abs(old_trust - new_trust) > 0.1 or abs(old_interest - new_interest) > 0.1):
            logger.info(
                "Agent %s updated scores for %s: trust %.2f→%.2f, interest %.2f→%.2f",
                agent.agent_id,
                target_agent_id,
                old_trust,
                new_trust,
                old_interest,
                new_interest,
            )


def _apply_trait_changes(
    agent: AgentSnapshot,
    trait_changes: Dict[str, Any] | None,
    logger: logging.Logger,
    *,
    detailed: bool = False,
) -> None:
    if not trait_changes:
        return

    if detailed:
        changes_made: List[str] = []
        if "traits" in trait_changes:
            normalized_traits = normalize_traits_list(trait_changes["traits"])
            old_traits = list(agent.profile.traits)
            agent.profile.traits = normalized_traits
            if old_traits != agent.profile.traits:
                changes_made.append(f"traits: {old_traits} → {agent.profile.traits}")

        if "self_awareness" in trait_changes:
            old_awareness = agent.profile.self_awareness
            agent.profile.self_awareness = trait_changes["self_awareness"]
            if old_awareness != agent.profile.self_awareness:
                changes_made.append("self_awareness updated")

        if "political_affiliation" in trait_changes:
            new_affiliation = trait_changes["political_affiliation"]
            if new_affiliation in VALID_AFFILIATIONS:
                old_affiliation = agent.profile.political_affiliation
                agent.profile.political_affiliation = new_affiliation
                if old_affiliation != new_affiliation:
                    agent.profile.__agent_person_init__()
                    changes_made.append(
                        f"political_affiliation: {old_affiliation} → {new_affiliation}"
                    )
            else:
                logger.warning(
                    "Agent %s attempted to set an invalid political affiliation: %s",
                    agent.agent_id,
                    new_affiliation,
                )

        if changes_made:
            logger.info(
                "Agent %s trait evolution: %s",
                agent.agent_id,
                "; ".join(changes_made),
            )
        return

    if "traits" in trait_changes:
        agent.profile.traits = normalize_traits_list(trait_changes["traits"])

    if "self_awareness" in trait_changes:
        agent.profile.self_awareness = trait_changes["self_awareness"]

    if "political_affiliation" in trait_changes:
        old_affiliation = agent.profile.political_affiliation
        new_affiliation = trait_changes["political_affiliation"]
        agent.profile.political_affiliation = new_affiliation
        if old_affiliation != agent.profile.political_affiliation:
            agent.profile.__agent_person_init__()
            logger.info(
                "Agent %s political affiliation changed from %s to %s",
                agent.agent_id,
                old_affiliation,
                agent.profile.political_affiliation,
            )
def collect_round_activities(agent: AgentSnapshot, round_id: int, all_agents: Dict[str, AgentSnapshot] = None) -> Dict[str, Any]:
    """
    收集智能体本轮活动信息
    Args:
        agent: 智能体快照
        round_id: 轮次ID
        all_agents: 所有智能体字典，用于生成完整的potential_friends列表防止幻觉
    Returns:
        包含本轮活动信息的字典
    """
    # 收集新闻相关活动
    # 调试：检查数据类型
    logger = logging.getLogger("update_mem")
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Agent %s received_news_inround type=%s published_news_inround type=%s",
            agent.agent_id,
            type(agent.memory.received_news_inround),
            type(agent.memory.published_news_inround),
        )

    memory_report = process_agent_memory(agent, round_id)

    received_news_count = len(agent.memory.received_news_inround)
    published_news_count = len(agent.memory.published_news_inround)
    
    # 收集讨论相关活动 - 直接访问discuss_mem字段
    recent_discussions = []
    for discussion in agent.memory.discuss_mem:
        if hasattr(discussion, 'group_id') and f"round_{round_id}" in discussion.group_id:
            recent_discussions.append(f"Joined discussion group {discussion.group_id}")
    
    # 构建活动记录，同时收集新闻来源信息
    received_news_summaries = []
    source_agent_news_map = {}  # 按来源分组的完整新闻信息
    
    for news in agent.memory.received_news_inround:
        summary = f"Received news: {news.content[:50]}..."
        received_news_summaries.append(summary)
        
        # 按发送者分组收集完整新闻信息
        sender = news.sender_id
        if sender not in source_agent_news_map:
            source_agent_news_map[sender] = []
        source_agent_news_map[sender].append({
            'news_id': news.news_id,
            'content': news.content,  # 完整内容，不截断
            'sender_id': news.sender_id,
            'receiver_id': news.receiver_id,
            'topic': news.topic
        })
    
    published_news_summaries = []
    for news in agent.memory.published_news_inround:
        summary = f"Published news: {news.content[:50]}..."
        published_news_summaries.append(summary)
    
    # 生成潜在关注对象列表
    potential_friends: List[str] = []

    def _add_potential(candidate_id: str) -> None:
        if candidate_id != agent.agent_id and candidate_id not in potential_friends:
            potential_friends.append(candidate_id)

    if all_agents:
        for candidate_id in all_agents.keys():
            _add_potential(candidate_id)
        logger.debug(
            "Agent %s received the full potential_friends list with %d candidates",
            agent.agent_id,
            len(potential_friends),
        )
    else:
        for candidate_id, score in agent.memory.following_list.items():
            sanitized_score = _sanitize_score(score)
            if sanitized_score.interest > 0.6 or sanitized_score.trust > 0.6:
                _add_potential(candidate_id)

    social_updates = memory_report.get("social_updates", {})
    new_followings = social_updates.get("new_followings", [])
    for new_follow in new_followings:
        _add_potential(new_follow)

    return {
        "news_interactions": received_news_count + published_news_count,
        "discussions": len(recent_discussions),
        "received_news": received_news_summaries,
        "received_news_by_source": source_agent_news_map,
        "received_news_count": received_news_count,
        "discussion_messages": recent_discussions,
        "potential_friends": potential_friends,
        "propagation_layer": agent.memory.propagation_layer,
        "published_count": published_news_count,
        "access_news_count": len(agent.memory.access_news) if isinstance(agent.memory.access_news, list) else 0,
        "short_term_memories": list(agent.memory.short_term_memory),
        "working_memory_snapshot": memory_report.get("working_memory_snapshot", []),
        "short_term_highlights": memory_report.get("short_term_highlights", []),
        "promoted_topics": memory_report.get("promoted_topics", []),
        "promotion_events": memory_report.get("promotion_events", []),
        "filtered_news": memory_report.get("filtered_news", []),
        "social_updates": social_updates,
        "source_agents": list(source_agent_news_map.keys()),
    }

async def single_agent_memory_update(
    agent: AgentSnapshot,
    round_activities: Dict[str, Any],
    executor
):
    """
    单个智能体的记忆更新
    Args:
        agent: 智能体快照
        round_activities: 本轮活动记录
        executor: LLM执行器
    Returns:
        None (直接更新agent状态)
    """
    logger = logging.getLogger("update_mem")
    
    prompt = build_update_agent_prompt(agent, round_activities)
    
    messages = format_messages(prompt, system_prompt=agent.profile.agent_prompt)

    parsed_result: Dict[str, Any] | None = None
    for attempt in range(1, PARSE_RETRY_LIMIT + 2):
        response = await submit(messages, executor, parse_type="json")
        if not response:
            logger.warning(
                "Agent %s memory update returned empty response on attempt %d",
                agent.agent_id,
                attempt,
            )
            if attempt < PARSE_RETRY_LIMIT + 1:
                continue
            break

        parsed_result = parse_update_mem_response(response)
        if parsed_result.get("status") == "success":
            break

        logger.warning(
            "Agent %s memory update parse failed on attempt %d: %s",
            agent.agent_id,
            attempt,
            parsed_result.get("reason") or parsed_result.get("status"),
        )

        if parsed_result.get("status") not in {"partial"}:
            break

    if not parsed_result or parsed_result.get("status") != "success":
        agent.memory.skip_decay_this_round = True
        logger.warning(f"Agent {agent.agent_id} memory update parsing failed after retries")
        return

    if "updated_long_term_memory" in parsed_result:
        agent.memory.long_term_memory = parsed_result["updated_long_term_memory"]

    _append_short_term_memory(agent, parsed_result.get("new_short_term_memories"))
    _apply_score_updates(agent, parsed_result.get("score_updates"), logger)
    _apply_trait_changes(agent, parsed_result.get("trait_changes"), logger)

    logger.debug(f"Agent {agent.agent_id} memory update completed")

async def single_agent_memory_update_enhanced(
    agent: AgentSnapshot,
    round_activities: Dict[str, Any],
    executor
):
    """
    增强版单个智能体的记忆更新，支持特征演化和全面的社交关系管理
    Args:
        agent: 智能体快照
        round_activities: 本轮活动记录（包含完整的potential_friends列表）
        executor: LLM执行器
    Returns:
        None (直接更新agent状态)
    """
    logger = logging.getLogger("update_mem")
    
    prompt = build_update_agent_prompt(agent, round_activities)
    
    messages = format_messages(prompt, system_prompt=agent.profile.agent_prompt)

    parsed_result: Dict[str, Any] | None = None
    for attempt in range(1, PARSE_RETRY_LIMIT + 2):
        response = await submit(messages, executor, parse_type="json")
        if not response:
            logger.warning(
                "Agent %s enhanced memory update returned empty response on attempt %d",
                agent.agent_id,
                attempt,
            )
            if attempt < PARSE_RETRY_LIMIT + 1:
                continue
            break

        parsed_result = parse_mem_updatenew_response(response)
        if parsed_result.get("status") == "success":
            break

        logger.warning(
            "Agent %s enhanced memory update parse failed on attempt %d: %s",
            agent.agent_id,
            attempt,
            parsed_result.get("reason") or parsed_result.get("status"),
        )

        if parsed_result.get("status") not in {"partial", "failed"}:
            break

    if not parsed_result or parsed_result.get("status") != "success":
        agent.memory.skip_decay_this_round = True
        logger.warning(
            "Agent %s enhanced memory update parsing failed after retries",
            agent.agent_id,
        )
        return

    if "updated_long_term_memory" in parsed_result:
        agent.memory.long_term_memory = parsed_result["updated_long_term_memory"]

    _append_short_term_memory(agent, parsed_result.get("new_short_term_memories"))
    _apply_score_updates(agent, parsed_result.get("score_updates"), logger, log_changes=True)
    _apply_trait_changes(agent, parsed_result.get("trait_changes"), logger, detailed=True)

    logger.debug(f"Agent {agent.agent_id} enhanced memory update completed")


async def update_mem(
    round_id: int,
    agents: Dict[str, AgentSnapshot],
    executor
):
    """
    批量更新所有智能体的记忆状态
    Args:
        round_id: 轮次ID
        agents: 智能体字典
        executor: LLM执行器
    Returns:
        Dict[str, Dict[str, Any]]: 每个智能体的活动快照
    """
    logger = logging.getLogger("update_mem")
    if os.getenv("UNICOON_SJT_MEMORY") == "1":
        logger.debug("SJT memory guidance enabled via UNICOON_SJT_MEMORY")
    logger.info(f"[Round {round_id}] Starting batch memory updates")

    activities_log: Dict[str, Dict[str, Any]] = {}

    update_tasks = []
    for agent_id, agent in agents.items():
        round_activities = collect_round_activities(agent, round_id)
        activities_log[agent_id] = round_activities

        task = single_agent_memory_update(agent, round_activities, executor)
        update_tasks.append((agent_id, task))

    if update_tasks:
        update_results = await asyncio.gather(*[task for _, task in update_tasks], return_exceptions=True)

        # 处理异常结果
        for i, (agent_id, _) in enumerate(update_tasks):
            result = update_results[i]
            if isinstance(result, Exception):
                logger.error(f"Agent {agent_id} memory update failed: {result}")

    logger.info(f"[Round {round_id}] Completed memory updates for {len(agents)} agents")

    return activities_log

async def update_mem_enhanced(
    round_id: int,
    agents: Dict[str, AgentSnapshot],
    executor
):
    """
    批量更新所有智能体的记忆状态（增强版，支持特征演化和防幻觉关注管理）
    Args:
        round_id: 轮次ID
        agents: 智能体字典
        executor: LLM执行器
    Returns:
        Dict[str, Dict[str, Any]]: 每个智能体的活动快照
    """
    logger = logging.getLogger("update_mem")
    if os.getenv("UNICOON_SJT_MEMORY") == "1":
        logger.debug("SJT memory guidance enabled via UNICOON_SJT_MEMORY")
    logger.info(f"[Round {round_id}] Starting enhanced batch memory updates ({len(agents)} agents)")

    activities_log: Dict[str, Dict[str, Any]] = {}

    update_tasks = []
    for agent_id, agent in agents.items():
        # 收集活动信息，提供所有智能体列表防止幻觉
        round_activities = collect_round_activities(agent, round_id, all_agents=agents)
        activities_log[agent_id] = round_activities

        # 创建增强更新任务
        task = single_agent_memory_update_enhanced(agent, round_activities, executor)
        update_tasks.append((agent_id, task))

    successful_updates = 0

    if not update_tasks:
        logger.info(f"[Round {round_id}] No enhanced memory updates required (0 agents)")
        return activities_log

    update_results = await asyncio.gather(*[task for _, task in update_tasks], return_exceptions=True)

    # 处理异常结果和记录统计信息
    for i, (agent_id, _) in enumerate(update_tasks):
        result = update_results[i]
        if isinstance(result, Exception):
            logger.error(f"Agent {agent_id} enhanced memory update failed: {result}")
        else:
            successful_updates += 1

    logger.info(f"[Round {round_id}] Enhanced memory updates completed: {successful_updates}/{len(agents)} succeeded")

    return activities_log
