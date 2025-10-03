"""分层记忆与社交演化的简单实现。"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

from dataclass import (
    AgentSnapshot,
    WorkingMemoryEntry,
    ShortTermMemoryCluster,
    News,
    Score,
)
from traits_utils import traits_to_tokens

WORKING_MEMORY_LIMIT = 20
WORKING_SUMMARY_LENGTH = 160
SHORT_TERM_PROMOTION_THRESHOLD = 3
LONG_TERM_MEMORY_LIMIT = 2000
RELATIONSHIP_DECAY_STEP = 0.02  # 每轮未互动时信任与兴趣的衰减幅度
RELATIONSHIP_REMOVE_THRESHOLD = 0.3  # 任一维度低于该阈值将移除关注关系

logger = logging.getLogger("memory_pipeline")


def _is_relevant_to_agent(agent: AgentSnapshot, news: News) -> bool:
    """Heuristic relevance check for short-term retention."""
    topic = (getattr(news, "topic", "") or "").lower()
    if not topic:
        return True

    profile = agent.profile
    trait_tokens = traits_to_tokens(getattr(profile, "traits", []))
    if topic in trait_tokens:
        return True
    if topic in (profile.political_affiliation or "").lower():
        return True

    score = agent.memory.following_list.get(getattr(news, "sender_id", ""))
    if score and (score.trust + score.interest) / 2 >= 0.6:
        return True

    return False


def _compress_content(content: str) -> str:
    text = (content or "").strip()
    if len(text) <= WORKING_SUMMARY_LENGTH:
        return text
    snippet = text[:WORKING_SUMMARY_LENGTH]
    for sep in ("。", "！", "!", "？", "?", "；", ";", "."):
        idx = snippet.rfind(sep)
        if idx >= WORKING_SUMMARY_LENGTH // 2:
            return snippet[: idx + 1]
    return snippet + "..."


def _merge_summary(old: str, new: str) -> str:
    if not old:
        return new
    if new in old:
        return old
    combined = f"{old}; {new}" if old else new
    if len(combined) > WORKING_SUMMARY_LENGTH * 2:
        combined = combined[-WORKING_SUMMARY_LENGTH * 2 :]
    return combined


def _promote_to_long_term(existing: str, addition: str) -> str:
    addition = addition.strip()
    if not addition:
        return existing
    if existing and addition in existing:
        return existing
    combined = f"{existing}\n- {addition}" if existing else addition
    if len(combined) > LONG_TERM_MEMORY_LIMIT:
        combined = combined[-LONG_TERM_MEMORY_LIMIT:]
    return combined


def process_agent_memory(agent: AgentSnapshot, round_id: int) -> Dict[str, Any]:
    """执行分层记忆更新并调整社交评分。"""
    working_entries: List[WorkingMemoryEntry] = []
    relevant_entries: List[WorkingMemoryEntry] = []
    social_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "topics": set()})
    filtered_entries: List[Dict[str, Any]] = []
    promotion_events: List[Dict[str, Any]] = []
    timestamp = datetime.utcnow().isoformat()
    skip_decay = getattr(agent.memory, "skip_decay_this_round", False)

    for news in agent.memory.received_news_inround:
        summary = _compress_content(getattr(news, "content", ""))
        entry = WorkingMemoryEntry(
            news_id=news.news_id,
            topic=getattr(news, "topic", "unknown"),
            summary=summary,
            source_id=getattr(news, "sender_id", ""),
            round_id=round_id,
            timestamp=timestamp,
        )
        agent.memory.working_memory.append(entry)
        working_entries.append(entry)

        sender = getattr(news, "sender_id", "")
        if sender and sender not in {agent.agent_id, "sys", "system", ""}:
            social_stats[sender]["count"] += 1
            social_stats[sender]["topics"].add(getattr(news, "topic", "unknown"))

        if _is_relevant_to_agent(agent, news):
            relevant_entries.append(entry)
        else:
            filtered_entries.append(
                {
                    "news_id": entry.news_id,
                    "topic": entry.topic,
                    "source_id": entry.source_id,
                }
            )

    if len(agent.memory.working_memory) > WORKING_MEMORY_LIMIT:
        agent.memory.working_memory = agent.memory.working_memory[-WORKING_MEMORY_LIMIT:]

    short_term_highlights: List[str] = []
    promoted_topics: List[str] = []

    for entry in relevant_entries:
        cluster = agent.memory.short_term_clusters.get(entry.topic)
        if cluster is None:
            cluster = ShortTermMemoryCluster(topic=entry.topic, summary=entry.summary, count=1)
        else:
            cluster.summary = _merge_summary(cluster.summary, entry.summary)
            cluster.count += 1
        agent.memory.short_term_clusters[entry.topic] = cluster
        short_term_highlights.append(f"{cluster.topic}: {cluster.summary} (x{cluster.count})")

        if cluster.count >= SHORT_TERM_PROMOTION_THRESHOLD:
            agent.memory.long_term_memory = _promote_to_long_term(
                agent.memory.long_term_memory,
                cluster.summary,
            )
            cluster.count = 0
            agent.memory.short_term_clusters[entry.topic] = cluster
            promoted_topics.append(entry.topic)
            promotion_events.append(
                {
                    "topic": entry.topic,
                    "summary": cluster.summary,
                    "news_id": entry.news_id,
                    "source_id": entry.source_id,
                }
            )

    for highlight in short_term_highlights:
        agent.memory.short_term_memory.append(highlight)

    social_updates: Dict[str, Any] = {
        "boosted": [],
        "new_followings": [],
        "decayed": [],
        "removed": [],
    }
    for sender, stats in social_stats.items():
        existing = agent.memory.following_list.get(sender)
        delta = min(0.2, 0.05 * stats["count"])
        if existing:
            old_trust, old_interest = existing.trust, existing.interest
            existing.trust = min(1.0, existing.trust + delta)
            existing.interest = min(1.0, existing.interest + delta / 2)
            social_updates["boosted"].append(
                {
                    "agent_id": sender,
                    "old_trust": round(old_trust, 3),
                    "new_trust": round(existing.trust, 3),
                    "old_interest": round(old_interest, 3),
                    "new_interest": round(existing.interest, 3),
                }
            )
        elif stats["count"] >= SHORT_TERM_PROMOTION_THRESHOLD:
            agent.memory.following_list[sender] = Score(trust=0.55, interest=0.55)
            social_updates["new_followings"].append(sender)

    if skip_decay:
        social_updates["skipped_due_to_failure"] = True
        logger.debug("Agent %s skipped decay due to earlier LLM failure", agent.agent_id)
    else:
        # 对本轮没有互动的关注关系进行衰减，必要时直接移除
        for target_id in list(agent.memory.following_list.keys()):
            if target_id == agent.agent_id:
                continue
            if target_id in social_stats:
                continue
            if target_id in social_updates["new_followings"]:
                continue

            score_obj = agent.memory.following_list.get(target_id)
            if not isinstance(score_obj, Score):
                continue

            old_trust = score_obj.trust
            old_interest = score_obj.interest
            new_trust = max(0.0, old_trust - RELATIONSHIP_DECAY_STEP)
            new_interest = max(0.0, old_interest - RELATIONSHIP_DECAY_STEP)

            if new_trust != old_trust or new_interest != old_interest:
                score_obj.trust = new_trust
                score_obj.interest = new_interest
                social_updates["decayed"].append(
                    {
                        "agent_id": target_id,
                        "old_trust": round(old_trust, 3),
                        "new_trust": round(new_trust, 3),
                        "old_interest": round(old_interest, 3),
                        "new_interest": round(new_interest, 3),
                    }
                )

            if new_trust < RELATIONSHIP_REMOVE_THRESHOLD or new_interest < RELATIONSHIP_REMOVE_THRESHOLD:
                del agent.memory.following_list[target_id]
                social_updates["removed"].append(target_id)

    working_snapshot = [f"{entry.topic}: {entry.summary}" for entry in working_entries]

    agent.memory.skip_decay_this_round = False

    return {
        "working_memory_snapshot": working_snapshot,
        "short_term_highlights": short_term_highlights,
        "promoted_topics": promoted_topics,
        "promotion_events": promotion_events,
        "filtered_news": filtered_entries,
        "social_updates": social_updates,
    }
