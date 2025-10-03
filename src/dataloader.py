import json
import os
from collections import deque
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from dataclass import (
    AgentMemory,
    AgentProfile,
    AgentSnapshot,
    DiscussionRec,
    News,
    Score,
    TalkInTurn,
    TransformedNews,
)

logger = logging.getLogger("dataloader")

# 默认评分
DEFAULT_TRUST = 0.5
DEFAULT_INTEREST = 0.5
_SNAPSHOT_VERSION = 1

# 新闻缓存，避免每轮重复加载
_NEWS_CACHE: Dict[str, Dict[str, Any]] = {}


def _serialize_news(news: News) -> Dict[str, Any]:
    return {
        "news_id": news.news_id,
        "sender_id": news.sender_id,
        "receiver_id": news.receiver_id,
        "content": news.content,
        "topic": news.topic,
    }


def _serialize_transformed_news(news: TransformedNews) -> Dict[str, Any]:
    return {
        "news_id": news.news_id,
        "editor_id": news.editor_id,
        "content": news.content,
        "original_news_id": news.original_news_id,
        "transformer_agent_id": news.transformer_agent_id,
        "transform_time": news.transform_time,
        "propagation_layer": news.propagation_layer,
        "confidence_score": news.confidence_score,
    }


def _serialize_discussion_rec(rec: DiscussionRec) -> Dict[str, Any]:
    return {
        "group_id": rec.group_id,
        "round_num": rec.round_num,
        "news_list": rec.news_list,
        "agents_id": rec.agents_id,
        "turns": [
            {
                "turn_id": turn.turn_id,
                "agent_id": turn.agent_id,
                "content": turn.content,
            }
            for turn in rec.turns
        ],
        "summary": rec.summary,
    }


def _get_news_payload(news_file: str) -> List[Dict[str, Any]]:
    path = Path(news_file)
    mtime = path.stat().st_mtime
    cached = _NEWS_CACHE.get(news_file)

    if not cached or cached.get("mtime") != mtime:
        with path.open('r', encoding='utf-8') as f:
            payload = json.load(f)
        _NEWS_CACHE[news_file] = {"mtime": mtime, "payload": payload}
    else:
        payload = cached["payload"]

    return payload


def load_graph_from_file(
    edge_file: str,
    round_id: int = 0
) -> Tuple[nx.DiGraph, Set[str], List[Tuple[str, str, float, float]]]:
    """从边文件加载图数据。"""
    graph = nx.DiGraph()
    node_list: Set[str] = set()
    edge_list: List[Tuple[str, str, float, float]] = []

    with open(edge_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            source, target = parts[0], parts[1]
            node_list.add(source)
            node_list.add(target)

            trust = float(parts[2]) if len(parts) >= 3 else DEFAULT_TRUST
            interest = float(parts[3]) if len(parts) >= 4 else DEFAULT_INTEREST

            if source != target:
                edge_list.append((source, target, trust, interest))
                graph.add_edge(source, target, trust=trust, interest=interest)

    logger.info("[Graph Loader] Read %s: %d nodes, %d edges", edge_file, len(node_list), len(edge_list))
    return graph, node_list, edge_list


def load_agent_profile(
    profile_file: str
) -> Dict[str, AgentSnapshot]:
    """从文件加载智能体配置文件。"""
    with open(profile_file, 'r', encoding='utf-8') as f:
        data = json.load(f)["backgrounds"]

    agents: Dict[str, AgentSnapshot] = {}

    for agent_id, agent_data in data.items():
        profile = AgentProfile(
            agent_id=agent_id,
            is_leader=False,
            name=agent_data["name"],
            age=agent_data["age"],
            education_level=agent_data["education level"],
            gender=agent_data["gender"],
            traits=agent_data["traits"],
            political_affiliation=agent_data["political_affiliation"],
            self_awareness=agent_data["self-awareness"],
        )
        profile.__agent_person_init__()

        memory = AgentMemory(
            agent_id=agent_id,
            following_list={},
            access_id=[],
            access_news=[],
            published_news=[],
            discuss_mem=[],
            received_news_inround=[],
            published_news_inround=[],
            short_term_memory=deque(maxlen=3),
            long_term_memory="",
        )

        agents[agent_id] = AgentSnapshot(
            round_id=0,
            agent_id=agent_id,
            profile=profile,
            memory=memory,
        )

    return agents


def load_news(
    news_file: str,
    news_start: int = 0,
    news_end: int = -1
) -> List[News]:
    """加载新闻文件并返回指定范围的新闻对象。"""
    payload = _get_news_payload(news_file)
    news_list: List[News] = []

    end_idx = len(payload) if news_end == -1 else min(news_end, len(payload))

    for i in range(news_start, end_idx):
        news_item = payload[i]
        news = News(
            news_id=str(i),
            sender_id="system",
            receiver_id="",
            content=news_item["content"],
            topic=news_item["topic"],
        )
        news_list.append(news)

    logger.debug("[News Loader] Loaded %d articles from %s", len(news_list), news_file)
    return news_list


def sync_edges_from_agents(agents: Dict[str, AgentSnapshot]) -> List[Tuple[str, str, float, float]]:
    """从智能体内存中同步评分到边列表。"""
    edge_list = []
    for agent_id, agent in agents.items():
        for target_id, score in agent.memory.following_list.items():
            edge_list.append((agent_id, target_id, score.trust, score.interest))
    return edge_list


def save_graph(
    edge_list: List[Tuple[str, str, float, float]],
    round_id: int,
    output_dir: str
) -> bool:
    """保存边文件。"""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"round_{round_id}_edges.edges")

    with open(file_path, 'w', encoding='utf-8') as f:
        for edge in edge_list:
            f.write(f"{edge[0]} {edge[1]} {edge[2]} {edge[3]}\n")

    logger.info("[Graph Saver] Wrote edge list to %s", file_path)
    return True


def save_agent_profile(
    agents: Dict[str, AgentSnapshot],
    round_id: int,
    output_dir: str
) -> bool:
    """保存智能体配置和记忆状态。"""
    os.makedirs(output_dir, exist_ok=True)
    profile_path = os.path.join(output_dir, f"round_{round_id}_profile.json")
    mem_path = os.path.join(output_dir, f"round_{round_id}_mem.json")

    profiles: Dict[str, Any] = {}
    memories: Dict[str, Any] = {}

    for agent_id, agent_snapshot in agents.items():
        profiles[agent_id] = {
            "agent_id": agent_snapshot.profile.agent_id,
            "is_leader": agent_snapshot.profile.is_leader,
            "name": agent_snapshot.profile.name,
            "age": agent_snapshot.profile.age,
            "education_level": agent_snapshot.profile.education_level,
            "gender": agent_snapshot.profile.gender,
            "traits": agent_snapshot.profile.traits,
            "political_affiliation": agent_snapshot.profile.political_affiliation,
            "self_awareness": agent_snapshot.profile.self_awareness,
        }

        following_list_serialized = {
            agent_id_key: {"trust": score.trust, "interest": score.interest}
            for agent_id_key, score in agent_snapshot.memory.following_list.items()
        }

        memories[agent_id] = {
            "__version": _SNAPSHOT_VERSION,
            "agent_id": agent_snapshot.memory.agent_id,
            "following_list": following_list_serialized,
            "access_id": agent_snapshot.memory.access_id,
            "access_news": [
                _serialize_news(news) for news in agent_snapshot.memory.access_news
            ],
            "published_news": [
                _serialize_transformed_news(news)
                for news in agent_snapshot.memory.published_news
            ],
            "discuss_mem": [
                _serialize_discussion_rec(rec)
                for rec in agent_snapshot.memory.discuss_mem
            ],
            "received_news_inround": [
                _serialize_news(news)
                for news in agent_snapshot.memory.received_news_inround
            ],
            "published_news_inround": [
                _serialize_transformed_news(news)
                for news in agent_snapshot.memory.published_news_inround
            ],
            "working_memory": [
                {
                    "news_id": entry.news_id,
                    "topic": entry.topic,
                    "summary": entry.summary,
                    "source_id": entry.source_id,
                    "round_id": entry.round_id,
                    "timestamp": entry.timestamp,
                }
                for entry in agent_snapshot.memory.working_memory
            ],
            "short_term_clusters": {
                topic: {
                    "topic": cluster.topic,
                    "summary": cluster.summary,
                    "count": cluster.count,
                }
                for topic, cluster in agent_snapshot.memory.short_term_clusters.items()
            },
            "short_term_memory": list(agent_snapshot.memory.short_term_memory),
            "long_term_memory": agent_snapshot.memory.long_term_memory,
            "received_news": [
                _serialize_news(news)
                for news in agent_snapshot.memory.received_news
            ],
            "has_published_this_round": agent_snapshot.memory.has_published_this_round,
            "propagation_layer": agent_snapshot.memory.propagation_layer,
            "skip_decay_this_round": agent_snapshot.memory.skip_decay_this_round,
        }


    with open(profile_path, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)

    with open(mem_path, 'w', encoding='utf-8') as f:
        json.dump(memories, f, indent=2, ensure_ascii=False)

    logger.info("[Agent Saver] Wrote profiles to %s", profile_path)
    logger.info("[Agent Saver] Wrote memories to %s", mem_path)
    return True


def load_prompt_json(
    prompt_file: str
) -> Dict[str, str]:
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = json.load(f)
    return prompts



def get_prompt_template(
    template_name: str,
    prompts: Optional[Dict[str, str]] = None
) -> str:
    default_templates = {
        "rec_prompt_template": "Based on your profile and interests, pick the news items that attract you the most\n{news_info}\nand briefly explain why.",
        "news_process_template": "Restate the following news items from your own perspective and stance\n{news_content}\nKeep the facts accurate while adjusting the tone.",
        "discussion_template": "In a group discussion about '{topic}', share your point of view\n{context}\nKeep your contribution concise.",
        "memory_update_template": "Summarize today's activities and update your memory and opinions\n{activities}\nHighlight the key information.",
    }

    if prompts and template_name in prompts:
        return prompts[template_name]
    if template_name in default_templates:
        logger.info("Using default template for %s", template_name)
        return default_templates[template_name]
    logger.warning(
        "Template %s not found in provided prompts or defaults; using fallback placeholder",
        template_name,
    )
    return "请处理以下任务：{content}"
