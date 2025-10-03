
import asyncio
import random
from typing import List, Dict, Any, Set
import logging

from dataclass import AgentSnapshot, News
from build_prompt import build_rec_prompt
from parse import parse_recommend_response
from servellm import submit, format_messages

# 推荐超边数据结构
class HyperEdge:
    """超边对象，表示共同关注某条新闻的智能体群组"""
    def __init__(self, news: News, agents_id: List[str]):
        self.news_id = news.news_id
        self.agents_id = agents_id
        self.news_list = [news.news_id]
        self.news_items = [news]


def _normalize_news_id(value):
    """将模型返回的 news_id 统一转换为字符串。"""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        for item in value:
            normalized = _normalize_news_id(item)
            if normalized:
                return normalized
        return None
    return None


async def get_rec(
    agents: Dict[str, AgentSnapshot],
    news_list: List[News],
    executor,
    rec_type: str = "llm"
) -> List[HyperEdge]:
    """
    为所有智能体生成新闻推荐，构建超边
    Args:
        agents: 智能体字典
        news_list: 新闻列表
        executor: LLM执行器
        rec_type: 推荐类型 ("llm", "random", "none")
    Returns:
        超边列表，每个超边包含对同一新闻感兴趣的智能体
    """
    logger = logging.getLogger("rec")
    logger.info(f"Starting recommendations for {len(agents)} agents with {len(news_list)} news items using strategy: {rec_type}")
    
    if rec_type == "none":
        # 消融实验：leader节点接收所有新闻
        raw_edges = build_leader_hyperedges(agents, news_list)
    elif rec_type == "random":
        # 随机推荐
        raw_edges = build_random_hyperedges(agents, news_list)
    else:  # llm推荐
        # 使用LLM进行个性化推荐
        raw_edges = await build_llm_hyperedges(agents, news_list, executor)

    return post_process_hyperedges(raw_edges, agents)

async def build_llm_hyperedges(
    agents: Dict[str, AgentSnapshot],
    news_list: List[News],
    executor
) -> List[HyperEdge]:
    """使用 LLM 构建个性化推荐超边。"""
    logger = logging.getLogger("rec")

    news_map = {news.news_id: news for news in news_list}

    recommendation_tasks = [recommend_news_single(agent, news_list, executor) for agent in agents.values()]
    recommendations = await asyncio.gather(*recommendation_tasks, return_exceptions=True)

    news_to_agents: Dict[str, Set[str]] = {}
    agent_ids = list(agents.keys())

    for agent_id, recommendation in zip(agent_ids, recommendations):
        if isinstance(recommendation, Exception):
            logger.warning("Agent %s recommendation failed: %s", agent_id, recommendation)
            continue
        if not isinstance(recommendation, list):
            if recommendation:
                logger.warning("Agent %s returned recommendations in an unexpected format: %s", agent_id, type(recommendation))
            continue

        seen_news = set()
        for rec_item in recommendation:
            raw_news_id = rec_item.get("news_id") if isinstance(rec_item, dict) else rec_item
            news_id = _normalize_news_id(raw_news_id)
            if not news_id:
                if raw_news_id:
                    logger.debug("Agent %s returned invalid news_id %r", agent_id, raw_news_id)
                continue
            if news_id in seen_news:
                continue
            if news_id not in news_map:
                logger.debug("news_id %s is not present in the current news batch", news_id)
                continue
            news_to_agents.setdefault(news_id, set()).add(agent_id)
            seen_news.add(news_id)

    hyperedges: List[HyperEdge] = []
    for news_id, agent_set in news_to_agents.items():
        if not agent_set:
            continue
        hyperedges.append(HyperEdge(news_map[news_id], sorted(agent_set)))

    if hyperedges:
        logger.info("Constructed %d LLM recommendation hyperedges", len(hyperedges))
    return hyperedges


def build_random_hyperedges(
    agents: Dict[str, AgentSnapshot],
    news_list: List[News]
) -> List[HyperEdge]:
    """
    构建随机推荐超边
    Args:
        agents: 智能体字典
        news_list: 新闻列表
    Returns:
        随机分配的超边列表
    """
    logger = logging.getLogger("rec")
    agent_ids = list(agents.keys())
    hyperedges = []
    
    for news in news_list:
        # 随机选择30%-70%的智能体对此新闻感兴趣
        sample_ratio = random.uniform(0.3, 0.7)
        sample_size = max(1, int(len(agent_ids) * sample_ratio))
        
        interested_agents = random.sample(agent_ids, sample_size)
        hyperedges.append(HyperEdge(news, interested_agents))
    
    logger.info(f"Constructed {len(hyperedges)} random recommendation hyperedges")
    return hyperedges

def build_leader_hyperedges(
    agents: Dict[str, AgentSnapshot],
    news_list: List[News]
) -> List[HyperEdge]:
    """
    构建领导者推荐超边（消融实验用）
    Args:
        agents: 智能体字典
        news_list: 新闻列表
    Returns:
        领导者接收所有新闻的超边列表
    """
    logger = logging.getLogger("rec")
    
    # 找出所有leader节点
    leader_ids = []
    for agent_id, agent in agents.items():
        if agent.profile.is_leader:
            leader_ids.append(agent_id)
    
    # 如果没有显式的leader，选择关注者最多的节点作为leader
    if not leader_ids:
        follower_counts = {}
        for agent_id, agent in agents.items():
            follower_counts[agent_id] = len(agent.memory.following_list)
        
        # 选择关注者数量前20%的节点作为leader
        sorted_agents = sorted(follower_counts.items(), key=lambda x: x[1], reverse=True)
        leader_count = max(1, int(len(sorted_agents) * 0.2))
        leader_ids = [agent_id for agent_id, _ in sorted_agents[:leader_count]]
    
    hyperedges = []
    for news in news_list:
        hyperedges.append(HyperEdge(news, leader_ids.copy()))
    
    logger.info(f"Constructed {len(hyperedges)} leader-driven hyperedges covering {len(leader_ids)} leader nodes")
    return hyperedges

async def recommend_news_single(
    agent: AgentSnapshot,
    news_list: List[News],
    executor
) -> List[Dict[str, Any]]:
    """
    为单个智能体推荐新闻
    Args:
        agent: 智能体快照
        news_list: 新闻列表
        executor: LLM执行器
    Returns:
        推荐结果列表
    """
    logger = logging.getLogger("rec")
    
    # 构建推荐提示词
    prompt = build_rec_prompt(agent, news_list)
    messages = format_messages(prompt, system_prompt=agent.profile.agent_prompt)
    
    # 调用LLM服务
    response = await submit(messages, executor, parse_type="json")

    if not response:
        logger.warning(f"Agent {agent.agent_id} returned an empty recommendation response")
        return []

    # 解析推荐结果
    try:
        recommendations = parse_recommend_response(response)
    except Exception as exc:  # noqa: BLE001
        truncated = response.replace("\n", " ")[:500]
        logger.warning(
            "Agent %s recommendation parse raised %s. Raw response (truncated): %s",
            agent.agent_id,
            exc,
            truncated,
        )
        raise

    logger.debug(f"Agent {agent.agent_id} received {len(recommendations)} recommended items")
    return recommendations

def split_large_hyperedges(
    hyperedges: List[HyperEdge],
    max_size: int = 10
) -> List[HyperEdge]:
    """
    将过大的超边分割成多个小超边
    Args:
        hyperedges: 原始超边列表
        max_size: 最大智能体数量
    Returns:
        分割后的超边列表
    """
    result = []

    for hyperedge in hyperedges:
        if len(hyperedge.agents_id) <= max_size:
            result.append(hyperedge)
            continue

        agents_list = hyperedge.agents_id.copy()
        random.shuffle(agents_list)

        for i in range(0, len(agents_list), max_size):
            sub_agents = agents_list[i:i + max_size]
            if not sub_agents:
                continue

            source_news = hyperedge.news_items[0] if hyperedge.news_items else None
            if source_news is not None:
                new_edge = HyperEdge(source_news, sub_agents)
                new_edge.news_items = list(hyperedge.news_items)
                new_edge.news_list = [item.news_id for item in new_edge.news_items]
                result.append(new_edge)
            else:
                dummy_news = News(
                    news_id=hyperedge.news_id,
                    sender_id='sys',
                    receiver_id='',
                    content='',
                    topic='',
                )
                result.append(HyperEdge(dummy_news, sub_agents))

    return result

def post_process_hyperedges(
    hyperedges: List[HyperEdge],
    agents: Dict[str, AgentSnapshot],
    min_size: int = 1,
    max_size: int = 10
) -> List[HyperEdge]:
    """去除重复、裁剪规模并保证超边内容有效。"""
    logger = logging.getLogger("rec")
    processed: List[HyperEdge] = []
    queue: List[HyperEdge] = list(hyperedges)
    seen_keys = set()

    while queue:
        hyperedge = queue.pop(0)
        if not hyperedge.news_items:
            continue

        unique_agents: List[str] = []
        agent_seen = set()
        for agent_id in hyperedge.agents_id:
            if agent_id in agents and agent_id not in agent_seen:
                unique_agents.append(agent_id)
                agent_seen.add(agent_id)
        if len(unique_agents) < min_size:
            continue

        if len(unique_agents) > max_size:
            primary_news = hyperedge.news_items[0]
            oversized = HyperEdge(primary_news, unique_agents)
            oversized.news_items = list(hyperedge.news_items)
            oversized.news_list = [item.news_id for item in oversized.news_items]
            queue.extend(split_large_hyperedges([oversized], max_size))
            continue

        seen_news = set()
        cleaned_news = []
        for item in hyperedge.news_items:
            if item and item.news_id not in seen_news:
                cleaned_news.append(item)
                seen_news.add(item.news_id)
        if not cleaned_news:
            continue

        hyperedge.agents_id = unique_agents
        hyperedge.news_items = cleaned_news
        hyperedge.news_list = [item.news_id for item in cleaned_news]
        hyperedge.news_id = cleaned_news[0].news_id

        key = (hyperedge.news_id, tuple(unique_agents))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        processed.append(hyperedge)

    if not processed:
        logger.debug("post_process_hyperedges received an empty result set")
    return processed
