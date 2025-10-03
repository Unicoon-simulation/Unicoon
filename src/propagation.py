import asyncio
import logging
from typing import List, Dict, Any, Tuple

from dataclass import AgentSnapshot, News, TransformedNews
from build_prompt import build_news_rewrite_prompt
from parse import parse_news_rewrite_response
from servellm import submit, format_messages
from persuasion import generate_persuasion_decision

PARSE_RETRY_LIMIT = 2  # 解析失败时的最大重试次数

def find_leadernodes(graph) -> List[str]:
    """
    根据节点度数识别leader节点
    Args:
        graph: NetworkX图对象
    Returns:
        度数最高的20%节点ID列表
    """
    logger = logging.getLogger("propagation")
    
    # 计算所有节点的度数
    degree_dict = dict(graph.degree())
    
    # 按度数排序，选择前20%作为leader
    sorted_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)
    leader_count = max(1, int(len(sorted_nodes) * 0.2))
    leader_nodes = [node_id for node_id, degree in sorted_nodes[:leader_count]]
    
    logger.info(f"Identified {len(leader_nodes)} leader nodes; average degree: {sum(degree_dict[n] for n in leader_nodes) / len(leader_nodes):.1f}")
    return leader_nodes

async def news_propagation(
    agents: Dict[str, AgentSnapshot],
    graph,
    round_id: int,
    executor
):
    """
    基于层次遍历的新闻传播主流程
    Args:
        agents: 智能体字典
        graph: 图结构
        round_id: 轮数
        executor: LLM执行器
    Returns:
        Dict[str, Any]: 包含层级传播日志的结构化信息
    """
    logger = logging.getLogger("propagation")
    logger.info(f"[Round {round_id}] Starting hierarchical propagation")
    
    reset_propagation_state(agents)
    
    leader_nodes = find_leadernodes(graph)
    current_layer_nodes = leader_nodes
    layer = 0

    propagation_log = {
        "round": round_id,
        "leader_nodes": leader_nodes,
        "layers": [],
    }

    distribute_initial_news_to_leaders(agents, leader_nodes)

    max_layers = 100  # 最大传播层数限制
    while current_layer_nodes and layer < max_layers:
        logger.info(f"[Round {round_id}] Processing layer {layer} with {len(current_layer_nodes)} nodes")

        next_layer_nodes, layer_node_logs = await process_layer_propagation(
            current_layer_nodes, agents, graph, executor, layer
        )

        if layer_node_logs:
            propagation_log["layers"].append({
                "layer": layer,
                "nodes": layer_node_logs,
            })

        current_layer_nodes = next_layer_nodes
        layer += 1

    for agent in agents.values():
        agent.memory.received_news.clear()

    propagation_log["total_layers"] = layer

    logger.info(f"[Round {round_id}] Propagation finished after {layer} layer(s)")

    return propagation_log

async def process_layer_propagation(
    layer_nodes: List[str],
    agents: Dict[str, AgentSnapshot],
    graph,
    executor,
    layer: int
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    处理单层的新闻传播
    Args:
        layer_nodes: 当前层的节点列表
        agents: 智能体字典
        graph: 图结构
        executor: LLM执行器
        layer: 当前层次
    Returns:
        (下一层需要处理的节点列表, 当前层的传播记录)
    """
    logger = logging.getLogger("propagation")
    next_layer_nodes = set()
    layer_node_logs: List[Dict[str, Any]] = []

    transformation_tasks = []
    for node_id in layer_nodes:
        if node_id in agents:
            agent = agents[node_id]
            agent.memory.propagation_layer = layer

            # 只处理有新闻且未发布的节点
            if agent.memory.received_news_inround and not agent.memory.has_published_this_round:
                task = trans_news(agent, agent.memory.received_news_inround, executor)
                transformation_tasks.append((node_id, task))

    if transformation_tasks:
        transformation_results = await asyncio.gather(
            *[task for _, task in transformation_tasks],
            return_exceptions=True,
        )

        for (node_id, _), result in zip(transformation_tasks, transformation_results):
            if isinstance(result, Exception):
                logger.error("Propagation task failed for node %s: %s", node_id, result)
                continue

            entry: Dict[str, Any] = {
                "agent_id": node_id,
                "layer": layer,
                "status": result.get("status"),
            }

            if result.get("status") == "success":
                agents[node_id].memory.has_published_this_round = True
                followers = get_node_followers(node_id, graph, agents)
                transformed_news = result.get("transformed_news", [])
                entry["transformed_news"] = [
                    {
                        "news_id": tn.news_id,
                        "content": tn.content,
                        "original_news_id": tn.original_news_id,
                        "transformer_id": tn.transformer_agent_id,
                    }
                    for tn in transformed_news
                ]

                delivered_map: Dict[str, List[Dict[str, Any]]] = {}

                for follower_id in followers:
                    if follower_id in agents and not agents[follower_id].memory.has_published_this_round:
                        for transformed_news_item in transformed_news:
                            original_topic = "unknown"
                            for original_news in agents[node_id].memory.received_news_inround:
                                if original_news.news_id == transformed_news_item.news_id:
                                    original_topic = original_news.topic
                                    break

                            propagated_news = News(
                                news_id=transformed_news_item.news_id,
                                sender_id=node_id,
                                receiver_id=follower_id,
                                content=transformed_news_item.content,
                                topic=original_topic,
                            )
                            agents[follower_id].memory.received_news_inround.append(propagated_news)
                            delivered_map.setdefault(follower_id, []).append(
                                {
                                    "news_id": propagated_news.news_id,
                                    "content": propagated_news.content,
                                    "topic": propagated_news.topic,
                                    "sender": propagated_news.sender_id,
                                }
                            )
                        next_layer_nodes.add(follower_id)

                entry["delivered"] = [
                    {
                        "follower_id": follower_id,
                        "news": delivered_map[follower_id],
                    }
                    for follower_id in delivered_map
                ]

                if delivered_map:
                    node_persuasion_inputs = []
                    agent_context = [news.content for news in agents[node_id].memory.received_news_inround]
                    for follower_id, items in delivered_map.items():
                        target_agent = agents.get(follower_id)
                        if not target_agent:
                            continue
                        topic_for_prompt = items[0]["topic"] if items else ""
                        context_snippets = agent_context + [item["content"] for item in items]
                        node_persuasion_inputs.append(
                            (follower_id, target_agent, topic_for_prompt, context_snippets)
                        )

                    if node_persuasion_inputs:
                        raw_results = await asyncio.gather(
                            *[
                                generate_persuasion_decision(
                                    agent,
                                    target_agent,
                                    topic_for_prompt,
                                    context_snippets,
                                    executor,
                                )
                                for follower_id, target_agent, topic_for_prompt, context_snippets in node_persuasion_inputs
                            ],
                            return_exceptions=True,
                        )
                        persuasion_outputs = []
                        for (follower_id, _, topic_for_prompt, _), decision in zip(node_persuasion_inputs, raw_results):
                            if isinstance(decision, Exception):
                                logger.error("Failed to generate persuasion output for node %s: %s", node_id, decision)
                                continue

                            will = decision.get("will", "no")
                            message = decision.get("message", "")
                            record = {
                                "target_id": follower_id,
                                "topic": topic_for_prompt,
                                "will": will,
                                "message": message,
                            }
                            if will == "yes" and message:
                                agents[follower_id].memory.short_term_memory.append(
                                    f"Persuasion from {node_id}: {message}"
                                )
                            persuasion_outputs.append(record)

                        if persuasion_outputs:
                            entry["persuasion"] = persuasion_outputs
            else:
                details = result.get("result")
                if details is not None:
                    entry["details"] = details

            layer_node_logs.append(entry)

    return list(next_layer_nodes), layer_node_logs




async def trans_news(
    agent: AgentSnapshot,
    news_list: List[News],
    executor
) -> Dict[str, Any]:
    """
    单个智能体的新闻改写，返回用于下层传播的新闻
    Args:
        agent: 智能体快照
        news_list: 要改写的新闻列表
        executor: LLM执行器
    Returns:
        转换结果字典，包含transformed_news用于传播
    """
    logger = logging.getLogger("propagation")
    
    if not news_list:
        return {"status": "no_news", "agent_id": agent.agent_id, "transformed_news": [], "result": None}
    
    prompt = build_news_rewrite_prompt(
        agent=agent,
        news_list=news_list
    )
    
    messages = format_messages(prompt, system_prompt=agent.profile.agent_prompt)

    parsed_result = None
    for attempt in range(1, PARSE_RETRY_LIMIT + 2):
        response = await submit(messages, executor, parse_type="json")
        if not response:
            logger.warning(
                "Agent %s news rewrite returned empty response on attempt %d",
                agent.agent_id,
                attempt,
            )
            if attempt < PARSE_RETRY_LIMIT + 1:
                continue
            break

        parsed_result = parse_news_rewrite_response(response)
        if parsed_result.get("status") == "success":
            break

        logger.warning(
            "Agent %s news rewrite parse failed on attempt %d: %s",
            agent.agent_id,
            attempt,
            parsed_result.get("reason") or parsed_result.get("status"),
        )

        parse_reason = (parsed_result.get("reason") or "").lower()
        if parse_reason != "unable to fully parse json" and parsed_result.get("status") not in {"partial"}:
            break

    if not parsed_result or parsed_result.get("status") != "success":
        if parsed_result:
            raw_response = parsed_result.get("original_response")
            if raw_response:
                truncated = raw_response.replace("\n", " ")[:500]
                logger.warning(
                    "Agent %s news rewrite raw response (truncated): %s",
                    agent.agent_id,
                    truncated,
                )
        agent.memory.skip_decay_this_round = True
        logger.warning("Agent %s news rewrite aborted after retries", agent.agent_id)
        return {
            "status": "failed",
            "agent_id": agent.agent_id,
            "transformed_news": [],
            "result": parsed_result,
        }

    transformed_news_list: List[TransformedNews] = []
    try:
        for rewritten in parsed_result.get("rewritten_news", []) or []:
            if not isinstance(rewritten, dict):
                logger.error(
                    "Agent %s rewrite entry has invalid type %s: %s",
                    agent.agent_id,
                    type(rewritten).__name__,
                    rewritten,
                )
                continue

            if rewritten.get("action") == "rewrite" and rewritten.get("transformed_content"):
                transformed_news = TransformedNews(
                    news_id=str(rewritten["news_id"]),
                    editor_id=agent.agent_id,
                    content=rewritten["transformed_content"],
                    original_news_id=str(rewritten["news_id"]),
                    transformer_agent_id=agent.agent_id,
                    transform_time=f"round_{agent.memory.propagation_layer}",
                    propagation_layer=agent.memory.propagation_layer,
                )
                transformed_news_list.append(transformed_news)
                agent.memory.published_news_inround.append(transformed_news)
                agent.memory.published_news.append(transformed_news)
                agent.memory.short_term_memory.append(
                    f"Forwarded news: {transformed_news.content[:50]}..."
                )
            elif rewritten.get("action") == "skip":
                logger.info(
                    "Agent %s skipped news %s: %s",
                    agent.agent_id,
                    rewritten.get("news_id"),
                    rewritten.get("reason", "no reason provided"),
                )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Agent %s failed to consume parsed rewrite payload: %s | payload=%s",
            agent.agent_id,
            exc,
            parsed_result,
        )
        raise

    if not transformed_news_list:
        logger.debug("Agent %s produced no rewrites this round", agent.agent_id)
        return {"status": "no_output", "agent_id": agent.agent_id, "transformed_news": [], "result": parsed_result}

    agent.memory.access_news.extend(news_list)
    logger.debug(
        "Agent %s (layer %d) rewrote %d news items",
        agent.agent_id,
        agent.memory.propagation_layer,
        len(transformed_news_list),
    )

    return {
        "status": "success", 
        "agent_id": agent.agent_id, 
        "result": parsed_result,
        "transformed_news": transformed_news_list
    }
def get_node_followers(node_id: str, graph, agents: Dict[str, AgentSnapshot]) -> List[str]:
    """获取关注该节点的所有智能体。"""
    followers = []

    if hasattr(graph, "predecessors"):
        followers.extend(graph.predecessors(node_id))

    for agent_id, agent in agents.items():
        if node_id in agent.memory.following_list:
            followers.append(agent_id)

    return list(set(followers))


def distribute_initial_news_to_leaders(agents: Dict[str, AgentSnapshot], leader_nodes: List[str]):
    """为 leader 节点分配推荐新闻。"""
    for leader_id in leader_nodes:
        if leader_id in agents:
            agent = agents[leader_id]
            recommended_news = agent.memory.received_news
            agent.memory.received_news_inround.extend(recommended_news)


def reset_propagation_state(agents: Dict[str, AgentSnapshot]):
    """重置所有智能体的传播状态。"""
    for agent in agents.values():
        agent.memory.received_news_inround.clear()
        agent.memory.published_news_inround.clear()
        agent.memory.has_published_this_round = False
        agent.memory.propagation_layer = -1
