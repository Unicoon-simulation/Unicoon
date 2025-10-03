import asyncio
import logging
from typing import List, Dict, Any
from collections import Counter
import re

from dataclass import AgentSnapshot, News
from build_prompt import build_group_discussion_prompt
from parse import parse_group_discussion_response
from servellm import submit, format_messages
from persuasion import generate_persuasion_decision

_KEYWORD_PATTERN_LATIN = re.compile(r"[A-Za-z]{4,}")
_KEYWORD_PATTERN_CJK = re.compile(r"[\u4e00-\u9fa5]{2,}")

def extract_keywords(discussion_log: List[Dict[str, Any]], limit: int = 5) -> List[str]:
    """基于规则的关键词提取，覆盖英文和中文。"""
    counter: Counter = Counter()
    for entry in discussion_log:
        utterance = entry.get("utterance", "")
        for token in _KEYWORD_PATTERN_LATIN.findall(utterance):
            counter[token.lower()] += 1
        for token in _KEYWORD_PATTERN_CJK.findall(utterance):
            counter[token] += 1
    return [word for word, _ in counter.most_common(limit)]

def build_discussion_summary(
    discussion_log: List[Dict[str, Any]],
    news_list: List[News] | None
) -> Dict[str, Any]:
    """生成便于分析的讨论摘要。"""
    if not discussion_log:
        topics = [getattr(news, "topic", "") for news in (news_list or []) if getattr(news, "topic", "")]
        return {
            "total_turns": 0,
            "top_speakers": [],
            "keywords": [],
            "topic_focus": topics,
            "first_utterance": "",
            "final_utterance": "",
        }

    speaker_counts = Counter(entry["speaker_id"] for entry in discussion_log)
    speaker_name_map = {entry["speaker_id"]: entry.get("speaker", entry["speaker_id"]) for entry in discussion_log}
    top_speakers = [
        {
            "agent_id": agent_id,
            "name": speaker_name_map.get(agent_id, agent_id),
            "turns": speaker_counts[agent_id],
        }
        for agent_id, _ in speaker_counts.most_common(3)
    ]

    topics = [getattr(news, "topic", "") for news in (news_list or []) if getattr(news, "topic", "")]
    keywords = extract_keywords(discussion_log)

    return {
        "total_turns": len(discussion_log),
        "top_speakers": top_speakers,
        "keywords": keywords,
        "topic_focus": topics,
        "first_utterance": discussion_log[0]["utterance"],
        "final_utterance": discussion_log[-1]["utterance"],
    }

async def generate_utterance(
    agent: AgentSnapshot,
    discussion_history: List[Dict],
    news_items: List[News],
    executor
) -> str:
    """
    为智能体生成一句讨论发言
    Args:
        agent: 智能体快照
        discussion_history: 讨论历史记录列表
        news_items: 讨论的新闻对象列表
        executor: LLM执行器
    Returns:
        发言内容字符串
    """
    logger = logging.getLogger("discuss")
    
    discussion_context = {
        "history": discussion_history,
        "news": news_items or []
    }
    
    prompt = build_group_discussion_prompt(agent, discussion_context)
    
    messages = format_messages(prompt, system_prompt=agent.profile.agent_prompt)
    
    response = await submit(messages, executor, parse_type="json")
    
    if not response:
        logger.warning(f"Agent {agent.agent_id} returned an empty discussion response")
        return f"[{agent.profile.name} remains silent]"
    
    parsed_result = parse_group_discussion_response(response)
    
    utterance = ""
    if parsed_result["status"] == "success":
        utterance = parsed_result["utterance"]
    else:
        utterance = f"[{agent.profile.name}'s utterance could not be parsed]"
    
    if utterance:
        memory_item = f"Group discussion utterance: {utterance}"
        agent.memory.short_term_memory.append(memory_item)
    
    logger.debug(f"Agent {agent.agent_id} produced utterance: {utterance}")
    
    return utterance

async def run_discussion_group(
    group_members: List[str],
    group_id: str,
    agents: Dict[str, AgentSnapshot],
    executor,
    news_list: List[News] | None = None
):
    """
    驱动单个讨论小组完成3轮对话
    Args:
        group_members: 小组成员的agent_id列表
        group_id: 该小组的唯一标识符
        agents: 智能体字典
        executor: LLM执行器
        news_list: 讨论的新闻列表
    Returns:
        包含完整对话记录的字典
    """
    logger = logging.getLogger("discuss")
    max_turns = 3  # 固定3轮讨论
    discussion_log = []

    # 进行3轮讨论，每轮每人发言一次
    for turn in range(1, max_turns + 1):
        
        for speaker_id in group_members:
            if speaker_id in agents:
                speaker_agent = agents[speaker_id]                  
                utterance = await generate_utterance(
                    speaker_agent,
                    discussion_log,
                    news_list or [],
                    executor,
                )

                persuasion_info = None
                if len(group_members) > 1:
                    target_agent = None
                    for candidate_id in group_members:
                        if candidate_id == speaker_id:
                            continue
                        candidate = agents.get(candidate_id)
                        if not candidate:
                            continue
                        if candidate.profile.political_affiliation != speaker_agent.profile.political_affiliation:
                            target_agent = candidate
                            break
                        if target_agent is None:
                            target_agent = candidate
                    if target_agent:
                        topic = news_list[0].topic if news_list else ""
                        context_snippets = [entry.get("utterance", "") for entry in discussion_log[-5:]]
                        context_snippets.extend(news.content for news in (news_list or []))
                        decision = await generate_persuasion_decision(
                            speaker_agent,
                            target_agent,
                            topic,
                            context_snippets,
                            executor,
                        )
                        if decision.get("will") == "yes" and decision.get("message"):
                            utterance = f"{utterance} {decision['message']}".strip()
                            agents[target_agent.agent_id].memory.short_term_memory.append(
                                f"Persuasion in discussion from {speaker_agent.agent_id}: {decision['message']}"
                            )
                        decision["target_id"] = target_agent.agent_id
                        decision["topic"] = topic
                        persuasion_info = decision

                # 记录发言到讨论日志
                discussion_entry = {
                    "turn": turn,
                    "speaker_id": speaker_id,
                    "speaker": speaker_agent.profile.name,
                    "utterance": utterance,
                    "timestamp": f"turn_{turn}"
                }
                if persuasion_info:
                    discussion_entry["persuasion"] = persuasion_info
                discussion_log.append(discussion_entry)
                
    
    
    # 创建讨论记录并存储到每个参与者的记忆中
    from dataclass import DiscussionRec, TalkInTurn
    
    discussion_summary = build_discussion_summary(discussion_log, news_list)

    discussion_rec = DiscussionRec(
        group_id=group_id,
        round_num=int(group_id.split('_')[1]) if '_' in group_id else 0,
        news_list=[news.news_id if hasattr(news, 'news_id') else str(news) for news in (news_list or [])],
        agents_id=group_members,
        turns=[TalkInTurn(
            turn_id=f"{entry['turn']}_{entry['speaker_id']}", 
            agent_id=entry['speaker_id'], 
            content=entry['utterance']
        ) for entry in discussion_log],
        summary=discussion_summary,
    )
    
    # 将讨论记录存入每个参与者的记忆
    for member_id in group_members:
        if member_id in agents:
            agents[member_id].memory.discuss_mem.append(discussion_rec)
    
    # 返回讨论记录
    return {
        "group_id": group_id,
        "members": group_members,
        "news_topics": [
            {"news_id": news.news_id, "content": news.content, "topic": getattr(news, "topic", "")} if hasattr(news, "news_id")
            else {"news_id": news, "content": f"News ID: {news}", "topic": ""}
            for news in (news_list or [])
        ],
        "turns": discussion_log,
        "summary": discussion_summary
    }
