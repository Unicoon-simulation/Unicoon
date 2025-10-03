import json
import os
from typing import List, Dict, Any

from dataclass import AgentSnapshot, News


def render_prompt_template(template: str, context: Dict[str, Any]) -> str:
    """
    Replace `{key}` placeholders without touching other brace pairs.
    """
    rendered = template
    for key, value in context.items():
        rendered = rendered.replace(f"{{{key}}}", str(value))
    return rendered

# 全局提示词配置缓存
_prompts_config = None


SJT_MEMORY_GUIDANCE = (
    "Additional SJT Guidance:\n"
    "1. Identify the agent's anchor stance and define the latitudes of acceptance, noncommitment, and rejection for the current topic.\n"
    "2. For each message, classify its latitude, predict assimilation or contrast, and record the reasoning.\n"
    "3. Before updating affiliations or memories, explain any shift in latitudes and how ego involvement changes openness or resistance."
)

def load_prompts_config(config_path: str = "config/prompts.json") -> Dict[str, Any]:
    """
    加载提示词配置文件
    Args:
        config_path: 配置文件路径
    Returns:
        提示词配置字典
    """
    global _prompts_config
    if _prompts_config is None:
        with open(config_path, 'r', encoding='utf-8') as f:
            _prompts_config = json.load(f)
    return _prompts_config

def format_news_list(news_list: List[News]) -> str:
    """
    格式化新闻列表为英文格式
    Args:
        news_list: 新闻列表
    Returns:
        格式化后的新闻列表字符串
    """
    return "\n".join([f"{i}. **[{news.topic}]** {news.content}" 
                     for i, news in enumerate(news_list, 1)])

def format_discussion_history(history: List[Dict[str, Any]]) -> str:
    """
    格式化讨论历史记录
    Args:
        history: 讨论历史列表
    Returns:
        格式化后的讨论历史字符串
    """
    return "\n".join([f"**{turn['speaker']}**: {turn['utterance']}" for turn in history])

def format_social_relationships(following_list: Dict[str, Any]) -> str:
    """
    格式化社交关系信息
    Args:
        following_list: 关注列表字典
    Returns:
        格式化后的社交关系字符串
    """
    return "\n".join([f"- {agent_id} (Trust: {score.trust:.2f}, Interest: {score.interest:.2f})" 
                     for agent_id, score in following_list.items()])

def build_rec_prompt(
    agent: AgentSnapshot,
    news_list: List[News],
    prompts: Dict[str, str] = None
) -> str:
    """
    构建推荐prompt，使用英文提示词模板
    Args:
        agent: 智能体快照对象
        news_list: 可推荐的新闻列表
        prompts: 自定义提示词字典，可选
    Returns:
        构建好的推荐提示词字符串
    """
    config = load_prompts_config()
    rec_config = config["prompts"]["recommendation"]

    profile = agent.profile
    traits_str = ", ".join(profile.traits)

    system_prompt = rec_config["system_prompt"]

    recent_memories = "; ".join(list(agent.memory.short_term_memory))
    if not recent_memories:
        recent_memories = "No recent short-term memories recorded"

    user_prompt = render_prompt_template(
        rec_config["user_template"],
        {
            "agent_id": profile.agent_id,
            "name": profile.name,
            "gender": profile.gender,
            "age": profile.age,
            "education_level": profile.education_level,
            "traits": traits_str,
            "political_affiliation": profile.political_affiliation,
            "recent_memories": recent_memories,
            "news_list": format_news_list(news_list),
        },
    )

    return f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}"


def build_news_rewrite_prompt(
    agent: AgentSnapshot,
    news_list: List[News],
    prompts: Dict[str, str] = None
) -> str:
    """
    构建新闻改写prompt，用于智能体决定是否改写新闻
    Args:
        agent: 智能体快照对象
        news_list: 要处理的新闻列表
        prompts: 自定义提示词字典，可选
    Returns:
        构建好的新闻改写提示词字符串
    """
    config = load_prompts_config()
    rewrite_config = config["prompts"]["news_rewrite"]

    profile = agent.profile
    traits_str = ", ".join(profile.traits)

    grouped_news_text = "\n\n".join([
        f"**News ID: {news.news_id}**\n{news.content}"
        for news in news_list
    ])

    news_ids = [news.news_id for news in news_list]

    system_prompt = rewrite_config["system_template"]

    short_term_memories = "; ".join(list(agent.memory.short_term_memory))
    if not short_term_memories:
        short_term_memories = "No short-term memories captured this round"

    user_prompt = rewrite_config["user_template"].format(
        agent_id=profile.agent_id,
        political_affiliation=profile.political_affiliation,
        traits=traits_str,
        short_term_memories=short_term_memories,
        news_ids=", ".join(news_ids) if news_ids else "None",
        grouped_news_text=grouped_news_text or "(No news items provided)"
    )

    return f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}"


def build_news_process_prompt(
    agent: AgentSnapshot,
    grouped_news: List[News],
    source_agents: List[str],
    news_ids: List[str],
    prompts: Dict[str, str] = None
) -> str:
    """
    构建新闻处理prompt，用于新闻改写和评分更新（向后兼容）
    Args:
        agent: 智能体快照对象
        grouped_news: 要处理的新闻列表
        source_agents: 新闻来源智能体列表
        news_ids: 新闻ID列表
        prompts: 自定义提示词字典，可选
    Returns:
        构建好的新闻处理提示词字符串
    """
    return build_news_rewrite_prompt(agent, grouped_news, prompts)


def build_group_discussion_prompt(
    agent: AgentSnapshot,
    discussion_context: Dict[str, Any],
    prompts: Dict[str, str] = None
) -> str:
    """
    构建群组讨论prompt，生成智能体在群组讨论中的发言
    Args:
        agent: 智能体快照对象
        discussion_context: 讨论上下文，包含话题、历史对话等
        prompts: 自定义提示词字典，可选
    Returns:
        构建好的群组讨论提示词字符串
    """
    config = load_prompts_config()
    discussion_config = config["prompts"]["group_discussion"]
    
    # 获取智能体特征信息
    profile = agent.profile
    memory = agent.memory
    traits_str = ", ".join(profile.traits)
    
    # 构建系统提示
    system_prompt = discussion_config["system_prompt"]
    
    # 构建用户提示
    user_prompt = discussion_config["user_template"].format(
        name=profile.name,
        age=profile.age,
        traits=traits_str,
        self_assessment=profile.self_awareness,
        long_term_memory=memory.long_term_memory,
        discussion_history=format_discussion_history(discussion_context["history"])
    )
    
    news_items = discussion_context.get("news", [])
    if news_items:
        news_context = format_news_list(news_items)
        user_prompt = f"{user_prompt}\n\nRelated news:\n{news_context}"

    return f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}"



def build_persuasion_prompt(
    agent: AgentSnapshot,
    target: AgentSnapshot,
    topic: str,
    context_messages: str,
    environment: str = "the current socio-political environment"
) -> str:
    """构建劝说型提示词。"""
    profile = agent.profile
    target_profile = target.profile

    agent_reasons = agent.memory.long_term_memory or profile.self_awareness or "No explicit stance provided."
    target_reasons = target.memory.long_term_memory or target_profile.self_awareness or "No explicit stance provided."
    if not topic:
        topic = "the shared topic"

    prompt = (
        f"Assume you are someone who cares about {environment}.\n"
        f"Your thoughts about {topic} are: <<<"
        f"{agent_reasons}>>>\n"
        f"You have received some messages from friends: <<<"
        f"{context_messages or 'No recent messages.'}>>>\n"
        f"Do you want to persuade a friend of yours whose current thoughts are: <<<"
        f"{target_reasons}>>>?\n"
        "If yes, please output a persuasive message (around 50 words) to support your stance.\n"
        "Return a JSON object with keys 'will' and 'message'.\n"
        "'will' must be 'yes' or 'no'. If 'will' is 'no', keep 'message' empty.\n"
        "Ensure the response is valid JSON."
    )
    return prompt

def build_update_agent_prompt(
    agent: AgentSnapshot,
    round_activities: Dict[str, Any],
    prompts: Dict[str, str] = None
) -> str:
    """
    构建智能体更新prompt，用于记忆整理和状态更新
    Args:
        agent: 智能体快照对象
        round_activities: 本轮活动记录，包含新闻互动、讨论等
        prompts: 自定义提示词字典，可选
    Returns:
        构建好的智能体更新提示词字符串
    """
    config = load_prompts_config()
    memory_config = config["prompts"]["memory_update"]
    
    # 获取智能体信息
    profile = agent.profile
    memory = agent.memory
    traits_str = ", ".join(profile.traits)
    
    # 构建活动历史
    action_history = f"News interactions: {round_activities['news_interactions']} items. Group discussions: {round_activities['discussions']} sessions."
    
    # 构建详细的新闻来源信息
    news_by_source_text = ""
    if "received_news_by_source" in round_activities:
        for sender_id, news_list in round_activities["received_news_by_source"].items():
            news_by_source_text += f"\n**News received from {sender_id}:**\n"
            for news in news_list:
                news_by_source_text += f"- News ID: {news['news_id']}\n"
                news_by_source_text += f"  Content: {news['content']}\n"  # 完整内容
                news_by_source_text += f"  Topic: {news['topic']}\n\n"
    else:
        news_by_source_text = "本轮未收到任何新闻"
    
    # 构建接收消息信息（使用详细的按来源分组信息）
    received_messages = news_by_source_text
    
    # 构建系统提示
    system_prompt = memory_config["system_prompt"]
    if os.getenv("UNICOON_SJT_MEMORY") == "1":
        system_prompt = f"{system_prompt}\n\n{SJT_MEMORY_GUIDANCE}"
    
    # 构建用户提示
    user_prompt = memory_config["user_template"].format(
        name=profile.name,
        age=profile.age,
        traits=traits_str,
        political_affiliation=profile.political_affiliation,
        self_assessment=profile.self_awareness,
        long_term_memory=memory.long_term_memory,
        recent_short_term_memories="; ".join(list(memory.short_term_memory)),
        following_list=format_social_relationships(memory.following_list),
        potential_friends=", ".join(round_activities["potential_friends"]),
        action_history=action_history,
        received_messages=received_messages,
        output_format=json.dumps(memory_config["output_format"], indent=2, ensure_ascii=False)
    )

    working_snapshot = "\n".join(round_activities.get("working_memory_snapshot", []))
    short_term_highlights = "; ".join(round_activities.get("short_term_highlights", []))
    social_updates = round_activities.get("social_updates", {})

    extras = []
    if social_updates.get("skipped_due_to_failure"):
        extras.append("System notice: LLM outputs failed; maintain social scores unchanged this round.")
    if working_snapshot:
        extras.append(f"Working memory digest:\n{working_snapshot}")
    if short_term_highlights:
        extras.append(f"Short-term highlights: {short_term_highlights}")

    boosted = social_updates.get("boosted", [])
    if boosted:
        boosted_text = ", ".join(
            f"{item['agent_id']} (trust {item['old_trust']}→{item['new_trust']})"
            for item in boosted
        )
        extras.append(f"Social boosts: {boosted_text}")

    new_follow = social_updates.get("new_followings", [])
    if new_follow:
        extras.append(f"New followings: {', '.join(new_follow)}")

    decayed = social_updates.get("decayed", [])
    if decayed:
        decayed_text = ", ".join(
            f"{item['agent_id']} (trust {item['old_trust']}→{item['new_trust']}, interest {item['old_interest']}→{item['new_interest']})"
            for item in decayed
        )
        extras.append(f"Social decay: {decayed_text}")

    removed = social_updates.get("removed", [])
    if removed:
        extras.append(f"Relationships pruned: {', '.join(removed)}")

    if extras:
        user_prompt = f"{user_prompt}\n\n" + "\n\n".join(extras)

    return f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}"


def build_content_correction_prompt(
    agent: AgentSnapshot,
    news_summary: str,
    failure_analysis: str,
    original_failed_output: str,
    specific_errors: str,
    prompts: Dict[str, str] = None
) -> str:
    """
    构建内容修正prompt，用于修复失败的新闻改写任务
    Args:
        agent: 智能体快照对象
        news_summary: 原始新闻摘要
        failure_analysis: 失败分析
        original_failed_output: 原始失败输出
        specific_errors: 具体错误信息
        prompts: 自定义提示词字典，可选
    Returns:
        构建好的内容修正提示词字符串
    """
    config = load_prompts_config()
    correction_config = config["prompts"]["content_correction"]
    
    profile = agent.profile
    traits_str = ", ".join(profile.traits)
    
    system_prompt = correction_config["system_prompt"].format(
        name=profile.name,
        traits=traits_str
    )
    
    user_prompt = correction_config["user_template"].format(
        agent_id=profile.agent_id,
        name=profile.name,
        age=profile.age,
        traits=traits_str,
        news_summary=news_summary,
        failure_analysis=failure_analysis,
        original_failed_output=original_failed_output,
        specific_errors=specific_errors
    )
    
    return f"SYSTEM: {system_prompt}\nUSER: {user_prompt}"
