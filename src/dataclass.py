from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import deque
from typing import Deque
import datetime

@dataclass
class Score:
    """信任度 (trust) 与 兴趣度 (interest) 的封装。"""
    trust: float  # 信任度 [0.0, 1.0]
    interest: float  # 兴趣度 [0.0, 1.0]

@dataclass
class News:
    """系统产生的原始新闻。"""
    news_id: str  # 新闻唯一 ID
    sender_id: str  # 新闻发送者智能体 ID
    receiver_id: str  # 新闻接收者智能体 ID
    content: str  # 新闻正文内容
    topic: str  # 新闻政治倾向

@dataclass
class TransformedNews:
    """智能体改写后的新闻。"""
    news_id: str  # 对应原新闻的唯一 ID
    editor_id: str  # 本次改写者（智能体）ID
    content: str  # 改写后的新闻内容
    original_news_id: str = ""  # 原始新闻ID（用于层次传播追踪）
    transformer_agent_id: str = ""  # 转换者智能体ID
    transform_time: str = ""  # 转换时间标记
    propagation_layer: int = 0  # 传播层次
    confidence_score: float = 1.0  # 可信度评分

@dataclass
class WorkingMemoryEntry:
    """工作记忆条目，保存最新的新闻压缩信息。"""
    news_id: str
    topic: str
    summary: str
    source_id: str
    round_id: int
    timestamp: str


@dataclass
class ShortTermMemoryCluster:
    """短期记忆簇，聚合同主题线索。"""
    topic: str
    summary: str
    count: int = 0


@dataclass
class NewsInteraction:
    """新闻交互记录，用于详细追踪智能体与新闻的互动"""
    news_id: str  # 新闻ID
    from_id: str  # 新闻来源
    content: str  # 新闻内容
    received_at: str  # 接收时间戳
    action: str  # 执行的动作：received, transformed, published, forwarded
    action_details: Optional[Dict[str, Any]] = None  # 动作的详细信息

@dataclass
class SocialRelationshipChange:
    """社交关系变化记录"""
    target_agent_id: str  # 目标智能体ID
    action: str  # 动作类型：follow, unfollow, score_update
    previous_score: Optional[Score] = None  # 之前的评分
    new_score: Optional[Score] = None  # 新的评分
    reason: str = ""  # 变化原因
    timestamp: str = ""  # 变化时间戳

@dataclass
class TalkInTurn:
    turn_id: str  # 发言轮次 ID
    agent_id: str  # 发言智能体 ID
    content: str  # 发言内容

@dataclass
class DiscussionRec:
    group_id: str  # 讨论组ID
    round_num: int  # 所属仿真轮次
    news_list:List[str] # 讨论的新闻ID列表
    agents_id: List[str]  # 参与智能体 ID 列表
    turns: List[TalkInTurn]  # 对话轮列表
    summary: Dict[str, Any] = field(default_factory=dict)  # 讨论摘要信息

@dataclass
class DiscussionMem:
    """讨论记忆"""
    agent_id: str # 智能体ID
    news_list:List[str] # 讨论的新闻ID列表
    discuss_mem: List[DiscussionRec] # 讨论的记忆


@dataclass
class AgentProfile:
    """智能体配置文件"""
    agent_id: str  # 智能体唯一 ID
    is_leader: bool  # 是否为领导者角色
    name:str # 姓名 不可修改
    age:int # 年龄 不可修改
    education_level: str # 学历 不可修改
    gender: str # 性别 不可修改
    traits:List[str] # 性格特点，可修改
    political_affiliation: str # 政治倾向 可修改 far right , right , moderate , left , far left
    self_awareness: str # 自我评估 可修改 
    # 初始模板，不带默认值，因为会在 __post_init__ 中生成
    _agent_prompt_template: str = "Imagine you are a human. Your name is {name}, and your gender is {gender}.You are {age} years old. Your personality is shaped by these specific traits:{traits_str}.Your educational background is at the level of {education_level} Degree.Your political affiliation is {political_affiliation}.Act according to this human identity, letting these details fully define your thoughts, responses, interactions, and decisions."
    agent_prompt: str = "" # 最终生成的prompt
    
    def __agent_person_init__(self):
        # 将 traits 列表转换为逗号分隔的字符串
        traits_str = ", ".join(self.traits)
        self.agent_prompt = self._agent_prompt_template.format(
            name=self.name,
            gender=self.gender,
            age=self.age,
            traits_str=traits_str,
            education_level=self.education_level,
            political_affiliation=self.political_affiliation
        )   
         
@dataclass
class AgentMemory:
    """智能体记忆"""
    agent_id: str # 唯一标识 ID
    following_list: Dict[str, Score]  # 智能体关注的节点及其评分
    access_id: List[str]  # 被允许访问的智能体ID
    access_news:List[News] # 收到的新闻信息
    published_news:List[TransformedNews] # 已发布的新闻
    discuss_mem:List[DiscussionRec] # 讨论记忆
    received_news_inround:List[News] # 回合内收到的新闻信息
    published_news_inround:List[TransformedNews] # 回合内发布的新闻信息
    working_memory: List[WorkingMemoryEntry] = field(default_factory=list)  # 工作记忆压缩结果
    short_term_clusters: Dict[str, ShortTermMemoryCluster] = field(default_factory=dict)  # 短期记忆簇
    short_term_memory:Deque[str] = field(default_factory=lambda: deque(maxlen=3)) # 短期记忆 三条，每回合整理一次，替换最旧的
    long_term_memory:str = "" # 不断归纳，尽量保持长度不变
    
    # 新增传播状态字段
    has_published_this_round: bool = False  # 本轮是否已发布过新闻
    propagation_layer: int = -1  # 传播层次，-1表示未参与传播，0表示leader层
    received_news: List[News] = field(default_factory=list)  # 接收的新闻总列表
    skip_decay_this_round: bool = False  # 因LLM故障跳过本轮衰减
    
@dataclass
class AgentSnapshot:
    """智能体快照，用于记录智能体在每轮的状态"""
    round_id: int # 回合ID
    agent_id: str # 智能体ID
    profile: AgentProfile  # 静态配置
    memory: AgentMemory  # 动态记忆状态

@dataclass
class AgentAction:
    """智能体动作"""
    agent_id: str # 智能体ID
    action: str # 动作类型
    action_details: Dict[str, Any] = None # 动作详情
@dataclass
class DecisionRecord:
    """决策记录，用于追踪智能体的关键决策过程"""
    decision_type: str  # 决策类型：news_processing, social_interaction, discussion_participation
    context: Dict[str, Any]  # 决策上下文
    options_considered: List[str]  # 考虑的选项
    final_decision: str  # 最终决策
    reasoning: str  # 决策推理过程
    timestamp: str  # 决策时间戳

@dataclass
class DiscussionParticipation:
    """讨论参与记录，用于追踪智能体在群组讨论中的参与情况"""
    discussion_id: str  # 讨论组ID
    agent_id: str  # 参与智能体ID
    join_time: str  # 加入讨论时间
    utterance_count: int  # 发言次数
    topics_discussed: List[str]  # 讨论的话题列表
    influence_score: float  # 在讨论中的影响力评分
    final_stance: str  # 讨论后的最终立场

@dataclass
class RoundActivityLog:
    """单轮活动日志，记录智能体在一轮中的所有活动"""
    round_idx: int  # 轮次索引
    news_interactions: List[NewsInteraction] = field(default_factory=list)  # 新闻交互记录
    social_changes: List[SocialRelationshipChange] = field(default_factory=list)  # 社交关系变化
    discussion_participations: List[DiscussionParticipation] = field(default_factory=list)  # 讨论参与记录
    decisions: List[DecisionRecord] = field(default_factory=list)  # 决策记录
    memory_changes: Dict[str, Any] = field(default_factory=dict)  # 记忆变化记录
    summary: str = ""  # 本轮活动摘要
