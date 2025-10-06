from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import deque
from typing import Deque
import datetime

@dataclass
class Score:
    trust: float
    interest: float

@dataclass
class News:
    news_id: str
    sender_id: str
    receiver_id: str
    content: str
    topic: str

@dataclass
class TransformedNews:
    news_id: str
    editor_id: str
    content: str
    original_news_id: str = ""
    transformer_agent_id: str = ""
    transform_time: str = ""
    propagation_layer: int = 0
    confidence_score: float = 1.0

@dataclass
class WorkingMemoryEntry:
    news_id: str
    topic: str
    summary: str
    source_id: str
    round_id: int
    timestamp: str


@dataclass
class ShortTermMemoryCluster:
    topic: str
    summary: str
    count: int = 0


@dataclass
class NewsInteraction:
    news_id: str
    from_id: str
    content: str
    received_at: str
    action: str
    action_details: Optional[Dict[str, Any]] = None

@dataclass
class SocialRelationshipChange:
    target_agent_id: str
    action: str
    previous_score: Optional[Score] = None
    new_score: Optional[Score] = None
    reason: str = ""
    timestamp: str = ""

@dataclass
class TalkInTurn:
    turn_id: str
    agent_id: str
    content: str

@dataclass
class DiscussionRec:
    group_id: str
    round_num: int
    news_list:List[str]
    agents_id: List[str]
    turns: List[TalkInTurn]
    summary: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DiscussionMem:
    agent_id: str
    news_list:List[str]
    discuss_mem: List[DiscussionRec]


@dataclass
class AgentProfile:
    agent_id: str
    is_leader: bool
    name:str
    age:int
    education_level: str
    gender: str
    traits:List[str]
    political_affiliation: str
    self_awareness: str
    _agent_prompt_template: str = "Imagine you are a human. Your name is {name}, and your gender is {gender}.You are {age} years old. Your personality is shaped by these specific traits:{traits_str}.Your educational background is at the level of {education_level} Degree.Your political affiliation is {political_affiliation}.Act according to this human identity, letting these details fully define your thoughts, responses, interactions, and decisions."
    agent_prompt: str = ""
    
    def __agent_person_init__(self):
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
    agent_id: str
    following_list: Dict[str, Score]
    access_id: List[str]
    access_news:List[News]
    published_news:List[TransformedNews]
    discuss_mem:List[DiscussionRec]
    received_news_inround:List[News]
    published_news_inround:List[TransformedNews]
    working_memory: List[WorkingMemoryEntry] = field(default_factory=list)
    short_term_clusters: Dict[str, ShortTermMemoryCluster] = field(default_factory=dict)
    short_term_memory:Deque[str] = field(default_factory=lambda: deque(maxlen=3))
    long_term_memory:str = ""
    
    has_published_this_round: bool = False
    propagation_layer: int = -1
    received_news: List[News] = field(default_factory=list)
    skip_decay_this_round: bool = False
    
@dataclass
class AgentSnapshot:
    round_id: int
    agent_id: str
    profile: AgentProfile
    memory: AgentMemory

@dataclass
class AgentAction:
    agent_id: str
    action: str
    action_details: Dict[str, Any] = None
@dataclass
class DecisionRecord:
    decision_type: str
    context: Dict[str, Any]
    options_considered: List[str]
    final_decision: str
    reasoning: str
    timestamp: str

@dataclass
class DiscussionParticipation:
    discussion_id: str
    agent_id: str
    join_time: str
    utterance_count: int
    topics_discussed: List[str]
    influence_score: float
    final_stance: str

@dataclass
class RoundActivityLog:
    round_idx: int
    news_interactions: List[NewsInteraction] = field(default_factory=list)
    social_changes: List[SocialRelationshipChange] = field(default_factory=list)
    discussion_participations: List[DiscussionParticipation] = field(default_factory=list)
    decisions: List[DecisionRecord] = field(default_factory=list)
    memory_changes: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
