from datetime import datetime
import argparse
import asyncio
import os
import json
from collections import deque
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any
import logging
from dataclasses import dataclass

from dataloader import load_graph_from_file, load_agent_profile, load_news, save_graph, save_agent_profile, sync_edges_from_agents
from dataclass import AgentSnapshot, News, TransformedNews, DiscussionRec, TalkInTurn, AgentMemory, AgentProfile, Score, WorkingMemoryEntry, ShortTermMemoryCluster
from servellm import submit, call_openai_api, format_messages
from rec import get_rec
from propagation import news_propagation, reset_propagation_state
from discuss import generate_utterance, run_discussion_group
from political_shift import (
    capture_affiliation_snapshot,
    summarize_political_shift,
    dump_round_report,
)

from update_mem import update_mem
from traits_utils import normalize_traits_list

# 设置默认运行目录
DEFAULT_RUN_DIR = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")

@dataclass
class LLMConfig:
    """LLM配置管理类"""
    base_url: str = "http://localhost:8001/v1"
    api_key: str = "EMPTY"
    model: str = "qwen32"
    concurrency: int = 32
    timeout: float = 30000
    max_retries: int = 3
    temperature: float = 0.9
    max_tokens: int = 16384
    batch_size: int = 4
    queue_size: int = 512

# 日志配置
def setup_logging() -> logging.Logger:
    """
    设置日志记录器
    Returns:
        配置好的日志记录器
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("MultiAgentSimulation")

# LLM提供方类
class OpenAIProvider:
    """OpenAI兼容接口提供方"""
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger("OpenAIProvider")
    
    async def generate(self, messages: List[Dict], **kwargs) -> str:
        """生成响应，直接调用servellm.py的API函数"""
        # 合并配置参数
        api_params = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            **kwargs
        }
        
        # 调用servellm.py中的实际API函数
        response = await call_openai_api(
            messages=messages,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            model=self.config.model,
            **api_params
        )
        
        if response is None:
            self.logger.error("API call returned an empty response")
            return ""
        
        return response

class LLMExecutor:
    """LLM 执行器，使用任务队列控制并发。"""

    def __init__(
        self,
        provider,
        max_concurrency: int = 16,
        batch_size: int = 1,
        queue_size: int = 0,
        logger=None,
    ):
        self.provider = provider
        self.batch_size = max(1, batch_size)
        self.logger = logger or logging.getLogger("LLMExecutor")
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size or 0)
        self._closed = False
        self._workers: List[asyncio.Task] = []
        loop = asyncio.get_running_loop()
        for idx in range(max(1, max_concurrency)):
            self._workers.append(loop.create_task(self._worker(idx)))

    async def execute(self, messages: List[Dict], **kwargs):
        if self._closed:
            raise RuntimeError("LLMExecutor 已关闭")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put((messages, kwargs, future))
        return await future

    async def _worker(self, worker_id: int):
        while True:
            try:
                job = await self.queue.get()
            except asyncio.CancelledError:
                break

            batch = [job]
            if self.batch_size > 1:
                for _ in range(self.batch_size - 1):
                    try:
                        batch.append(self.queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

            results: List[Any] = []
            for messages, kwargs, future in batch:
                if future.cancelled():
                    results.append(asyncio.CancelledError())
                    continue
                try:
                    result = await self.provider.generate(messages, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    results.append(exc)
                else:
                    results.append(result)

            for (messages, kwargs, future), result in zip(batch, results):
                if future.cancelled():
                    pass
                elif isinstance(result, Exception):
                    future.set_exception(result)
                else:
                    future.set_result(result)
                self.queue.task_done()

        self.logger.debug("LLMExecutor worker %d stopped", worker_id)

    async def shutdown(self):
        if self._closed:
            return
        self._closed = True
        await self.queue.join()
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。
    
    Returns:
        argparse.Namespace: 包含所有命令行参数的命名空间对象。
    """
    p = argparse.ArgumentParser(description="多智能体仿真")
    p.add_argument("--edge_file", default="data/test_data/12831.edges", help="边列表文件")
    p.add_argument("--persona_file", default="data/agent_background.json", help="智能体背景JSON文件")
    p.add_argument("--news_file", default="data/synthetic_news.json", help="新闻JSON列表文件（可选）")
    p.add_argument("--rounds", type=int, default=100, help="仿真轮数")
    p.add_argument("--output_dir", default=str(DEFAULT_RUN_DIR), help="输出目录")
    p.add_argument("--recommendation_strategy", choices=["llm", "random", "none"], default="llm", 
                   help="推荐策略选择：llm=LLM推荐（默认），random=随机推荐，none=leader接收所有新闻（消融实验用）")
    p.add_argument("--resume_round", type=int, help="恢复的轮次编号")
    p.add_argument("--resume_dir", help="恢复的运行目录路径")
    p.add_argument("--topic", default="", help="当前运行所属的议题标签")
    return p.parse_args()


def load_resume_state(resume_dir: str, resume_round: int) -> Dict[str, AgentSnapshot]:
    """从指定轮次恢复智能体状态。"""
    import json

    round_dir = Path(resume_dir) / f"round_{resume_round:03d}"
    profile_file = round_dir / f"round_{resume_round}_profile.json"
    mem_file = round_dir / f"round_{resume_round}_mem.json"

    with open(profile_file, 'r', encoding='utf-8') as f:
        profile_data = json.load(f)

    with open(mem_file, 'r', encoding='utf-8') as f:
        memory_data = json.load(f)

    def _restore_news(record: Dict[str, Any]) -> News:
        return News(
            news_id=str(record.get('news_id', '')),
            sender_id=record.get('sender_id', ''),
            receiver_id=record.get('receiver_id', ''),
            content=record.get('content', ''),
            topic=record.get('topic', ''),
        )

    def _restore_transformed(record: Dict[str, Any]) -> TransformedNews:
        return TransformedNews(
            news_id=str(record.get('news_id', '')),
            editor_id=record.get('editor_id', ''),
            content=record.get('content', ''),
            original_news_id=str(record.get('original_news_id', '')),
            transformer_agent_id=record.get('transformer_agent_id', ''),
            transform_time=record.get('transform_time', ''),
            propagation_layer=int(record.get('propagation_layer', 0)),
            confidence_score=float(record.get('confidence_score', 1.0)),
        )

    def _restore_discussions(records: List[Dict[str, Any]]) -> List[DiscussionRec]:
        discussions: List[DiscussionRec] = []
        for rec in records:
            turns = [
                TalkInTurn(
                    turn_id=turn.get('turn_id', ''),
                    agent_id=turn.get('agent_id', ''),
                    content=turn.get('content', ''),
                )
                for turn in rec.get('turns', [])
            ]
            discussions.append(
                DiscussionRec(
                    group_id=rec.get('group_id', ''),
                    round_num=int(rec.get('round_num', resume_round)),
                    news_list=[str(n) for n in rec.get('news_list', [])],
                    agents_id=[str(a) for a in rec.get('agents_id', [])],
                    turns=turns,
                    summary=rec.get('summary', {}),
                )
            )
        return discussions

    agents: Dict[str, AgentSnapshot] = {}

    for agent_id, profile_info in profile_data.items():
        traits_payload = profile_info.get('traits', [])
        agent_profile = AgentProfile(
            agent_id=profile_info.get('agent_id', agent_id),
            is_leader=profile_info.get('is_leader', False),
            name=profile_info.get('name', ''),
            age=profile_info.get('age', 0),
            education_level=profile_info.get('education_level', ''),
            gender=profile_info.get('gender', ''),
            traits=normalize_traits_list(traits_payload),
            political_affiliation=profile_info.get('political_affiliation', ''),
            self_awareness=profile_info.get('self_awareness', ''),
        )
        agent_profile.__agent_person_init__()

        mem_info = memory_data.get(agent_id, {})

        following_list = {}
        for target_id, score_data in mem_info.get('following_list', {}).items():
            following_list[target_id] = Score(
                trust=float(score_data.get('trust', 0.5)),
                interest=float(score_data.get('interest', 0.5)),
            )

        access_news = [_restore_news(rec) for rec in mem_info.get('access_news', [])]
        published_news = [
            _restore_transformed(rec) for rec in mem_info.get('published_news', [])
        ]
        discuss_mem = _restore_discussions(mem_info.get('discuss_mem', []))
        received_news_inround = [
            _restore_news(rec) for rec in mem_info.get('received_news_inround', [])
        ]
        published_news_inround = [
            _restore_transformed(rec)
            for rec in mem_info.get('published_news_inround', [])
        ]
        received_news = [_restore_news(rec) for rec in mem_info.get('received_news', [])]

        working_memory = [
            WorkingMemoryEntry(
                news_id=record.get('news_id', ''),
                topic=record.get('topic', ''),
                summary=record.get('summary', ''),
                source_id=record.get('source_id', ''),
                round_id=int(record.get('round_id', resume_round)),
                timestamp=record.get('timestamp', ''),
            )
            for record in mem_info.get('working_memory', [])
        ]
        short_term_clusters = {
            topic: ShortTermMemoryCluster(
                topic=record.get('topic', topic),
                summary=record.get('summary', ''),
                count=int(record.get('count', 0)),
            )
            for topic, record in mem_info.get('short_term_clusters', {}).items()
        }

        short_term_memory = deque(mem_info.get('short_term_memory', []), maxlen=3)
        long_term_memory = mem_info.get('long_term_memory', '')

        agent_memory = AgentMemory(
            agent_id=mem_info.get('agent_id', agent_id),
            following_list=following_list,
            access_id=mem_info.get('access_id', []),
            access_news=access_news,
            published_news=published_news,
            discuss_mem=discuss_mem,
            received_news_inround=received_news_inround,
            published_news_inround=published_news_inround,
            working_memory=working_memory,
            short_term_clusters=short_term_clusters,
            short_term_memory=short_term_memory,
            long_term_memory=long_term_memory,
            has_published_this_round=mem_info.get('has_published_this_round', False),
            propagation_layer=int(mem_info.get('propagation_layer', -1)),
            received_news=received_news,
            skip_decay_this_round=mem_info.get('skip_decay_this_round', False),
        )

        agent_snapshot = AgentSnapshot(
            round_id=resume_round,
            agent_id=agent_id,
            profile=agent_profile,
            memory=agent_memory,
        )

        agents[agent_id] = agent_snapshot

    return agents


def build_agents(edge_file: Path, persona_file: Path) -> Dict[str, AgentSnapshot]:
    """Construct agents from edge and persona data."""
    # 加载智能体profile
    agents = load_agent_profile(str(persona_file))
    
    # 加载图结构
    _, node_list, edge_list = load_graph_from_file(str(edge_file))
    
    # 为每个智能体建立社交关系
    for agent_id, agent in agents.items():
        if agent_id in node_list:
            # 获取该智能体关注的所有节点
            following = {}
            for source, target, trust, interest in edge_list:
                if source == agent_id:
                    following[target] = Score(trust=trust, interest=interest)
            
            # 更新智能体的关注列表
            agent.memory.following_list = following
    
    print(f"[Agent Builder] Created {len(agents)} agents")
    return agents

def build_provider(config: LLMConfig, logger: logging.Logger = None) -> LLMExecutor:
    """构建LLM提供方实例
    
    Args:
        config: LLM配置对象
        logger: 日志记录器
    
    Returns:
        包装好并发控制的LLMExecutor对象
    """
    provider = OpenAIProvider(config)
    return LLMExecutor(
        provider,
        max_concurrency=config.concurrency,
        batch_size=config.batch_size,
        queue_size=config.queue_size,
        logger=logger,
    )


"""
=== 恢复功能使用说明 ===

本系统支持从任意轮次恢复仿真，用于处理中途运行失败的情况。

## 参数说明
--resume_round: 指定要恢复的轮次编号（整数）
--resume_dir: 指定恢复的运行目录路径（字符串）
--rounds: 从恢复轮次开始，再执行的轮数（方案A语义）

## 使用示例

### 正常启动新仿真
python src/main.py --edge_file data/small_world_50.edges --persona_file data/50_profile.json --news_file data/combined_news.json --rounds 10

### 从第7轮恢复，再执行3轮（第8、9、10轮）
python src/main.py --edge_file data/small_world_50.edges --persona_file data/50_profile.json --news_file data/combined_news.json --resume_round 7 --resume_dir runs/20250726_234223 --rounds 3

## 执行逻辑
- 恢复模式：加载第N轮的完整状态（智能体profile+memory+网络），从第N+1轮开始继续执行
- 输出目录：复用原运行目录，从恢复轮次+1开始覆盖后续数据
- 轮数计算：--rounds表示从恢复点再执行多少轮，而不是执行到第几轮

## 文件要求
恢复需要以下文件存在：
- {resume_dir}/round_{resume_round:03d}/round_{resume_round}_profile.json
- {resume_dir}/round_{resume_round:03d}/round_{resume_round}_edges.edges  
- {resume_dir}/round_{resume_round:03d}/round_{resume_round}_mem.json

## 注意事项
- 文件不存在时会自然报错，便于调试
- 恢复轮次编号从0开始计数
- 确保原始数据文件（edge_file, persona_file, news_file）与原仿真一致
"""
async def main_async():
    """Entry point orchestrating the round-based simulation."""
    args=parse_args()
    edge_file=Path(args.edge_file)
    persona_file=Path(args.persona_file)
    news_file=Path(args.news_file)
    run_dir=Path(args.output_dir)
    topic_label = args.topic.strip() or None
    run_dir.mkdir(parents=True, exist_ok=True)

    # 解析消融模式配置
    def _parse_flag(value: str | None, fallback: bool = False) -> bool:
        if value is None or value.strip() == "":
            return fallback
        return value.strip().lower() in {"1", "true", "yes", "on"}

    ablation_mode = (os.getenv("UNICOON_ABLATION_MODE") or "").strip().lower()
    if not ablation_mode:
        ablation_mode = "social_only" if args.recommendation_strategy == "none" else "full"

    disable_discussion = _parse_flag(
        os.getenv("UNICOON_DISABLE_DISCUSSION"),
        fallback=ablation_mode == "social_only",
    )
    disable_propagation = _parse_flag(
        os.getenv("UNICOON_DISABLE_PROPAGATION"),
        fallback=ablation_mode == "rec_only",
    )
    
    # 直接使用默认LLM配置
    llm_config = LLMConfig()
    llm_config.base_url = os.getenv("LLM_BASE_URL", llm_config.base_url)
    llm_config.api_key = os.getenv("LLM_API_KEY", llm_config.api_key)
    llm_config.model = os.getenv("LLM_MODEL", llm_config.model)
    llm_config.concurrency = int(os.getenv("LLM_CONCURRENCY", llm_config.concurrency))
    llm_config.batch_size = int(os.getenv("LLM_BATCH_SIZE", llm_config.batch_size))
    llm_config.queue_size = int(os.getenv("LLM_QUEUE_SIZE", llm_config.queue_size))
    llm_config.timeout = float(os.getenv("LLM_TIMEOUT", llm_config.timeout))
    llm_config.max_tokens = int(os.getenv("LLM_MAX_TOKENS", llm_config.max_tokens))
    
    # 初始化日志系统
    logger = setup_logging()
    
    # 构建LLM执行器
    executor = build_provider(llm_config, logger=logger)
    try:
        print(f"[LLM Config] URL: {llm_config.base_url}, model: {llm_config.model}, concurrency: {llm_config.concurrency}, batch size: {llm_config.batch_size}, queue: {llm_config.queue_size}")
    
        # 检测恢复模式
        if args.resume_round is not None:
            print(f"[Resume Mode] Resuming from round {args.resume_round} in {args.resume_dir}")
            agents = load_resume_state(args.resume_dir, args.resume_round)
            start_round = args.resume_round + 1
            run_dir = Path(args.resume_dir)  # 复用原目录
        else:
            print("[Normal Mode] Starting a new simulation")
            agents = build_agents(edge_file, persona_file)
            start_round = 0
            meta_path = run_dir / "run_meta.json"
            if not meta_path.exists():
                run_metadata = {
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "topic": topic_label,
                    "edge_file": str(edge_file),
                    "persona_file": str(persona_file),
                    "news_file": str(news_file),
                    "rounds": args.rounds,
                    "llm": {
                        "base_url": llm_config.base_url,
                        "model": llm_config.model,
                        "temperature": llm_config.temperature,
                        "max_tokens": llm_config.max_tokens,
                    },
                }
                with meta_path.open("w", encoding="utf-8") as fp:
                    json.dump(run_metadata, fp, indent=2, ensure_ascii=False)
    
        graph, _, current_edge_list = load_graph_from_file(str(edge_file))

        for r in range(start_round, start_round + args.rounds):
            print(f"\n[Round {r+1}/{start_round + args.rounds}] Simulation start")

            affiliation_baseline = capture_affiliation_snapshot(agents)

            # 加载新闻
            news_list = load_news(str(news_file), r * 10, (r + 1) * 10)  # 每轮10条新闻

            # 构建超边 (推荐阶段)
            hyperedges = await get_rec(agents, news_list, executor, args.recommendation_strategy)

            recommendation_log = []

            # 将推荐结果分配给智能体的received_news字段
            for hyperedge in hyperedges:
                recommendation_log.append({
                    "news": [
                        {
                            "news_id": news_item.news_id,
                            "topic": news_item.topic,
                            "sender": news_item.sender_id,
                        }
                        for news_item in hyperedge.news_items
                    ],
                    "agents": hyperedge.agents_id,
                })
                for agent_id in hyperedge.agents_id:
                    if agent_id in agents:
                        # 为该智能体分配超边对应的新闻
                        for original_news in hyperedge.news_items:
                            recommended_news = News(
                                news_id=original_news.news_id,
                                sender_id="sys",
                                receiver_id=agent_id,
                                content=original_news.content,
                                topic=original_news.topic,
                            )
                            agents[agent_id].memory.received_news.append(recommended_news)

            # 新闻传播改写
            if disable_propagation:
                logger.info("[Round %d] Propagation disabled by configuration", r)
                reset_propagation_state(agents)
                for agent in agents.values():
                    if agent.memory.received_news:
                        agent.memory.received_news_inround.extend(agent.memory.received_news)
                    agent.memory.received_news.clear()
                propagation_log = {
                    "round": r,
                    "status": "skipped",
                    "reason": "propagation_disabled",
                }
            else:
                propagation_log = await news_propagation(agents, graph, r, executor)

            # 群组划分，超边中选取2-10个智能体组成小组
            discussion_logs: list[Dict[str, Any]] = []
            discussion_logs_for_dump: list[Dict[str, Any]] = []
            if disable_discussion:
                logger.info("[Round %d] Discussion disabled by configuration", r)
            else:
                discussion_groups: list[tuple[list[str], list[News]]] = []
                for hyperedge in hyperedges:
                    size = len(hyperedge.agents_id)
                    if 2 <= size <= 10:
                        discussion_groups.append((hyperedge.agents_id, hyperedge.news_items))
                    elif size > 10:
                        agents_list = hyperedge.agents_id
                        for i in range(0, size, 8):
                            group_agents = agents_list[i:i + 8]
                            if len(group_agents) >= 2:
                                discussion_groups.append((group_agents, hyperedge.news_items))

                discussion_tasks = []
                for i, (members, items) in enumerate(discussion_groups):
                    task = run_discussion_group(
                        members,
                        f"group_{r}_{i}",
                        agents,
                        executor,
                        items
                    )
                    discussion_tasks.append(task)

                discussion_logs = await asyncio.gather(*discussion_tasks)

                for log in discussion_logs:
                    if log:
                        discussion_record = {
                            "round": r,
                            "group_id": log["group_id"],
                            "participants": log["members"],
                            "discussed_news": log["news_topics"],
                            "total_turns": len(log["turns"]),
                            "discussion_log": log["turns"],
                            "summary": log.get("summary", {}),
                        }
                        discussion_logs_for_dump.append(discussion_record)

            # 持久化小组讨论结果
            round_dir = run_dir / f"round_{r:03d}"
            round_dir.mkdir(exist_ok=True)

            # 保存讨论记录到轮次专属文件
            discussion_file = round_dir / f"round_{r}_discussions.json"
            with open(discussion_file, 'w', encoding='utf-8') as f:
                json.dump(discussion_logs_for_dump, f, indent=2, ensure_ascii=False)

            # 记忆更新和整理
            activities_log = await update_mem(r, agents, executor)

            politics_summary = summarize_political_shift(affiliation_baseline, agents, r, topic=topic_label)
            dump_round_report(round_dir, politics_summary)

            with open(round_dir / f"round_{r}_recommendations.json", 'w', encoding='utf-8') as f:
                json.dump(recommendation_log, f, indent=2, ensure_ascii=False)

            with open(round_dir / f"round_{r}_propagation.json", 'w', encoding='utf-8') as f:
                json.dump(propagation_log, f, indent=2, ensure_ascii=False)

            with open(round_dir / f"round_{r}_activities.json", 'w', encoding='utf-8') as f:
                json.dump(activities_log, f, indent=2, ensure_ascii=False)

            # 从智能体内存中同步最新的评分到边列表，并更新下一轮所用的图
            if disable_propagation:
                logger.info("[Round %d] Graph rewiring skipped by propagation setting", r)
            else:
                updated_edge_list = sync_edges_from_agents(agents)
                current_edge_list = updated_edge_list
                graph = nx.DiGraph()
                for source, target, trust, interest in updated_edge_list:
                    if source == target:
                        continue
                    graph.add_edge(source, target, trust=trust, interest=interest)
                for agent_id in agents:
                    graph.add_node(agent_id)

            save_agent_profile(agents, r, str(round_dir))
            save_graph(current_edge_list, r, str(round_dir))

            print(f"[Round {r+1}] Completed; results stored at: {round_dir}")
    finally:
        await executor.shutdown()


if __name__ == "__main__":
    asyncio.run(main_async())
