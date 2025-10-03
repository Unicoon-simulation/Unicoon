#!/usr/bin/env python3
"""
智能体背景生成脚本（为small_world_network.edges定制）

功能特点：
1. 支持随机生成或 LLM 生成两种方式
2. 保留断点续传和流式写入
3. 输出格式为 JSON
4. 自动从边文件读取节点数量
5. 针对small_world_network.edges文件的20个节点生成对应智能体背景
"""

import asyncio
import json
import os
import logging
import random
from datetime import datetime
from typing import Dict, List, Any

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== 配置参数 ==========
GENERATION_MODE = "random"  # 可选: "random", "llm"
OUTPUT_FILE = "data/small_world_profile.json"
EDGE_FILE = "data/test_data/small_world_network.edges"  # 用于确定节点数量
BATCH_SIZE = 5
POLITICAL_DISTRIBUTION = {
    "far_left": 0.15,
    "left": 0.25,
    "moderate": 0.20,
    "right": 0.25,
    "far_right": 0.15
}
RESUME = True  # 是否从断点继续生成
USE_LOCAL_LLM = True  # 是否使用本地模型
LOCAL_LLM_URL = "http://58.135.84.152:12345/v1"  # 本地模型地址
LOCAL_LLM_MODEL = "qwen3-235b"  # 模型名称
# =============================

if USE_LOCAL_LLM:
    try:
        import openai
    except ImportError:
        logger.warning("openai库未安装，LLM模式将不可用")
        USE_LOCAL_LLM = False


class BatchAgentGenerator:
    def __init__(self, batch_size: int = 5):
        self.batch_size = batch_size
        self.output_file = OUTPUT_FILE
        self.progress_file = OUTPUT_FILE + ".progress"

    def _create_batch_prompt(self, batch_specs: List[Dict]) -> str:
        """创建批量生成提示"""
        batch_info = []
        for i, spec in enumerate(batch_specs):
            batch_info.append(f"""
Agent {i+1}:
- Agent ID: {spec['agent_id']}
- Gender: {spec['gender']}
- Education level: {spec['education']}
- Political affiliation: {spec['political_affiliation']}
- Age group: {spec['age_group']}
""")
        
        prompt = f"""Generate detailed background information for {len(batch_specs)} agents. Each agent must include the following fields and return as a JSON object:

{batch_info}

Requirements:
1. Return format: JSON object with a "backgrounds" key.
2. Each agent must include: name, age, gender, education level, traits, political_affiliation, system_prompt, self-awareness
3. Traits should be a list with a single string containing 5 comma-separated traits
4. System prompt should be a string with the format:
    Imagine you are a human. Your name is Agent_X, and your gender is Y.
    You are Z years old. Your personality is shaped by these specific traits: {{traits}}.
    Your educational background is at the level of {{education level}}.
    Your political affiliation is {{political_affiliation}}.
    Act according to this human identity, letting these details fully define your thoughts, responses, interactions, and decisions.
5. Self-awareness should describe how the agent behaves online, considering their political views.
6. Political affiliation must match the specified value exactly.

Return example:
{{
  "backgrounds": {{
    "0": {{
      "name": "Agent_0",
      "age": 49,
      "education level": "Bachelor's Degree",
      "traits": ["Naturalness, Punctuality, Inhibition, Emotionality, Unimaginativeness"],
      "gender": "male",
      "political_affiliation": "moderate",
      "system_prompt": "...",
      "self-awareness": "..."
    }}
  }}
}}
Ensure the response is valid JSON object."""
        return prompt

    async def _call_llm_batch(self, prompt: str) -> Dict[str, Any]:
        """调用本地 LLM 生成响应"""
        if not USE_LOCAL_LLM:
            raise Exception("LLM 调用未启用，请设置 USE_LOCAL_LLM=True")

        client = openai.AsyncOpenAI(base_url=LOCAL_LLM_URL, api_key="not-needed")
        try:
            response = await client.chat.completions.create(
                model=LOCAL_LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个智能体背景生成器。请根据用户的提示生成符合要求的 JSON 格式数据。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2048,
                response_format={"type": "text"}
            )
            content = response.choices[0].message.content.strip()
            # 尝试解析JSON
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            return json.loads(content)
        except Exception as e:
            logger.error(f"调用 LLM 失败: {e}")
            return {"backgrounds": {}}

    def _generate_random_agent(self, spec: Dict) -> Dict:
        """随机生成智能体背景"""
        trait_list = [
            "Naturalness", "Punctuality", "Inhibition", "Emotionality",
            "Unimaginativeness", "Openness", "Introversion", "Curiosity",
            "Assertiveness", "Cautiousness", "Empathy", "Creativity",
            "Conscientiousness", "Agreeableness", "Neuroticism", "Optimism"
        ]
        
        # 根据政治倾向调整性格特征倾向
        political_traits = {
            "far_left": ["Activism", "Idealism", "Collectivism", "Progressiveness", "Anti_establishment"],
            "left": ["Empathy", "Social_Justice", "Cooperation", "Environmentalism", "Equality"],
            "moderate": ["Pragmatism", "Compromise", "Stability", "Reasonableness", "Balance"],
            "right": ["Traditionalism", "Individual_Responsibility", "Conservatism", "Order", "Patriotism"],
            "far_right": ["Nationalism", "Authoritarianism", "Traditionalism", "Anti_globalism", "Strong_Leadership"]
        }
        
        # 混合政治特征和通用特征
        available_traits = trait_list + political_traits.get(spec['political_affiliation'], [])
        trait_string = ", ".join(random.sample(available_traits, min(5, len(available_traits))))

        # 根据年龄组生成具体年龄
        age_ranges = {
            "young": (18, 35),
            "middle": (36, 55),
            "senior": (56, 75)
        }
        age_range = age_ranges.get(spec['age_group'], (18, 70))
        age = random.randint(*age_range)

        system_prompt = (
            f"Imagine you are a human. Your name is Agent_{spec['agent_id']}, and your gender is {spec['gender']}.\n"
            f"You are {age} years old. Your personality is shaped by these specific traits: {trait_string}.\n"
            f"Your educational background is at the level of {spec['education']}.\n"
            f"Your political affiliation is {spec['political_affiliation']}.\n"
            f"Act according to this human identity, letting these details fully define your thoughts, responses, interactions, and decisions."
        )

        # 根据政治倾向生成不同的自我认知
        political_awareness = {
            "far_left": [
                "I'm passionate about social justice and systemic change. I actively share content about inequality and progressive causes.",
                "I believe in radical reform and am not afraid to challenge mainstream narratives online.",
                "My posts often focus on workers' rights, environmental justice, and anti-establishment perspectives."
            ],
            "left": [
                "I care deeply about social issues and try to promote empathy and understanding in my online interactions.",
                "I share content about climate change, social equality, and community welfare.",
                "I believe in progressive policies and try to engage constructively with different viewpoints."
            ],
            "moderate": [
                "I try to see multiple sides of issues and avoid extreme positions in my online discussions.",
                "I focus on practical solutions and compromise rather than ideological purity.",
                "I share content from various sources and try to fact-check before posting."
            ],
            "right": [
                "I value traditional principles and individual responsibility in my online presence.",
                "I share content about family values, economic freedom, and constitutional rights.",
                "I believe in conservative policies and try to defend traditional viewpoints respectfully."
            ],
            "far_right": [
                "I'm deeply concerned about preserving traditional values and national sovereignty.",
                "I share content about patriotism, law and order, and traditional family structures.",
                "I believe in strong leadership and am skeptical of globalist influences."
            ]
        }
        
        self_awareness = random.choice(political_awareness.get(spec['political_affiliation'], [
            "I share my thoughts and interests online while trying to be respectful of others.",
            "I use social media to stay connected with friends and learn about current events.",
            "I'm thoughtful about what I post and try to contribute positively to online discussions."
        ]))

        return {
            "name": f"Agent_{spec['agent_id']}",
            "age": age,
            "education level": spec['education'],
            "traits": [trait_string],
            "gender": spec['gender'],
            "political_affiliation": spec['political_affiliation'],
            "system_prompt": system_prompt,
            "self-awareness": self_awareness
        }

    def _load_progress(self) -> set:
        """加载进度文件"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    return set(json.load(f))
            except:
                return set()
        return set()

    def _save_progress(self, completed: set):
        """保存进度"""
        with open(self.progress_file, 'w') as f:
            json.dump(list(completed), f)

    def _generate_political_assignments(self, node_count: int, distribution: Dict[str, float]) -> List[str]:
        """生成政治倾向分配"""
        affiliations = []
        weights = []
        for aff, weight in distribution.items():
            affiliations.append(aff)
            weights.append(weight)
        return random.choices(affiliations, weights=weights, k=node_count)

    async def generate_agents(self, node_count: int):
        """主生成函数"""
        logger.info(f"开始生成 {node_count} 个智能体背景...")
        
        generated_agents = {}
        completed_agents = self._load_progress() if RESUME else set()

        # 生成政治倾向分配
        political_assignments = self._generate_political_assignments(node_count, POLITICAL_DISTRIBUTION)
        
        # 为每个节点创建规格
        all_specs = []
        for i in range(node_count):
            if str(i) in completed_agents:
                continue
            spec = {
                'agent_id': str(i),
                'political_affiliation': political_assignments[i],
                'age_group': random.choice(['young', 'middle', 'senior']),
                'gender': random.choice(['male', 'female']),
                'education': random.choice(['High School', "Bachelor's Degree", "Master's Degree", "PhD"]),
            }
            all_specs.append(spec)

        total = len(all_specs)
        logger.info(f"需要生成 {total} 个智能体，每批 {self.batch_size} 个")

        # 如果启用断点续传，加载现有数据
        if os.path.exists(self.output_file) and RESUME:
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    generated_agents = data.get("backgrounds", {})
                    logger.info(f"从断点继续，已有 {len(generated_agents)} 个智能体")
            except Exception as e:
                logger.warning(f"加载现有数据失败: {e}")

        # 批量生成
        batch_count = (total + self.batch_size - 1) // self.batch_size
        for i in range(batch_count):
            start = i * self.batch_size
            end = min(start + self.batch_size, total)
            batch_specs = all_specs[start:end]
            
            logger.info(f"处理批次 {i+1}/{batch_count}，共 {len(batch_specs)} 个智能体")

            if GENERATION_MODE == "random":
                # 随机生成模式
                for spec in batch_specs:
                    agent = self._generate_random_agent(spec)
                    generated_agents[spec['agent_id']] = agent
                    completed_agents.add(spec['agent_id'])
                    logger.info(f"随机生成智能体 {spec['agent_id']} ({agent['political_affiliation']})")
                    
            elif GENERATION_MODE == "llm":
                # LLM生成模式
                if not USE_LOCAL_LLM:
                    logger.error("LLM模式未启用，切换到随机生成模式")
                    for spec in batch_specs:
                        agent = self._generate_random_agent(spec)
                        generated_agents[spec['agent_id']] = agent
                        completed_agents.add(spec['agent_id'])
                else:
                    try:
                        prompt = self._create_batch_prompt(batch_specs)
                        response = await self._call_llm_batch(prompt)
                        agents = response.get("backgrounds", {})
                        
                        # 处理LLM返回的结果
                        for spec in batch_specs:
                            agent_id = spec['agent_id']
                            if agent_id in agents:
                                agent = agents[agent_id]
                                # 确保政治倾向正确
                                agent['political_affiliation'] = spec['political_affiliation']
                                generated_agents[agent_id] = agent
                                completed_agents.add(agent_id)
                                logger.info(f"LLM生成智能体 {agent_id} ({agent['political_affiliation']})")
                            else:
                                # LLM生成失败，使用随机生成作为备选
                                logger.warning(f"LLM未能生成智能体 {agent_id}，使用随机生成")
                                agent = self._generate_random_agent(spec)
                                generated_agents[agent_id] = agent
                                completed_agents.add(agent_id)
                                
                    except Exception as e:
                        logger.error(f"LLM生成失败: {e}，使用随机生成")
                        for spec in batch_specs:
                            agent = self._generate_random_agent(spec)
                            generated_agents[spec['agent_id']] = agent
                            completed_agents.add(spec['agent_id'])
            else:
                raise ValueError(f"不支持的生成模式: {GENERATION_MODE}")

            # 保存进度
            output_data = {"backgrounds": generated_agents}
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            self._save_progress(completed_agents)
            logger.info(f"已生成 {len(completed_agents)} / {node_count} 个智能体")

        # 清理进度文件
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)
            
        logger.info("智能体背景生成完成!")
        
        # 统计政治倾向分布
        political_stats = {}
        for agent in generated_agents.values():
            affiliation = agent.get('political_affiliation', 'unknown')
            political_stats[affiliation] = political_stats.get(affiliation, 0) + 1
        
        logger.info("政治倾向分布统计:")
        for affiliation, count in political_stats.items():
            percentage = (count / len(generated_agents)) * 100
            logger.info(f"  {affiliation}: {count} ({percentage:.1f}%)")


def read_node_count_from_edge_file(file_path: str) -> tuple:
    """从边文件中读取节点信息，返回(唯一节点数量, 最小节点ID, 最大节点ID)"""
    node_ids = set()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    src, dst = int(parts[0]), int(parts[1])
                    node_ids.add(src)
                    node_ids.add(dst)
                except ValueError:
                    continue
    
    if not node_ids:
        return 0, 0, 0
    
    unique_count = len(node_ids)
    min_id = min(node_ids)
    max_id = max(node_ids)
    return unique_count, min_id, max_id


async def main():
    """主函数"""
    logger.info("=== 智能体背景生成脚本 ===")
    logger.info(f"生成模式: {GENERATION_MODE}")
    logger.info(f"输入文件: {EDGE_FILE}")
    logger.info(f"输出文件: {OUTPUT_FILE}")
    
    # 读取节点信息
    node_count, min_id, max_id = read_node_count_from_edge_file(EDGE_FILE)
    logger.info(f"检测到唯一节点数量: {node_count}")
    logger.info(f"节点编号范围: {min_id} - {max_id}")
    
    if node_count == 0:
        logger.error("未检测到有效节点，请检查边文件格式")
        return
    
    # 创建生成器并开始生成
    generator = BatchAgentGenerator(batch_size=BATCH_SIZE)
    start_time = datetime.now()
    
    try:
        await generator.generate_agents(node_count)
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"总耗时: {elapsed:.2f} 秒")
        logger.info(f"平均每个智能体耗时: {elapsed/node_count:.2f} 秒")
    except KeyboardInterrupt:
        logger.info("用户中断生成，进度已保存")
    except Exception as e:
        logger.error(f"生成过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())