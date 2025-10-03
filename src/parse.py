import json
import re
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("parse")

def parse_recommend_response(
    response: str
) -> List[Dict[str, Any]]:
    """
    解析推荐服务的LLM响应，提取推荐结果。
    
    Args:
        response: LLM返回的原始响应字符串
        
    Returns:
        解析后的推荐结果列表，格式为 [{"news_id": "...", "reason": "...", "confidence": ...}]
    """
    if not response:
        logger.warning("Recommendation response is empty")
        return []
    
    json_data = extract_json_from_text(response)
    if json_data is None:
        logger.warning("Failed to parse recommendation response as JSON")
        return []
    
    if isinstance(json_data, list):
        return json_data
    if isinstance(json_data, dict) and "news_id" in json_data:
        return [json_data]

    logger.error("Recommendation response has unexpected format: %s", type(json_data))
    return []

def parse_news_rewrite_response(
    response: str
) -> Dict[str, Any]:
    """
    解析新闻改写服务的LLM响应，提取处理结果。
    
    Args:
        response: LLM返回的原始响应字符串
        
    Returns:
        解析后的处理结果字典，包含处理状态、处理结果等信息
    """
    if not response:
        logger.error("News rewrite response is empty")
        return {"status": "failed", "content": "", "reason": "response is empty"}
    
    # 尝试提取JSON格式的响应
    json_data = extract_json_from_text(response)
    
    processed_news = _extract_processed_news(json_data)

    if processed_news is not None:
        if isinstance(processed_news, list):
            for i, news_item in enumerate(processed_news):
                if not isinstance(news_item, dict):
                    logger.error(f"processed_news[{i}] is not a dict: {news_item}")
                    logger.error(f"Raw LLM response: {response}")
                    continue
                required_fields = ["news_id", "action", "reason"]
                for field in required_fields:
                    if field not in news_item:
                        logger.error(f"processed_news[{i}] is missing {field}: {news_item}")
                        logger.error(f"Raw LLM response: {response}")

                if news_item.get("action") not in ["rewrite", "skip"]:
                    logger.error(f"processed_news[{i}] has invalid action value: {news_item.get('action')}")
                    logger.error(f"Raw LLM response: {response}")

        return {
            "status": "success",
            "rewritten_news": processed_news,
            "original_response": response
        }
    
    # 如果无法解析JSON，尝试提取主要内容
    content = extract_main_content(response)
    return {
        "status": "partial",
        "action": "unknown",
        "content": content,
        "reason": "unable to fully parse JSON",
        "original_response": response
    }

def _extract_processed_news(json_data: Any) -> Optional[List[Dict[str, Any]]]:
    """Normalize different rewrite payload variants into a processed_news list."""
    if json_data is None:
        return None

    # Direct dict with expected key
    if isinstance(json_data, dict):
        if "processed_news" in json_data:
            return json_data["processed_news"]

        # Handle escaped or quoted key variants, e.g. "processed_news" or 'processed_news'
        for key in list(json_data.keys()):
            normalized = key.strip().strip('"').strip("'")
            if normalized == "processed_news":
                return json_data[key]

        # Fallback for legacy field name
        if "rewritten_news" in json_data:
            return json_data["rewritten_news"]

    # Some models may return the array directly
    if isinstance(json_data, list) and all(isinstance(item, dict) for item in json_data):
        return json_data

    return None

def parse_news_process_response(
    response: str
) -> Dict[str, Any]:
    """
    解析新闻处理服务的LLM响应，提取处理结果（向后兼容）。
    
    Args:
        response: LLM返回的原始响应字符串
        
    Returns:
        解析后的处理结果字典，包含处理状态、处理结果等信息
    """
    # 首先尝试使用新的news_rewrite解析器
    result = parse_news_rewrite_response(response)
    if result["status"] == "success":
        return result
    
    # 向后兼容：处理旧的transform格式
    if not response:
        logger.warning("News processing response is empty")
        return {"status": "failed", "content": "", "reason": "response is empty"}
    
    # 尝试提取JSON格式的响应
    json_data = extract_json_from_text(response)
    
    if json_data:
        # 优先处理传播所需的格式
        if "ratings_update" in json_data and "rewritten_news" in json_data:
            # 验证ratings_update数组中每个元素的必需字段
            ratings_update = json_data["ratings_update"]
            if isinstance(ratings_update, list):
                for i, rating in enumerate(ratings_update):
                    if not isinstance(rating, dict):
                        logger.error(f"ratings_update[{i}] is not a dict: {rating}")
                        logger.error(f"Raw LLM response: {response}")
                        continue
                    if "agent_id" not in rating:
                        logger.error(f"ratings_update[{i}] is missing agent_id: {rating}")
                        logger.error(f"Raw LLM response: {response}")
                    if "trust_score" not in rating:
                        logger.error(f"ratings_update[{i}] is missing trust_score: {rating}")
                    if "interest_score" not in rating:
                        logger.error(f"ratings_update[{i}] is missing interest_score: {rating}")
            
            return {
                "status": "success",
                "action": "transform", 
                "ratings_update": json_data["ratings_update"],
                "rewritten_news": json_data["rewritten_news"],
                "original_response": response
            }
        
        # 处理转换类型的响应
        if "transformed_content" in json_data:
            return {
                "status": "success",
                "action": "transform",
                "content": json_data["transformed_content"],
                "changes": json_data["key_changes"],
                "original_response": response
            }
        
        # 处理转发类型的响应
        elif "will_forward" in json_data:
            return {
                "status": "success",
                "action": "forward",
                "will_forward": json_data["will_forward"],
                "comment": json_data["comment"],
                "reason": json_data["reason"],
                "original_response": response
            }
        
        # 处理评论类型的响应
        elif "comment" in json_data:
            return {
                "status": "success",
                "action": "comment",
                "comment": json_data["comment"],
                "stance": json_data["stance"],
                "key_points": json_data["key_points"],
                "original_response": response
            }
    
    # 如果无法解析JSON，尝试提取主要内容
    content = extract_main_content(response)
    return {
        "status": "partial",
        "action": "unknown",
        "content": content,
        "reason": "unable to fully parse JSON",
        "original_response": response
    }

MAX_DISCUSSION_UTTERANCE_LEN = 400


def parse_group_discussion_response(
    response: str
) -> Dict[str, Any]:
    """
    解析群组讨论服务的LLM响应，提取讨论结果。
    使用JSONSchema验证器进行统一的JSON验证和修复。
    
    Args:
        response: LLM返回的原始响应字符串
        
    Returns:
        解析后的讨论结果字典，包含发言内容、情绪等信息
    """
    if not response:
        logger.warning("Group discussion response is empty")
        return {"status": "failed", "utterance": ""}
    
    # 使用新的JSONSchema验证器
    from json_validator import validator
    
    data, success, message = validator.validate_and_fix(response, "group_discussion")
    
    if success and data:
        utterance = data.get("utterance", "")[:MAX_DISCUSSION_UTTERANCE_LEN].strip()
        return {
            "status": "success",
            "utterance": utterance,
            "confidence": 1.0,
            "original_response": response
        }
    else:
        # 保留原始输出用于调试（满足用户要求）
        logger.error(f"Failed to parse group discussion JSON: {message}")
        logger.error(f"Raw LLM output: {response}")
        
        # 尝试备用解析方法
        utterance = extract_main_content(response)
    
    return {
        "status": "partial",
        "utterance": utterance[:MAX_DISCUSSION_UTTERANCE_LEN].strip(),
        "confidence": 0.5,
        "original_response": response
    }

def parse_update_mem_response(
    response: str
) -> Dict[str, Any]:
    """
    解析记忆更新服务的LLM响应，提取更新结果。
    
    Args:
        response: LLM返回的原始响应字符串
        
    Returns:
        解析后的更新结果字典，包含更新状态、新记忆等信息
    """
    if not response:
        logger.warning("Memory update response is empty")
        return {
            "status": "failed",
            "updated_long_term_memory": "",
            "new_short_term_memories": "",
            "score_updates": [],
            "trait_changes": {}
        }
    
    # 尝试提取JSON格式的响应
    json_data = extract_json_from_text(response)
    
    if json_data:
        # 解析trait_changes字段
        trait_changes = {}
        if "trait_changes" in json_data:
            trait_changes = json_data["trait_changes"]
            if isinstance(trait_changes, dict):
                # 验证political_affiliation值的有效性
                if "political_affiliation" in trait_changes:
                    normalized = normalize_political_affiliation(trait_changes["political_affiliation"])
                    if normalized:
                        trait_changes["political_affiliation"] = normalized
                    else:
                        logger.error(f"Invalid political_affiliation value: {trait_changes['political_affiliation']}")
                        logger.error("Valid values: far left, left, moderate, right, far right")
                        logger.error(f"Raw LLM response: {response}")
                
                # 验证traits字段类型
                if "traits" in trait_changes and not isinstance(trait_changes["traits"], list):
                    logger.error(f"trait_changes.traits must be a list: {trait_changes['traits']}")
                    logger.error(f"Raw LLM response: {response}")
                
                # 验证self_awareness字段类型
                if "self_awareness" in trait_changes and not isinstance(trait_changes["self_awareness"], str):
                    logger.error(f"trait_changes.self_awareness must be a string: {trait_changes['self_awareness']}")
                    logger.error(f"Raw LLM response: {response}")
            else:
                logger.error(f"trait_changes must be a dict: {trait_changes}")
                logger.error(f"Raw LLM response: {response}")
                trait_changes = {}
        
        # 验证和解析score_updates字段
        score_updates = json_data.get("score_updates", [])
        if isinstance(score_updates, list):
            for i, score_update in enumerate(score_updates):
                if not isinstance(score_update, dict):
                    logger.error(f"score_updates[{i}] is not a dict: {score_update}")
                    logger.error(f"Raw LLM response: {response}")
                    continue
                required_fields = ["agent_id", "trust", "interest"]
                for field in required_fields:
                    if field not in score_update:
                        logger.error(f"score_updates[{i}] is missing {field}: {score_update}")
                        logger.error(f"Raw LLM response: {response}")
        elif score_updates is not None:
            logger.error(f"score_updates must be a list: {score_updates}")
            logger.error(f"Raw LLM response: {response}")
            score_updates = []
        
        return {
            "status": "success",
            "updated_long_term_memory": json_data.get("long_term_memory_update", ""),
            "new_short_term_memories": json_data.get("short_term_memory_update", ""),
            "score_updates": score_updates,
            "trait_changes": trait_changes,
            "original_response": response
        }
    
    # 如果无法解析JSON，尝试从文本中提取关键信息
    summary = extract_main_content(response)
    
    return {
        "status": "partial",
        "updated_long_term_memory": summary,
        "new_short_term_memories": summary,
        "score_updates": [],
        "trait_changes": {},
        "original_response": response
    }



def parse_persuasion_response(response: str) -> Dict[str, Any]:
    """解析劝说提示的 LLM 响应。"""
    if not response:
        logger.warning("Persuasion response is empty")
        return {"status": "failed"}

    json_data = extract_json_from_text(response)
    if not json_data:
        logger.warning("Failed to parse persuasion response as JSON: %s", response)
        return {"status": "failed"}

    will = str(json_data.get("will", "no")).strip().lower()
    if will not in {"yes", "no"}:
        logger.warning("Persuasion response has invalid will value: %s", will)
        will = "no"
    message = json_data.get("message", "")
    if not isinstance(message, str):
        message = str(message)

    return {
        "status": "success",
        "will": will,
        "message": message.strip(),
    }

NORMALIZED_AFFILIATIONS = {
    "far left": "far left",
    "far_left": "far left",
    "farleft": "far left",
    "left": "left",
    "centre-left": "left",
    "center-left": "left",
    "moderate": "moderate",
    "centre": "moderate",
    "center": "moderate",
    "centrist": "moderate",
    "right": "right",
    "far right": "far right",
    "far_right": "far right",
    "farright": "far right",
}


AFFILIATION_PATTERNS = [
    (re.compile(r"\bfar[\s\-_]*left\b"), "far left"),
    (re.compile(r"\bfar[\s\-_]*right\b"), "far right"),
    (re.compile(r"\bleft\b"), "left"),
    (re.compile(r"\bright\b"), "right"),
    (re.compile(r"\bmoderate\b"), "moderate"),
    (re.compile(r"\bcent(?:er|re)\b"), "moderate"),
]


def normalize_political_affiliation(value: str) -> str | None:
    if not value:
        return None

    cleaned = value.strip().lower()

    direct_match = NORMALIZED_AFFILIATIONS.get(cleaned)
    if direct_match:
        return direct_match

    simplified = re.sub(r"[_\-]+", " ", cleaned)
    simplified = re.sub(r"\s+", " ", simplified).strip()
    direct_match = NORMALIZED_AFFILIATIONS.get(simplified)
    if direct_match:
        return direct_match

    matches: List[tuple[int, int, str]] = []
    for idx, (pattern, normalized) in enumerate(AFFILIATION_PATTERNS):
        match = pattern.search(cleaned)
        if match:
            matches.append((match.start(), idx, normalized))

    if matches:
        _, _, normalized = min(matches)
        return normalized

    return None


# 辅助函数


def extract_json_from_text(text: str) -> Optional[Any]:
    """从文本中提取 JSON 对象或数组，失败时返回 None。"""
    if not text:
        return None

    stripped = text.strip()

    if '```json' in stripped:
        start = stripped.find('```json') + len('```json')
        end = stripped.find('```', start)
        if end != -1:
            candidate = stripped[start:end].strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                logger.debug("Failed to parse markdown JSON block")

    if stripped.startswith('{') or stripped.startswith('['):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            logger.debug("Failed to parse JSON from entire response")
            # 尝试补齐缺失的尾部括号
            if stripped.startswith('{') and stripped.count('{') == stripped.count('}') + 1:
                try:
                    return json.loads(stripped + '}')
                except json.JSONDecodeError:
                    logger.debug("Failed after appending closing brace")
            if stripped.startswith('[') and stripped.count('[') == stripped.count(']') + 1:
                try:
                    return json.loads(stripped + ']')
                except json.JSONDecodeError:
                    logger.debug("Failed after appending closing bracket")

    brace_idx = stripped.find('{')
    bracket_idx = stripped.find('[')
    indices = [idx for idx in (brace_idx, bracket_idx) if idx != -1]
    if indices:
        start_idx = min(indices)
        candidate = stripped[start_idx:]
        for end_char in ('}', ']'):
            end_idx = candidate.rfind(end_char)
            if end_idx != -1:
                snippet = candidate[: end_idx + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    continue

    return None

def extract_news_numbers(text: str) -> List[int]:
    """
    从文本中提取新闻编号
    Args:
        text: 包含新闻编号的文本
    Returns:
        提取的新闻编号列表
    """
    # 查找数字模式
    patterns = [
        r'选择.*?(\d+).*?(\d+)',  # "选择1和3"
        r'(\d+)[,，]\s*(\d+)',    # "1, 3"
        r'第(\d+)条.*?第(\d+)条'   # "第1条和第3条"
    ]
    
    numbers = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                numbers.update(int(n) for n in match if n.isdigit())
            else:
                numbers.add(int(match))
    

    return sorted(list(numbers))

def extract_main_content(text: str) -> str:
    """
    从响应中提取主要内容
    Args:
        text: 原始响应文本
    Returns:
        提取的主要内容
    """
    # 移除常见的格式标记
    content = re.sub(r'```[a-z]*\n?', '', text)
    content = re.sub(r'\n+', ' ', content)
    content = content.strip()
    
    # 如果内容过长，截取前200个字符
    if len(content) > 1000:
        content = content[:1000] + "..."
    
    return content

def parse_mem_updatenew_response(
    response: str
) -> Dict[str, Any]:
    """
    解析增强版记忆更新服务的LLM响应，支持特征演化。
    使用JSONSchema验证器进行统一的JSON验证和修复。
    
    Args:
        response: LLM返回的原始响应字符串
        
    Returns:
        解析后的更新结果字典，包含记忆更新、社交评分和特征变化
    """
    if not response:
        logger.warning("Enhanced memory update response is empty")
        return {
            "status": "failed",
            "updated_long_term_memory": "",
            "new_short_term_memories": "",
            "score_updates": [],
            "trait_changes": {}
        }
    
    # 使用新的JSONSchema验证器
    from json_validator import validator
    
    data, success, message = validator.validate_and_fix(response, "mem_updatenew")
    
    if success and data:
        # 验证trait_changes中的political_affiliation
        trait_changes = data.get("trait_changes", {})
        if "political_affiliation" in trait_changes:
            normalized = normalize_political_affiliation(trait_changes["political_affiliation"])
            if normalized:
                trait_changes["political_affiliation"] = normalized
            else:
                logger.warning(f"Invalid political_affiliation value: {trait_changes['political_affiliation']}")
                # 保持原值而不是修改
        
        # 验证score_updates结构
        score_updates = data.get("score_updates", [])
        if isinstance(score_updates, list):
            for i, score_update in enumerate(score_updates):
                if not isinstance(score_update, dict):
                    logger.error(f"score_updates[{i}] is not a dict: {score_update}")
                    logger.error(f"Raw LLM output: {response}")
                    continue
                
                # 检查必需字段
                required_fields = ["agent_id", "trust", "interest"]
                for field in required_fields:
                    if field not in score_update:
                        logger.error(f"score_updates[{i}] is missing {field}: {score_update}")
                        logger.error(f"Raw LLM output: {response}")
                
                # 验证分数范围
                for score_field in ["trust", "interest"]:
                    if score_field in score_update:
                        score = score_update[score_field]
                        if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
                            logger.warning(f"score_updates[{i}].{score_field} is outside [0,1]: {score}")
        
        return {
            "status": "success",
            "updated_long_term_memory": data.get("long_term_memory_update", ""),
            "new_short_term_memories": data.get("short_term_memory_update", ""),
            "score_updates": score_updates,
            "trait_changes": trait_changes,
            "original_response": response
        }
    else:
        # 保留原始输出用于调试（满足用户要求）
        logger.error(f"Failed to parse enhanced memory update JSON: {message}")
        logger.error(f"Raw LLM output: {response}")
        
        # 尝试从文本中提取关键信息
        summary = extract_main_content(response)
        
        return {
            "status": "failed",
            "updated_long_term_memory": summary,
            "new_short_term_memories": summary,
            "score_updates": [],
            "trait_changes": {},
            "original_response": response
        }
