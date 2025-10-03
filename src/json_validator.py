"""
用于LLM响应的统一JSON验证和修复引擎。

该模块使用JSONSchema提供全面的JSON验证系统，
并具有针对常见LLM输出问题的智能修复能力。
"""

import json
import re
import logging
from typing import Dict, Any, Tuple, Optional, List
from jsonschema import validate, ValidationError
from schemas import SCHEMAS

logger = logging.getLogger("json_validator")

class JSONValidator:
    """
    具有渐进式修复能力的统一JSON验证器。

    该类处理LLM响应的所有JSON验证和修复，
    使用JSONSchema进行标准化验证，并实现针对常见格式问题的智能修复策略。
    """
    
    def __init__(self):
        """使用模式注册表初始化验证器。"""
        self.schemas = SCHEMAS
    
    def validate_and_fix(self, text: str, schema_name: str) -> Tuple[Optional[Dict], bool, str]:
        """
        主入口点：根据模式验证和修复JSON文本。

        参数:
            text: LLM响应的原始文本
            schema_name: 要验证的模式名称

        返回:
            元组 (parsed_data, success, error_message)
            - parsed_data: 成功解析的JSON字典或None
            - success: 如果验证通过则为True（无论是否修复）
            - error_message: 任何问题或修复操作的描述
        """
        schema = self.schemas.get(schema_name)
        if not schema:
            return None, False, f"Unknown schema: {schema_name}"
        
        # 阶段1: 预处理和清理文本
        cleaned_text = self._preprocess_text(text)
        
        # 阶段2: 尝试直接解析和验证
        try:
            data = json.loads(cleaned_text)
            validate(data, schema)
            return data, True, "Success"
        except (json.JSONDecodeError, ValidationError) as e:
            # 阶段3: 应用渐进式修复策略
            return self._progressive_fix(cleaned_text, schema, str(e))
    
    def _preprocess_text(self, text: str) -> str:
        """
        在JSON解析之前清理和预处理原始文本。

        参数:
            text: LLM响应的原始文本

        返回:
            准备进行JSON解析的清理后文本
        """
        # 移除markdown代码块标记
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text.rstrip())
        
        # 清理空白字符和控制字符
        text = text.strip()
        
        # 如果存在则移除BOM字符
        if text.startswith('\ufeff'):
            text = text[1:]
        
        # 移除不可打印字符（除换行符、制表符、回车符外）
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        return text
    
    def _progressive_fix(self, text: str, schema: dict, error_msg: str) -> Tuple[Optional[Dict], bool, str]:
        """
        应用渐进式修复策略来修复JSON问题。

        参数:
            text: 预处理后的文本
            schema: 用于验证的JSONSchema
            error_msg: 原始错误消息

        返回:
            元组 (parsed_data, success, repair_message)
        """
        repair_steps = []
        
        # 级别1: 基本结构修复
        text, steps = self._fix_json_structure(text)
        repair_steps.extend(steps)
        
        # 级别2: 模式驱动修复
        text, steps = self._fix_with_schema(text, schema)
        repair_steps.extend(steps)
        
        # 最终验证尝试
        try:
            data = json.loads(text)
            validate(data, schema)
            repair_message = f"Fixed: {error_msg} | Repairs: {', '.join(repair_steps)}"
            return data, True, repair_message
        except Exception as final_e:
            return None, False, f"Fix failed: {str(final_e)} | Original: {error_msg}"
    
    def _fix_json_structure(self, text: str) -> Tuple[str, List[str]]:
        """
        修复基本的JSON结构问题。

        参数:
            text: 要修复的文本

        返回:
            元组 (repaired_text, list_of_repair_actions)
        """
        repairs = []
        
        # 修复缺失的冒号分隔符
        original_text = text
        text = re.sub(r'"(\w+)"\s+"', r'"\1": "', text)
        if text != original_text:
            repairs.append("fixed_missing_colons")
        
        # 通过智能检测修复utterance字段
        if text.startswith('{"utterance": "'):
            needs_fix = False
            try:
                json.loads(text)  # 测试是否已经有效
            except json.JSONDecodeError:
                needs_fix = True
            
            if needs_fix and not text.endswith('"}'):
                if text.endswith('"'):
                    text = text + '}'
                    repairs.append("closed_utterance_object")
                else:
                    text = text + '"}'
                    repairs.append("closed_utterance_string_and_object")
        
        # 修复"额外数据"问题 - 提取第一个完整的JSON对象
        if text.startswith('{') and '}' in text:
            bracket_count = 0
            json_end = -1
            for i, char in enumerate(text):
                if char == '{':
                    bracket_count += 1
                elif char == '}':
                    bracket_count -= 1
                    if bracket_count == 0:
                        json_end = i + 1
                        break
            
            if json_end > 0 and json_end < len(text):
                original_length = len(text)
                text = text[:json_end]
                if len(text) < original_length:
                    repairs.append("removed_extra_data")
        
        # 修复数组格式问题
        if text.startswith('[') and text.endswith(']'):
            # 检查数组中的未终止字符串
            quote_count = text.count('"')
            if quote_count % 2 != 0:
                # 简单修复：在最后的括号前添加结束引号
                if text.endswith(']') and not text.endswith('"]'):
                    text = text[:-1] + '"]'
                    repairs.append("closed_array_string")
        
        return text, repairs
    
    def _fix_with_schema(self, text: str, schema: dict) -> Tuple[str, List[str]]:
        """
        应用模式驱动修复来修复缺失或错误的字段。

        参数:
            text: 要修复的文本
            schema: 用于指导的JSONSchema

        返回:
            元组 (repaired_text, list_of_repair_actions)
        """
        repairs = []
        
        try:
            data = json.loads(text)
            
            # 基于模式要求进行修复
            if 'required' in schema:
                for field in schema['required']:
                    if field not in data:
                        # 添加缺失的必需字段和适当的默认值
                        field_schema = schema.get('properties', {}).get(field, {})
                        default_value = self._get_default_value(field_schema)
                        data[field] = default_value
                        repairs.append(f"added_missing_field_{field}")
            
            # 如果additionalProperties为False则移除额外字段
            if schema.get('additionalProperties') is False and 'properties' in schema:
                allowed_fields = set(schema['properties'].keys())
                extra_fields = set(data.keys()) - allowed_fields
                for field in extra_fields:
                    del data[field]
                    repairs.append(f"removed_extra_field_{field}")
            
            # 针对常见问题的类型纠正
            if 'properties' in schema:
                for field, field_schema in schema['properties'].items():
                    if field in data:
                        data[field] = self._fix_field_type(data[field], field_schema)
            
            return json.dumps(data), repairs
            
        except json.JSONDecodeError:
            # 如果无法解析为JSON，则按原样返回
            return text, repairs
    
    def _get_default_value(self, field_schema: dict) -> Any:
        """
        根据字段模式获取适当的默认值。

        参数:
            field_schema: 字段的JSONSchema

        返回:
            适当的默认值
        """
        field_type = field_schema.get('type', 'string')
        
        if field_type == 'string':
            return ""
        elif field_type == 'array':
            return []
        elif field_type == 'object':
            return {}
        elif field_type == 'number':
            return 0.0
        elif field_type == 'integer':
            return 0
        elif field_type == 'boolean':
            return False
        else:
            return None
    
    def _fix_field_type(self, value: Any, field_schema: dict) -> Any:
        """
        修复字段类型不匹配问题。

        参数:
            value: 当前字段值
            field_schema: 字段的期望模式

        返回:
            类型纠正后的值
        """
        expected_type = field_schema.get('type')
        
        if expected_type == 'number' and isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return 0.0
        elif expected_type == 'integer' and isinstance(value, str):
            try:
                return int(float(value))
            except ValueError:
                return 0
        elif expected_type == 'string' and isinstance(value, (int, float)):
            return str(value)
        elif expected_type == 'array' and not isinstance(value, list):
            return [value] if value is not None else []
        
        return value
    
    def diagnose_error(self, text: str, schema_name: str) -> Dict[str, Any]:
        """
        提供有关JSON解析问题的详细诊断信息。

        参数:
            text: 解析失败的原始文本
            schema_name: 用于验证的模式名称

        返回:
            包含诊断信息的字典
        """
        schema = self.schemas.get(schema_name, {})
        
        diagnostic = {
            "original_text_length": len(text),
            "schema_name": schema_name,
            "has_schema": bool(schema),
            "preprocessing_issues": [],
            "json_issues": [],
            "schema_violations": []
        }
        
        # 检查预处理问题
        if '```json' in text or '```' in text:
            diagnostic["preprocessing_issues"].append("contains_markdown_markers")
        
        if text.startswith('\ufeff'):
            diagnostic["preprocessing_issues"].append("contains_bom")
        
        # 尝试JSON解析
        try:
            data = json.loads(text)
            diagnostic["json_parseable"] = True
            
            # 检查模式违规
            if schema:
                try:
                    validate(data, schema)
                    diagnostic["schema_valid"] = True
                except ValidationError as e:
                    diagnostic["schema_valid"] = False
                    diagnostic["schema_violations"].append(str(e))
        except json.JSONDecodeError as e:
            diagnostic["json_parseable"] = False
            diagnostic["json_issues"].append(f"{e.msg} at position {e.pos}")
        
        return diagnostic

# 全局验证器实例
validator = JSONValidator()