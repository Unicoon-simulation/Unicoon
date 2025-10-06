
import json
import re
import logging
from typing import Dict, Any, Tuple, Optional, List
from jsonschema import validate, ValidationError
from schemas import SCHEMAS

logger = logging.getLogger("json_validator")

class JSONValidator:
    
    def __init__(self):
        self.schemas = SCHEMAS
    
    def validate_and_fix(self, text: str, schema_name: str) -> Tuple[Optional[Dict], bool, str]:
        schema = self.schemas.get(schema_name)
        if not schema:
            return None, False, f"Unknown schema: {schema_name}"
        
        cleaned_text = self._preprocess_text(text)
        
        try:
            data = json.loads(cleaned_text)
            validate(data, schema)
            return data, True, "Success"
        except (json.JSONDecodeError, ValidationError) as e:
            return self._progressive_fix(cleaned_text, schema, str(e))
    
    def _preprocess_text(self, text: str) -> str:
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*$', '', text.rstrip())
        
        text = text.strip()
        
        if text.startswith('\ufeff'):
            text = text[1:]
        
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        return text
    
    def _progressive_fix(self, text: str, schema: dict, error_msg: str) -> Tuple[Optional[Dict], bool, str]:
        repair_steps = []
        
        text, steps = self._fix_json_structure(text)
        repair_steps.extend(steps)
        
        text, steps = self._fix_with_schema(text, schema)
        repair_steps.extend(steps)
        
        try:
            data = json.loads(text)
            validate(data, schema)
            repair_message = f"Fixed: {error_msg} | Repairs: {', '.join(repair_steps)}"
            return data, True, repair_message
        except Exception as final_e:
            return None, False, f"Fix failed: {str(final_e)} | Original: {error_msg}"
    
    def _fix_json_structure(self, text: str) -> Tuple[str, List[str]]:
        repairs = []
        
        original_text = text
        text = re.sub(r'"(\w+)"\s+"', r'"\1": "', text)
        if text != original_text:
            repairs.append("fixed_missing_colons")
        
        if text.startswith('{"utterance": "'):
            needs_fix = False
            try:
                json.loads(text)
            except json.JSONDecodeError:
                needs_fix = True
            
            if needs_fix and not text.endswith('"}'):
                if text.endswith('"'):
                    text = text + '}'
                    repairs.append("closed_utterance_object")
                else:
                    text = text + '"}'
                    repairs.append("closed_utterance_string_and_object")
        
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
        
        if text.startswith('[') and text.endswith(']'):
            quote_count = text.count('"')
            if quote_count % 2 != 0:
                if text.endswith(']') and not text.endswith('"]'):
                    text = text[:-1] + '"]'
                    repairs.append("closed_array_string")
        
        return text, repairs
    
    def _fix_with_schema(self, text: str, schema: dict) -> Tuple[str, List[str]]:
        repairs = []
        
        try:
            data = json.loads(text)
            
            if 'required' in schema:
                for field in schema['required']:
                    if field not in data:
                        field_schema = schema.get('properties', {}).get(field, {})
                        default_value = self._get_default_value(field_schema)
                        data[field] = default_value
                        repairs.append(f"added_missing_field_{field}")
            
            if schema.get('additionalProperties') is False and 'properties' in schema:
                allowed_fields = set(schema['properties'].keys())
                extra_fields = set(data.keys()) - allowed_fields
                for field in extra_fields:
                    del data[field]
                    repairs.append(f"removed_extra_field_{field}")
            
            if 'properties' in schema:
                for field, field_schema in schema['properties'].items():
                    if field in data:
                        data[field] = self._fix_field_type(data[field], field_schema)
            
            return json.dumps(data), repairs
            
        except json.JSONDecodeError:
            return text, repairs
    
    def _get_default_value(self, field_schema: dict) -> Any:
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
        schema = self.schemas.get(schema_name, {})
        
        diagnostic = {
            "original_text_length": len(text),
            "schema_name": schema_name,
            "has_schema": bool(schema),
            "preprocessing_issues": [],
            "json_issues": [],
            "schema_violations": []
        }
        
        if '```json' in text or '```' in text:
            diagnostic["preprocessing_issues"].append("contains_markdown_markers")
        
        if text.startswith('\ufeff'):
            diagnostic["preprocessing_issues"].append("contains_bom")
        
        try:
            data = json.loads(text)
            diagnostic["json_parseable"] = True
            
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

validator = JSONValidator()