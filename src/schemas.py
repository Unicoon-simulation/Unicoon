
SCHEMAS = {
    "recommendation": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "news_id": {
                    "type": "string",
                    "description": "Recommended news ID"
                },
                "reason": {
                    "type": "string",
                    "maxLength": 100,
                    "description": "Recommendation reason, concise and clear"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Recommendation confidence score"
                }
            },
            "required": ["news_id", "reason", "confidence"],
            "additionalProperties": False
        },
        "maxItems": 5,
        "minItems": 1
    },
    
    "news_rewrite": {
        "type": "object",
        "properties": {
            "processed_news": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "news_id": {"type": "string"},
                        "action": {
                            "type": "string",
                            "enum": ["rewrite", "skip"]
                        },
                        "transformed_content": {
                            "oneOf": [
                                {"type": "string", "minLength": 10, "maxLength": 300},
                                {"type": "null"}
                            ]
                        },
                        "reason": {"type": "string"}
                    },
                    "required": ["news_id", "action", "transformed_content", "reason"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["processed_news"],
        "additionalProperties": False
    },
    
    "group_discussion": {
        "type": "object",
        "properties": {
            "utterance": {
                "type": "string",
                "maxLength": 400,
                "minLength": 1,
                "description": "Discussion utterance content"
            }
        },
        "required": ["utterance"],
        "additionalProperties": False
    },
    
    "memory_update": {
        "type": "object",
        "properties": {
            "short_term_memory_update": {
                "type": "string",
                "description": "Summary of this round's experiences"
            },
            "long_term_memory_update": {
                "type": "string", 
                "description": "Updated long-term memory integrating new insights"
            },
            "score_updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "trust": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "interest": {
                            "type": "number", 
                            "minimum": 0.0,
                            "maximum": 1.0
                        }
                    },
                    "required": ["agent_id", "trust", "interest"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["short_term_memory_update", "long_term_memory_update", "score_updates"],
        "additionalProperties": False
    },
    
    "mem_updatenew": {
        "type": "object",
        "properties": {
            "short_term_memory_update": {
                "type": "string",
                "description": "Summary of this round's key experiences and learnings"
            },
            "long_term_memory_update": {
                "type": "string",
                "description": "Updated long-term memory integrating new insights and events"
            },
            "score_updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "trust": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "interest": {
                            "type": "number",
                            "minimum": 0.0, 
                            "maximum": 1.0
                        }
                    },
                    "required": ["agent_id", "trust", "interest"],
                    "additionalProperties": False
                }
            },
            "trait_changes": {
                "type": "object",
                "properties": {
                    "traits": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of current personality traits (can be modified)"
                    },
                    "political_affiliation": {
                        "type": "string",
                        "enum": ["far_left", "left", "moderate", "right", "far_right"],
                        "description": "Current political affiliation"
                    },
                    "self_awareness": {
                        "type": "string",
                        "description": "Current self-awareness statement (can be evolved)"
                    }
                },
                "required": ["traits", "political_affiliation", "self_awareness"],
                "additionalProperties": False
            }
        },
        "required": ["short_term_memory_update", "long_term_memory_update", "score_updates", "trait_changes"],
        "additionalProperties": False
    }
}

def get_schema(schema_name: str) -> dict:
    return SCHEMAS.get(schema_name, {})

def list_schemas() -> list:
    return list(SCHEMAS.keys())

def validate_schema_exists(schema_name: str) -> bool:
    return schema_name in SCHEMAS