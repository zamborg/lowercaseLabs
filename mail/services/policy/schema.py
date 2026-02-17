POLICY_SCHEMA = {
    "type": "object",
    "required": ["name", "autonomy_level", "allowed_actions"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "autonomy_level": {"type": "integer", "minimum": 0, "maximum": 3},
        "allowed_actions": {
            "type": "array",
            "items": {"type": "string"},
        },
        "reply_template": {"type": "string"},
        "style_profile": {"type": "object"},
        "escalation_rules": {"type": "object"},
    },
    "additionalProperties": False,
}
