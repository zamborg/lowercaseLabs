from jsonschema import Draft202012Validator

from services.policy.schema import POLICY_SCHEMA


def validate_policy_payload(payload: dict) -> list[str]:
    validator = Draft202012Validator(POLICY_SCHEMA)
    errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
    return [error.message for error in errors]
