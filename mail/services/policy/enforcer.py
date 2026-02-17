from typing import Dict, Tuple


def policy_allows_send(policy: Dict, subject: str, draft_text: str) -> Tuple[bool, str]:
    if not policy:
        return True, "no_policy"

    allowed = {action.lower() for action in policy.get("allowed_actions", [])}
    if "send" not in allowed and "delegate_reply" not in allowed:
        return False, "send_not_allowed"

    style_profile = policy.get("style_profile", {})
    signature = style_profile.get("signature")
    if signature and signature not in draft_text:
        return False, "signature_missing"

    escalation = policy.get("escalation_rules", {})
    subject_triggers = escalation.get("subject_contains", [])
    for term in subject_triggers:
        if term.lower() in subject.lower():
            return False, "escalation_subject_block"

    return True, "ok"
