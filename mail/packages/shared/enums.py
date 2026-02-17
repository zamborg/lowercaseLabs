from enum import StrEnum


class TriageAction(StrEnum):
    ARCHIVE = "archive"
    DELETE = "delete"
    SNOOZE = "snooze"
    REMEMBER = "remember"
    DELEGATE_REPLY = "delegate_reply"
    BREAK_GLASS_REPLY = "break_glass_reply"
    MARK_READ = "mark_read"
    LABEL = "label"


class AgentJobType(StrEnum):
    SUMMARIZE = "summarize"
    CLASSIFY = "classify"
    DRAFT_REPLY = "draft_reply"
    NOTES_UPDATE = "notes_update"
    SNOOZE_SUGGEST = "snooze_suggest"


class AgentJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
