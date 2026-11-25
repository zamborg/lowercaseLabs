"""Standalone ZOS module caller-identity helpers (zero dependency on `zos`)."""

from zos_module_sdk.caller import (
    DEFAULT_MAX_SKEW_SECONDS,
    HTTP_CALLER_HEADER,
    HTTP_CALLER_SIGNATURE_HEADER,
    MCP_CALLER_META_KEY,
    CallerVerificationError,
    VerifiedCaller,
    canonical_json,
    compute_args_digest,
    sign_http_caller,
    sign_mcp_caller,
    verify_http_caller,
    verify_mcp_caller,
)

__all__ = [
    "DEFAULT_MAX_SKEW_SECONDS",
    "HTTP_CALLER_HEADER",
    "HTTP_CALLER_SIGNATURE_HEADER",
    "MCP_CALLER_META_KEY",
    "CallerVerificationError",
    "VerifiedCaller",
    "canonical_json",
    "compute_args_digest",
    "sign_http_caller",
    "sign_mcp_caller",
    "verify_http_caller",
    "verify_mcp_caller",
]
