"""Built-in optional policy guard engines (not enabled by default)."""

from __future__ import annotations

import re
from fnmatch import fnmatchcase

from zos.policy.engine import FunctionPolicyEngine, PolicyInput, Verdict

# Match standalone `rm` command tokens while avoiding substrings like `farm`, `my_rm`, or `--rm`.
# Matches: `rm -rf`, `sudo rm`, `x;rm`, `\rm`, `/bin/rm`.
_RM_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_-])rm(?![A-Za-z0-9_-])")


def _deny_rm_in_shell(policy_input: PolicyInput) -> Verdict:
    """Return `deny` when calling `*_shell_run` with an `rm` token in `args.command`."""
    if not fnmatchcase(policy_input.tool.exposed_name, "*_shell_run"):
        return "pass"
    command = policy_input.args.get("command")
    if not isinstance(command, str):
        return "pass"
    if _RM_TOKEN_RE.search(command):
        return "deny"
    return "pass"


deny_rm_in_shell = FunctionPolicyEngine(name="deny_rm_in_shell", fn=_deny_rm_in_shell)
