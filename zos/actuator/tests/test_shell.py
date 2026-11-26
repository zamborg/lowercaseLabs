"""Real tmux integration tests for TmuxShellRunner (`mac_shell_run`)."""

from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path

import pytest

from zos_actuator.shell import TmuxShellRunner, cap_output_bytes

pytestmark = pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux binary not installed")


async def _kill_tmux_server(socket_name: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        "tmux",
        "-L",
        socket_name,
        "kill-server",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.communicate()


def test_cap_output_bytes_head_and_tail() -> None:
    small = b"hello world"
    out, trunc = cap_output_bytes(small, max_bytes=100)
    assert out == "hello world" and trunc is False

    big = (b"A" * 20000) + (b"B" * 20000)
    out2, trunc2 = cap_output_bytes(big, max_bytes=32 * 1024)
    assert trunc2 is True
    assert out2.startswith("A" * 100)
    assert out2.endswith("B" * 100)
    assert "truncated" in out2


@pytest.mark.skipif(not Path("/bin/zsh").exists(), reason="actuator shell is macOS /bin/zsh")
async def test_tmux_shell_persistence_hazards_busy_reset_exit_and_cap(tmp_path: Path) -> None:
    sock = f"zos-test-{uuid.uuid4().hex[:8]}"
    runner = TmuxShellRunner(socket_name=sock, session_name="zos", temp_root=tmp_path / "tmp")
    try:
        # 1. Basic command + quoting + multiline + `!` hazard + persistent env/cwd
        sub = tmp_path / "workdir"
        sub.mkdir()
        r1 = await runner.run(
            f"cd {sub}\nexport ZOS_TEST_VAR='hello! \"world\"'\necho \"$ZOS_TEST_VAR\"\n"
        )
        assert r1["ok"] is True
        assert r1["exit_code"] == 0
        assert 'hello! "world"' in r1["output"]

        # Verify cwd and exported env persisted into next call
        r2 = await runner.run("pwd; echo $ZOS_TEST_VAR")
        assert r2["ok"] is True
        assert r2["exit_code"] == 0
        assert str(sub.resolve()) in r2["output"] or sub.name in r2["output"]
        assert 'hello! "world"' in r2["output"]

        # 2. Non-zero exit code reported cleanly
        r_fail = await runner.run("echo boom; false")
        assert r_fail["ok"] is True
        assert r_fail["exit_code"] == 1
        assert "boom" in r_fail["output"]

        # 3. Output > 32KB is capped with head + tail
        r_big = await runner.run("python3 -c \"print('H' * 20000 + 'T' * 20000)\"")
        assert r_big["ok"] is True
        assert r_big["truncated"] is True
        assert "truncated" in r_big["output"]

        # 4. Timeout leaves pane busy -> next call returns PANE_BUSY, then reset=True recovers
        r_to = await runner.run("sleep 10", timeout_s=1)
        assert r_to["ok"] is False
        assert r_to["timed_out"] is True

        r_busy = await runner.run("echo should-not-run")
        assert r_busy["ok"] is False
        assert r_busy["error_code"] == "PANE_BUSY"

        r_reset = await runner.run("echo recovered", reset=True)
        assert r_reset["ok"] is True
        assert r_reset["exit_code"] == 0
        assert "recovered" in r_reset["output"]

        # 5. Command running `exit` kills pane -> runner recreates it and reports pane_recreated=True
        r_exit = await runner.run("echo bye; exit")
        assert r_exit["ok"] is True
        assert r_exit["pane_recreated"] is True

        # Subsequent command works cleanly in recreated pane
        r_after = await runner.run("echo alive-again")
        assert r_after["ok"] is True
        assert r_after["exit_code"] == 0
        assert "alive-again" in r_after["output"]
    finally:
        await _kill_tmux_server(sock)
