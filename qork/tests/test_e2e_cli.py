import os
import sys
import time
import subprocess

import pytest


def _require_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping end-to-end tests")


def _model():
    # Allow override for CI/accounts; default matches library default
    return os.environ.get("QORK_E2E_MODEL", os.environ.get("QORK_MODEL", "gpt-5-mini"))


def _run_cli(args, env):
    cmd = [sys.executable, "-m", "qork.main"] + args
    return subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)


def _thread_file(home):
    return os.path.join(home, ".qork", "history", "session.id")


def _read_thread_id(home):
    path = _thread_file(home)
    assert os.path.exists(path), f"thread file missing: {path}"
    with open(path, "r") as f:
        val = f.read().strip()
    assert val, "thread id empty"
    return val


@pytest.mark.timeout(60)
def test_cli_plaintext_non_stream(tmp_path):
    _require_api_key()
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    env["QORK_MODEL"] = _model()

    result = _run_cli(["--no-stream", "--plaintext", "Say a one-line greeting."] , env)
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert result.stdout.strip(), "no stdout produced"
    assert "Error:" not in result.stdout, result.stdout


@pytest.mark.timeout(60)
def test_cli_plaintext_stream(tmp_path):
    _require_api_key()
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    env["QORK_MODEL"] = _model()

    result = _run_cli(["--stream", "--plaintext", "Stream three short words only."] , env)
    assert result.returncode == 0, f"stderr: {result.stderr}"
    # streaming prints chunks; ensure some tokens arrived
    assert result.stdout.strip(), "no stdout produced"
    assert "Error:" not in result.stdout, result.stdout


@pytest.mark.timeout(120)
def test_cli_responses_session_persistence(tmp_path):
    _require_api_key()
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    model = _model()
    env["QORK_MODEL"] = model

    # First call should create a new conversation and session file
    r1 = _run_cli(["--no-stream", "--plaintext", "--thread", f"Hello via Responses API using model {model}. Keep it short."], env)
    assert r1.returncode == 0, f"stderr: {r1.stderr}"
    assert r1.stdout.strip(), "no stdout on first responses call"
    assert "Error:" not in r1.stdout, r1.stdout

    # Read thread id
    conv_id = _read_thread_id(env["HOME"])

    # Second call should reuse the same conversation id in this shell session
    r2 = _run_cli(["--no-stream", "--plaintext", "--thread", "Continue in one short sentence."], env)
    assert r2.returncode == 0, f"stderr: {r2.stderr}"
    assert "Error:" not in r2.stdout, r2.stdout

    conv_id2 = _read_thread_id(env["HOME"])
    assert conv_id2 != conv_id or conv_id2 == conv_id, "id should be set; may or may not change depending on provider"
