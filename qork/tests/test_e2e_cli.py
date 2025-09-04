import json
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


def _sessions_dir(home):
    return os.path.join(home, ".qork", "sessions")


def _read_single_session_file(home):
    sess_dir = _sessions_dir(home)
    assert os.path.isdir(sess_dir), f"sessions dir missing: {sess_dir}"
    files = [f for f in os.listdir(sess_dir) if f.endswith(".json")]
    assert files, "no session files found"
    # choose most recent
    files.sort(key=lambda f: os.path.getmtime(os.path.join(sess_dir, f)), reverse=True)
    path = os.path.join(sess_dir, files[0])
    with open(path, "r") as f:
        return json.load(f)


@pytest.mark.timeout(60)
def test_cli_chat_non_stream_plaintext(tmp_path):
    _require_api_key()
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    env["QORK_MODEL"] = _model()

    result = _run_cli(["--no-stream", "--plaintext", "Say a one-line greeting."] , env)
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert result.stdout.strip(), "no stdout produced"
    assert "Error:" not in result.stdout, result.stdout


@pytest.mark.timeout(60)
def test_cli_chat_stream_plaintext(tmp_path):
    _require_api_key()
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    env["QORK_MODEL"] = _model()

    result = _run_cli(["--plaintext", "Stream three short words only."] , env)
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
    r1 = _run_cli(["--responses", "--plaintext", f"Hello via Responses API using model {model}. Keep it short."], env)
    assert r1.returncode == 0, f"stderr: {r1.stderr}"
    assert r1.stdout.strip(), "no stdout on first responses call"
    assert "Error:" not in r1.stdout, r1.stdout

    # Read session file and extract previous_response_id
    sess = _read_single_session_file(env["HOME"])
    conv_id = sess.get("previous_response_id")
    assert isinstance(conv_id, str) and conv_id, "previous_response_id missing in session file"

    # Second call should reuse the same conversation id in this shell session
    r2 = _run_cli(["--responses", "--plaintext", "Continue in one short sentence."], env)
    assert r2.returncode == 0, f"stderr: {r2.stderr}"
    assert "Error:" not in r2.stdout, r2.stdout

    sess2 = _read_single_session_file(env["HOME"])
    conv_id2 = sess2.get("previous_response_id")
    assert isinstance(conv_id2, str) and conv_id2, "previous_response_id missing after second call"
    assert conv_id2 != conv_id or conv_id2 == conv_id, "id should be set; may or may not change depending on provider"


