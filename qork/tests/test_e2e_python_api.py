import os
import pytest

from qork.ask import ask


def _require_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping end-to-end tests")


def _model():
    return os.environ.get("QORK_E2E_MODEL", os.environ.get("QORK_MODEL", "gpt-5-mini"))


@pytest.mark.timeout(60)
def test_python_api_chat_plaintext(tmp_path, monkeypatch):
    _require_api_key()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("QORK_MODEL", _model())

    text = ask("Say one short sentence.", stream=False, plaintext=True, debug=False, return_text=True)
    assert isinstance(text, str) and text.strip(), "ask() returned empty text"


@pytest.mark.timeout(120)
def test_python_api_responses_plaintext(tmp_path, monkeypatch):
    _require_api_key()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("QORK_MODEL", _model())

    text1 = ask("Start a short thread via Responses API.", stream=False, plaintext=True, debug=True, return_text=True)
    assert isinstance(text1, str) and text1.strip(), "first responses call returned empty"

    text2 = ask("Continue the thread in one sentence.", stream=False, plaintext=True, debug=False, return_text=True)
    assert isinstance(text2, str) and text2.strip(), "second responses call returned empty"

