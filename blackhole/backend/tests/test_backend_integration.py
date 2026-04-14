import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient


def load_backend_modules(db_path: str):
    os.environ["ADMIN_USERNAME"] = "zubin"
    os.environ["ADMIN_PASSWORD"] = "Gogetass4$"
    os.environ["DB_PATH"] = db_path
    os.environ.setdefault("JWT_SECRET", "test-secret-with-32-bytes-length")

    for module_name in ["app.db", "app.auth", "app.analysis", "app.main"]:
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        else:
            importlib.import_module(module_name)

    return (
        sys.modules["app.db"],
        sys.modules["app.auth"],
        sys.modules["app.analysis"],
        sys.modules["app.main"],
    )


class BackendIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tempdir.name) / "blackhole-test.db")
        self.db, self.auth, self.analysis, self.main = load_backend_modules(self.db_path)
        self.db.init_db()
        self.db.upsert_user("user-1")
        self.client = TestClient(self.main.app, base_url="https://testserver")
        self.token = self.auth.create_session_token("user-1")

    def tearDown(self):
        self.tempdir.cleanup()

    def test_create_item_persists_llm_log_and_exposes_it_in_admin(self):
        fake_analysis = (
            {
                "type": "todo",
                "title": "Buy milk",
                "tags": ["errand"],
                "due_date": "2026-04-14T17:00:00",
            },
            {
                "operation": "analyze_transcript",
                "model": "gpt-5.4-mini-2026-03-17",
                "input_text": "buy milk tomorrow at 5",
                "system_prompt": "system prompt",
                "user_prompt": "user prompt",
                "raw_response": '{"type":"todo","title":"Buy milk","tags":["errand"]}',
                "parsed_response": '{"type":"todo","title":"Buy milk","tags":["errand"]}',
                "status": "success",
                "error": None,
            },
        )

        with patch.object(self.main.analysis, "run_transcript_analysis", return_value=fake_analysis):
            response = self.client.post(
                "/items",
                headers={"Authorization": f"Bearer {self.token}"},
                json={"content": "buy milk tomorrow at 5"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["title"], "Buy milk")

        login = self.client.post(
            "/admin/login",
            data={"username": "zubin", "password": "Gogetass4$"},
            follow_redirects=False,
        )
        self.assertEqual(login.status_code, 303)

        overview = self.client.get("/admin/api/overview")
        self.assertEqual(overview.status_code, 200)
        body = overview.json()

        self.assertEqual(body["summary"]["llm_total_logs"], 1)
        self.assertEqual(body["summary"]["llm_failed_logs"], 0)
        self.assertEqual(body["llm_logs"][0]["operation"], "analyze_transcript")
        self.assertEqual(body["llm_logs"][0]["input_text"], "buy milk tomorrow at 5")

        dashboard = self.client.get("/admin")
        self.assertIn("LLM Analysis Logs", dashboard.text)
        self.assertIn("buy milk tomorrow at 5", dashboard.text)


class AnalysisClientContractTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tempdir.name) / "analysis-test.db")
        _, _, self.analysis, _ = load_backend_modules(self.db_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_run_transcript_analysis_uses_max_completion_tokens(self):
        recorded = {}

        class FakeCompletions:
            def create(self, **kwargs):
                recorded.update(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{"type":"note","title":"Test","tags":[],"due_date":null}'))]
                )

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

        with patch.object(self.analysis, "get_client", return_value=fake_client):
            result, log = self.analysis.run_transcript_analysis("capture this")

        self.assertEqual(log["status"], "success")
        self.assertEqual(result["title"], "Test")
        self.assertEqual(recorded["max_completion_tokens"], 300)
        self.assertNotIn("max_tokens", recorded)

    def test_search_items_uses_max_completion_tokens(self):
        recorded = {}

        class FakeCompletions:
            def create(self, **kwargs):
                recorded.update(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{"indices":[0]}'))]
                )

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        items = [{"title": "Groceries", "content": "Buy milk"}]

        with patch.object(self.analysis, "get_client", return_value=fake_client):
            results = self.analysis.search_items("milk", items)

        self.assertEqual(results, items)
        self.assertEqual(recorded["max_completion_tokens"], 200)
        self.assertNotIn("max_tokens", recorded)

    def test_prior_items_context_injected_into_messages(self):
        """Context messages are inserted when prior_items are provided."""
        recorded = {}

        class FakeCompletions:
            def create(self, **kwargs):
                recorded.update(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(
                        content='{"type":"todo","title":"Buy oat milk","tags":["errand"],"due_date":null}'
                    ))]
                )

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
        prior = [
            {"type": "todo", "title": "Get groceries", "content": "buy eggs and bread", "completed": 0, "tags": '["errand"]', "due_date": None},
            {"type": "note", "title": "Recipe idea", "content": "pancakes", "completed": 0, "tags": "[]", "due_date": None},
        ]

        with patch.object(self.analysis, "get_client", return_value=fake_client):
            result, log = self.analysis.run_transcript_analysis("also pick up oat milk", prior_items=prior)

        self.assertEqual(log["status"], "success")
        messages = recorded["messages"]
        # system + context-user + context-ack + classify-user = 4
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Get groceries", messages[1]["content"])
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[3]["role"], "user")
        self.assertIn("also pick up oat milk", messages[3]["content"])

    def test_no_prior_items_sends_two_messages(self):
        """Without prior context only system + classify-user are sent."""
        recorded = {}

        class FakeCompletions:
            def create(self, **kwargs):
                recorded.update(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(
                        content='{"type":"note","title":"Test","tags":[],"due_date":null}'
                    ))]
                )

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

        with patch.object(self.analysis, "get_client", return_value=fake_client):
            self.analysis.run_transcript_analysis("just a note", prior_items=None)

        messages = recorded["messages"]
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

    @unittest.skipUnless(
        os.getenv("BLACKHOLE_RUN_OPENAI_TESTS") == "1" and os.getenv("OPENAI_API_KEY"),
        "Set BLACKHOLE_RUN_OPENAI_TESTS=1 and OPENAI_API_KEY to run the live OpenAI smoke test.",
    )
    def test_live_openai_transcript_analysis_smoke(self):
        result, log = self.analysis.run_transcript_analysis(
            "Remind me tomorrow at 5 PM to buy batteries and oat milk"
        )

        self.assertEqual(log["status"], "success")
        self.assertIn(result["type"], {"note", "todo"})
        self.assertIsInstance(result["title"], str)
        self.assertIsInstance(result["tags"], list)
        self.assertIn("due_date", result)


if __name__ == "__main__":
    unittest.main()
