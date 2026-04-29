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

    for module_name in [
        "app.db",
        "app.auth",
        "app.agent.responses.tools",
        "app.agent.responses.client",
        "app.agent.responses.logging",
        "app.agent.responses.prompts",
        "app.analysis",
        "app.main",
    ]:
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
            [
                {
                    "type": "todo",
                    "title": "Buy milk",
                    "tags": ["errand"],
                    "due_date": "2026-04-14T17:00:00",
                }
            ],
            {
                "operation": "analyze_transcript",
                "model": "gpt-5.4-mini-2026-03-17",
                "input_text": "buy milk tomorrow at 5",
                "system_prompt": "system prompt",
                "user_prompt": "user prompt",
                "raw_response": '{"items":[{"type":"todo","title":"Buy milk","tags":["errand"]}]}',
                "parsed_response": '[{"type":"todo","title":"Buy milk","tags":["errand"]}]',
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
        self.assertEqual(response.json()[0]["title"], "Buy milk")

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

    def test_search_persists_llm_log_and_falls_back_on_error(self):
        now = "2026-04-14T17:00:00"
        self.db.create_item({
            "id": "item-1",
            "user_id": "user-1",
            "content": "Buy milk from the corner store",
            "title": "Buy milk",
            "type": "todo",
            "epic_id": None,
            "due_date": None,
            "completed": 0,
            "tags": '["errand"]',
            "created_at": now,
            "updated_at": now,
        })
        fake_search = (
            None,
            {
                "operation": "search_items",
                "model": "gpt-5.4-mini-2026-03-17",
                "input_text": "errand",
                "system_prompt": "system prompt",
                "user_prompt": "user prompt",
                "raw_response": None,
                "parsed_response": None,
                "status": "error",
                "error": "boom",
            },
        )

        with patch.object(self.main.analysis, "run_search_items", return_value=fake_search):
            response = self.client.post(
                "/search",
                headers={"Authorization": f"Bearer {self.token}"},
                json={"query": "errand"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()[0]["title"], "Buy milk")
        logs = [dict(row) for row in self.db.list_llm_logs()]
        self.assertEqual(logs[0]["operation"], "search_items")
        self.assertEqual(logs[0]["status"], "error")

    def test_create_item_links_note_to_existing_epic(self):
        now = "2026-04-14T17:00:00"
        self.db.create_item({
            "id": "epic-1",
            "user_id": "user-1",
            "content": "Ship blackhole app",
            "title": "Blackhole App",
            "type": "epic",
            "epic_id": None,
            "due_date": None,
            "completed": 0,
            "tags": '["product"]',
            "created_at": now,
            "updated_at": now,
        })
        fake_analysis = (
            [
                {
                    "type": "note",
                    "title": "Backend deploy notes",
                    "tags": ["deploy"],
                    "due_date": None,
                    "epic_title": "Blackhole App",
                }
            ],
            {
                "operation": "analyze_transcript",
                "model": "gpt-5.4-mini-2026-03-17",
                "input_text": "deployment notes for blackhole app",
                "system_prompt": "system prompt",
                "user_prompt": "user prompt",
                "raw_response": '{"items":[{"type":"note","title":"Backend deploy notes","tags":["deploy","Blackhole App"],"due_date":null,"epic_title":"Blackhole App"}]}',
                "parsed_response": '[{"type":"note","title":"Backend deploy notes","tags":["deploy","Blackhole App"],"due_date":null,"epic_title":"Blackhole App"}]',
                "status": "success",
                "error": None,
            },
        )

        with patch.object(self.main.analysis, "run_transcript_analysis", return_value=fake_analysis):
            response = self.client.post(
                "/items",
                headers={"Authorization": f"Bearer {self.token}"},
                json={"content": "deployment notes for blackhole app"},
            )

        self.assertEqual(response.status_code, 200)
        created = response.json()[0]
        self.assertEqual(created["epic_id"], "epic-1")
        self.assertIn("Blackhole App", created["tags"])

    def test_create_item_links_to_epic_created_in_same_response(self):
        fake_analysis = (
            [
                {
                    "type": "todo",
                    "title": "Draft launch checklist",
                    "tags": ["launch"],
                    "due_date": None,
                    "epic_title": "Launch Plan",
                },
                {
                    "type": "epic",
                    "title": "Launch Plan",
                    "tags": ["launch"],
                    "due_date": None,
                    "epic_title": None,
                },
            ],
            {
                "operation": "analyze_transcript",
                "model": "gpt-5.4-mini-2026-03-17",
                "input_text": "create launch plan epic and draft checklist",
                "system_prompt": "system prompt",
                "user_prompt": "user prompt",
                "raw_response": "{}",
                "parsed_response": "[]",
                "status": "success",
                "error": None,
            },
        )

        with patch.object(self.main.analysis, "run_transcript_analysis", return_value=fake_analysis):
            response = self.client.post(
                "/items",
                headers={"Authorization": f"Bearer {self.token}"},
                json={"content": "create launch plan epic and draft checklist"},
            )

        self.assertEqual(response.status_code, 200)
        created = response.json()
        epic = next(item for item in created if item["type"] == "epic")
        todo = next(item for item in created if item["type"] == "todo")
        self.assertEqual(todo["epic_id"], epic["id"])
        self.assertIsNone(epic["epic_id"])

    def test_update_item_can_assign_epic(self):
        now = "2026-04-14T17:00:00"
        self.db.create_item({
            "id": "epic-1",
            "user_id": "user-1",
            "content": "Ship blackhole app",
            "title": "Blackhole App",
            "type": "epic",
            "epic_id": None,
            "due_date": None,
            "completed": 0,
            "tags": '["product"]',
            "created_at": now,
            "updated_at": now,
        })
        self.db.create_item({
            "id": "note-1",
            "user_id": "user-1",
            "content": "Backend notes",
            "title": "Backend notes",
            "type": "note",
            "epic_id": None,
            "due_date": None,
            "completed": 0,
            "tags": "[]",
            "created_at": now,
            "updated_at": now,
        })

        response = self.client.patch(
            "/items/note-1",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"epic_id": "epic-1"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["epic_id"], "epic-1")
        self.assertIn("Blackhole App", response.json()["tags"])


class AnalysisClientContractTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tempdir.name) / "analysis-test.db")
        _, _, self.analysis, _ = load_backend_modules(self.db_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_run_transcript_analysis_uses_responses_contract(self):
        recorded = {}

        class FakeResponses:
            def create(self, **kwargs):
                recorded.update(kwargs)
                return SimpleNamespace(output_text='{"items":[{"type":"note","title":"Test","tags":[],"due_date":null,"epic_title":null}]}')

        fake_client = SimpleNamespace(responses=FakeResponses())

        with patch.object(self.analysis, "get_client", return_value=fake_client):
            result, log = self.analysis.run_transcript_analysis("capture this")

        self.assertEqual(log["status"], "success")
        self.assertEqual(result[0]["title"], "Test")
        self.assertEqual(recorded["max_output_tokens"], 650)
        self.assertEqual(recorded["text"]["format"]["type"], "json_schema")
        self.assertEqual(recorded["text"]["format"]["name"], "transcript_items")
        self.assertIn("epic", recorded["text"]["format"]["schema"]["properties"]["items"]["items"]["properties"]["type"]["enum"])
        self.assertIn("epic_title", recorded["text"]["format"]["schema"]["properties"]["items"]["items"]["properties"])
        self.assertIn("Do not create a duplicate raw note", recorded["instructions"])
        self.assertIn("Epics are stronger categories", recorded["instructions"])
        self.assertIn("input", recorded)
        self.assertIn("instructions", recorded)
        self.assertNotIn("max_tokens", recorded)
        self.assertNotIn("max_completion_tokens", recorded)

    def test_search_items_uses_max_output_tokens(self):
        recorded = {}

        class FakeResponses:
            def create(self, **kwargs):
                recorded.update(kwargs)
                return SimpleNamespace(output_text='{"indices":[0]}')

        fake_client = SimpleNamespace(responses=FakeResponses())
        items = [
            {
                "title": "Groceries",
                "content": "Buy milk",
                "type": "todo",
                "completed": 0,
                "due_date": "2026-04-15T17:00:00",
                "tags": '["errand"]',
            }
        ]

        with patch.object(self.analysis, "get_client", return_value=fake_client):
            results = self.analysis.search_items("milk", items)

        self.assertEqual(results, items)
        self.assertEqual(recorded["max_output_tokens"], 200)
        self.assertEqual(recorded["text"]["format"]["name"], "search_indices")
        self.assertIn("todo | Groceries | open due:2026-04-15T17:00:00 tags:errand", recorded["input"][0]["content"])
        self.assertNotIn("max_tokens", recorded)
        self.assertNotIn("max_completion_tokens", recorded)

    def test_registered_tools_are_not_exposed_without_prompt_opt_in(self):
        recorded = {}
        tools = sys.modules["app.agent.responses.tools"]
        tools.register_tool(tools.ResponseTool(
            name="lookup",
            definition={
                "type": "function",
                "name": "lookup",
                "description": "Test lookup tool.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        ))

        class FakeResponses:
            def create(self, **kwargs):
                recorded.update(kwargs)
                return SimpleNamespace(output_text='{"indices":[0]}')

        fake_client = SimpleNamespace(responses=FakeResponses())
        items = [{"title": "Groceries", "content": "Buy milk"}]

        with patch.object(self.analysis, "get_client", return_value=fake_client):
            self.analysis.search_items("milk", items)

        self.assertNotIn("tools", recorded)

    def test_prior_items_context_injected_into_messages(self):
        """Context messages are inserted when prior_items are provided."""
        recorded = {}

        class FakeResponses:
            def create(self, **kwargs):
                recorded.update(kwargs)
                return SimpleNamespace(output_text='{"items":[{"type":"todo","title":"Buy oat milk","tags":["errand"],"due_date":null,"epic_title":null}]}')

        fake_client = SimpleNamespace(responses=FakeResponses())
        prior = [
            {"type": "todo", "title": "Get groceries", "content": "buy eggs and bread", "completed": 0, "tags": '["errand"]', "due_date": None},
            {"type": "epic", "title": "Home Ops", "content": "household errands", "completed": 0, "tags": "[]", "due_date": None},
        ]

        with patch.object(self.analysis, "get_client", return_value=fake_client):
            results, log = self.analysis.run_transcript_analysis("also pick up oat milk", prior_items=prior)

        self.assertEqual(log["status"], "success")
        self.assertEqual(results[0]["title"], "Buy oat milk")
        messages = recorded["input"]
        # context-user + context-ack + classify-user = 3; system prompt is instructions
        self.assertEqual(len(messages), 3)
        self.assertIn("existing notes, todos, and epics", recorded["instructions"])
        self.assertIn("epic_title", messages[2]["content"])
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("Get groceries", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[2]["role"], "user")
        self.assertIn("also pick up oat milk", messages[2]["content"])

    def test_no_prior_items_sends_two_messages(self):
        """Without prior context only system + classify-user are sent."""
        recorded = {}

        class FakeResponses:
            def create(self, **kwargs):
                recorded.update(kwargs)
                return SimpleNamespace(output_text='{"items":[{"type":"note","title":"Test","tags":[],"due_date":null,"epic_title":null}]}')

        fake_client = SimpleNamespace(responses=FakeResponses())

        with patch.object(self.analysis, "get_client", return_value=fake_client):
            self.analysis.run_transcript_analysis("just a note", prior_items=None)

        messages = recorded["input"]
        self.assertEqual(len(messages), 1)
        self.assertIn("classify text submitted to blackhole", recorded["instructions"])
        self.assertEqual(messages[0]["role"], "user")

    @unittest.skipUnless(
        os.getenv("BLACKHOLE_RUN_OPENAI_TESTS") == "1" and os.getenv("OPENAI_API_KEY"),
        "Set BLACKHOLE_RUN_OPENAI_TESTS=1 and OPENAI_API_KEY to run the live OpenAI smoke test.",
    )
    def test_live_openai_transcript_analysis_smoke(self):
        result, log = self.analysis.run_transcript_analysis(
            "Remind me tomorrow at 5 PM to buy batteries and oat milk"
        )

        self.assertEqual(log["status"], "success")
        self.assertGreaterEqual(len(result), 1)
        self.assertIn(result[0]["type"], {"note", "todo", "epic"})
        self.assertIsInstance(result[0]["title"], str)
        self.assertIsInstance(result[0]["tags"], list)
        self.assertIn("due_date", result[0])
        self.assertIn("epic_title", result[0])


if __name__ == "__main__":
    unittest.main()
