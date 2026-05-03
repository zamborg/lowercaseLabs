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
        "app.schemas",
        "app.services",
        "app.agent.responses.tools",
        "app.agent.responses.client",
        "app.agent.responses.logging",
        "app.agent.responses.prompts",
        "app.agent.loop",
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
        self.assertEqual(response.json()[0]["status"], "open")

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
        self.assertEqual(response.json()["items"][0]["title"], "Buy milk")
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

    def test_deterministic_item_create_and_status_compatibility(self):
        response = self.client.post(
            "/items",
            headers={"Authorization": f"Bearer {self.token}"},
            json={
                "type": "todo",
                "title": "Buy milk",
                "content": "Buy milk after work",
                "status": "done",
                "source": "manual",
                "tags": ["errand"],
            },
        )

        self.assertEqual(response.status_code, 200)
        item = response.json()
        self.assertEqual(item["status"], "done")
        self.assertTrue(item["completed"])
        self.assertEqual(item["source"], "manual")

        list_response = self.client.get(
            "/items?type=todo&status=done&tags=errand",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(list_response.json()["items"][0]["id"], item["id"])

        patch_response = self.client.patch(
            f"/items/{item['id']}",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"completed": False},
        )
        self.assertEqual(patch_response.status_code, 200)
        self.assertEqual(patch_response.json()["status"], "open")
        self.assertFalse(patch_response.json()["completed"])

    def test_parent_cycle_is_rejected(self):
        first = self.client.post(
            "/items",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"type": "note", "title": "First", "content": "First"},
        ).json()
        second = self.client.post(
            "/items",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"type": "note", "title": "Second", "content": "Second", "parent_id": first["id"]},
        ).json()

        response = self.client.patch(
            f"/items/{first['id']}",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"parent_id": second["id"]},
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "parent_cycle")

    def test_tables_validate_rows_and_protect_backing_item(self):
        table_response = self.client.post(
            "/tables",
            headers={"Authorization": f"Bearer {self.token}"},
            json={
                "title": "Expenses",
                "description": "Personal spending",
                "columns": [
                    {"name": "amount", "type": "number"},
                    {"name": "category", "type": "select", "options": ["food", "travel"]},
                ],
                "tags": ["finance"],
            },
        )
        self.assertEqual(table_response.status_code, 200)
        table = table_response.json()

        row_response = self.client.post(
            f"/tables/{table['id']}/rows",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"data": {"amount": 12.5, "category": "food"}},
        )
        self.assertEqual(row_response.status_code, 200)
        self.assertEqual(row_response.json()["data"]["category"], "food")

        invalid_row = self.client.post(
            f"/tables/{table['id']}/rows",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"data": {"amount": "twelve"}},
        )
        self.assertEqual(invalid_row.status_code, 400)
        self.assertEqual(invalid_row.json()["error"]["code"], "invalid_table_value")

        backing_delete = self.client.delete(
            f"/items/{table['item_id']}",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        self.assertEqual(backing_delete.status_code, 400)
        self.assertEqual(backing_delete.json()["error"]["code"], "table_item_delete_forbidden")

    def test_links_habit_completions_and_api_keys(self):
        habit = self.client.post(
            "/items",
            headers={"Authorization": f"Bearer {self.token}"},
            json={
                "type": "habit",
                "title": "Drink water",
                "content": "Drink water",
                "recurrence_rule": {"freq": "daily"},
            },
        ).json()
        note = self.client.post(
            "/items",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"type": "note", "title": "Hydration note", "content": "Hydration note"},
        ).json()

        link = self.client.post(
            "/links",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"source_id": note["id"], "target_id": habit["id"], "link_type": "relates_to"},
        )
        self.assertEqual(link.status_code, 200)

        duplicate_link = self.client.post(
            "/links",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"source_id": note["id"], "target_id": habit["id"], "link_type": "relates_to"},
        )
        self.assertEqual(duplicate_link.status_code, 409)
        self.assertEqual(duplicate_link.json()["error"]["code"], "duplicate_link")

        completion = self.client.post(
            f"/items/{habit['id']}/completions",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"completed_on": "2026-04-29", "note": "done"},
        )
        self.assertEqual(completion.status_code, 200)

        duplicate_completion = self.client.post(
            f"/items/{habit['id']}/completions",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"completed_on": "2026-04-29"},
        )
        self.assertEqual(duplicate_completion.status_code, 409)
        self.assertEqual(duplicate_completion.json()["error"]["code"], "duplicate_habit_completion")

        token_response = self.client.post(
            "/auth/tokens",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"label": "local mcp"},
        )
        self.assertEqual(token_response.status_code, 200)
        api_key = token_response.json()["key"]

        me_response = self.client.get("/me", headers={"Authorization": f"Bearer {api_key}"})
        self.assertEqual(me_response.status_code, 200)
        self.assertEqual(me_response.json()["id"], "user-1")

    def test_unauthenticated_requests_are_rejected(self):
        for method, path, kwargs in [
            ("get", "/items", {}),
            ("get", "/items/nonexistent", {}),
            ("post", "/items", {"json": {"type": "note", "title": "x", "content": "x"}}),
            ("patch", "/items/nonexistent", {"json": {"title": "y"}}),
            ("delete", "/items/nonexistent", {}),
            ("post", "/agent", {"json": {"message": "hello"}}),
            ("get", "/me", {}),
        ]:
            resp = getattr(self.client, method)(path, **kwargs)
            self.assertEqual(resp.status_code, 401, f"{method.upper()} {path} should require auth")

    def test_cross_user_isolation(self):
        item = self.client.post(
            "/items",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"type": "note", "title": "Private note", "content": "secret"},
        ).json()

        self.db.upsert_user("user-2")
        token2 = self.auth.create_session_token("user-2")

        get_resp = self.client.get(
            f"/items/{item['id']}",
            headers={"Authorization": f"Bearer {token2}"},
        )
        self.assertEqual(get_resp.status_code, 404)

        list_resp = self.client.get(
            "/items",
            headers={"Authorization": f"Bearer {token2}"},
        )
        self.assertEqual(list_resp.status_code, 200)
        self.assertEqual(list_resp.json()["items"], [])

    def test_list_and_get_items(self):
        t1 = self.client.post(
            "/items",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"type": "todo", "title": "Todo one", "content": "Do A", "tags": ["work"]},
        ).json()
        t2 = self.client.post(
            "/items",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"type": "note", "title": "Note one", "content": "Note content"},
        ).json()

        all_resp = self.client.get("/items", headers={"Authorization": f"Bearer {self.token}"})
        self.assertEqual(all_resp.status_code, 200)
        all_ids = [i["id"] for i in all_resp.json()["items"]]
        self.assertIn(t1["id"], all_ids)
        self.assertIn(t2["id"], all_ids)

        todo_resp = self.client.get("/items?type=todo", headers={"Authorization": f"Bearer {self.token}"})
        self.assertEqual(todo_resp.status_code, 200)
        todo_ids = [i["id"] for i in todo_resp.json()["items"]]
        self.assertIn(t1["id"], todo_ids)
        self.assertNotIn(t2["id"], todo_ids)

        get_resp = self.client.get(f"/items/{t1['id']}", headers={"Authorization": f"Bearer {self.token}"})
        self.assertEqual(get_resp.status_code, 200)
        self.assertEqual(get_resp.json()["title"], "Todo one")

        missing = self.client.get("/items/does-not-exist", headers={"Authorization": f"Bearer {self.token}"})
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["error"]["code"], "item_not_found")

    def test_delete_item(self):
        item = self.client.post(
            "/items",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"type": "note", "title": "To delete", "content": "ephemeral"},
        ).json()

        delete_resp = self.client.delete(
            f"/items/{item['id']}",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        self.assertEqual(delete_resp.status_code, 200)
        self.assertTrue(delete_resp.json()["deleted"])

        get_resp = self.client.get(
            f"/items/{item['id']}",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        self.assertEqual(get_resp.status_code, 404)

    @unittest.skipUnless(
        os.getenv("BLACKHOLE_RUN_OPENAI_TESTS") == "1" and os.getenv("OPENAI_API_KEY"),
        "Set BLACKHOLE_RUN_OPENAI_TESTS=1 and OPENAI_API_KEY to run live OpenAI tests.",
    )
    def test_live_agent_creates_todo_end_to_end(self):
        response = self.client.post(
            "/agent",
            headers={"Authorization": f"Bearer {self.token}"},
            json={"message": "Create a todo called 'agent smoke test'"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIsInstance(body["response"], str)
        self.assertGreater(len(body["response"]), 0)
        self.assertIn("run_id", body)
        tool_names = [tc["name"] for tc in body["tool_calls"]]
        self.assertIn("create_item", tool_names)
        self.assertEqual(len(body["items_created"]), 1)
        self.assertIn("smoke", body["items_created"][0]["title"].lower())
        self.assertEqual(body["items_created"][0]["type"], "todo")
        self.assertEqual(body["items_created"][0]["source"], "agent")

    def test_agent_loop_creates_item_and_logs_tool_calls(self):
        agent_loop = sys.modules["app.agent.loop"]

        turns = [
            {
                "response": None,
                "tool_calls": [{"name": "create_item", "arguments": {"type": "todo", "title": "Review V2", "content": "Review the V2 blackhole build", "tags": ["blackhole"]}}],
            },
            {"response": "Created Review V2.", "tool_calls": []},
        ]

        def fake_step(**_):
            return turns.pop(0)

        with patch.object(agent_loop, "_model_step", side_effect=fake_step):
            response = self.client.post(
                "/agent",
                headers={"Authorization": f"Bearer {self.token}"},
                json={"message": "Create a todo to review V2"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["response"], "Created Review V2.")
        self.assertEqual(body["items_created"][0]["title"], "Review V2")
        self.assertEqual(body["items_created"][0]["source"], "agent")
        self.assertEqual(body["tool_calls"], [{"name": "create_item", "status": "success"}])

        conn = self.db.get_conn()
        run = conn.execute("SELECT * FROM agent_runs WHERE id = ?", (body["run_id"],)).fetchone()
        self.assertEqual(run["status"], "success")
        self.assertIn("recent_items", run["context_json"])
        tool_call = conn.execute("SELECT * FROM agent_tool_calls WHERE run_id = ?", (body["run_id"],)).fetchone()
        self.assertEqual(tool_call["name"], "create_item")
        self.assertEqual(tool_call["status"], "success")

    def test_agent_loop_reports_tool_errors_without_mutating_response(self):
        agent_loop = sys.modules["app.agent.loop"]

        turns = [
            {
                "response": None,
                "tool_calls": [{"name": "update_item", "arguments": {"id": "missing", "title": "Nope"}}],
            },
            {"response": "I could not update that item because it was not found.", "tool_calls": []},
        ]

        def fake_step(**_):
            return turns.pop(0)

        with patch.object(agent_loop, "_model_step", side_effect=fake_step):
            response = self.client.post(
                "/agent",
                headers={"Authorization": f"Bearer {self.token}"},
                json={"message": "Rename the missing item"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["items_updated"], [])
        self.assertEqual(body["tool_calls"], [{"name": "update_item", "status": "error"}])

        conn = self.db.get_conn()
        tool_call = conn.execute("SELECT * FROM agent_tool_calls WHERE run_id = ?", (body["run_id"],)).fetchone()
        self.assertEqual(tool_call["status"], "error")
        self.assertIn("not found", tool_call["error"].lower())

    def test_agent_model_step_calls_openai_json_object_format(self):
        agent_loop = sys.modules["app.agent.loop"]
        recorded = {}
        fake_response = object()

        def fake_create(**kwargs):
            recorded.update(kwargs)
            return fake_response

        def fake_response_text(resp):
            self.assertIs(resp, fake_response)
            return '{"response": "Hello.", "tool_calls": []}'

        with patch.object(agent_loop.client, "create_json_response", side_effect=fake_create), \
             patch.object(agent_loop.client, "response_text", side_effect=fake_response_text):
            result = agent_loop._model_step(
                model="test-model",
                input_messages=[{"role": "user", "content": "Hello"}],
            )

        self.assertEqual(result, {"response": "Hello.", "tool_calls": []})
        self.assertEqual(recorded["model"], "test-model")
        self.assertEqual(recorded["instructions"], agent_loop.AGENT_SYSTEM)
        self.assertEqual(recorded["max_output_tokens"], agent_loop.AGENT_MAX_OUTPUT_TOKENS)
        # json_object format: no strict schema passed
        self.assertNotIn("schema_name", recorded)
        self.assertNotIn("schema", recorded)

    def test_normalize_tool_calls_handles_arguments_as_object_or_string(self):
        agent_loop = sys.modules["app.agent.loop"]

        # Primary case: arguments as a plain object (the bug fix)
        calls = [
            {"name": "create_item", "arguments": {"type": "todo", "title": "Test"}},
            {"name": "update_item", "arguments": {"id": "abc", "status": "done"}},
        ]
        result = agent_loop._normalize_tool_calls(calls)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["name"], "create_item")
        self.assertEqual(result[0]["arguments"]["title"], "Test")
        self.assertEqual(result[1]["arguments"]["id"], "abc")

        # Legacy: arguments_json as a JSON-encoded string
        legacy = [{"name": "list_items", "arguments_json": '{"type": "todo", "status": "open"}'}]
        result = agent_loop._normalize_tool_calls(legacy)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["arguments"]["type"], "todo")

        # Unknown tool names are filtered out
        result = agent_loop._normalize_tool_calls([{"name": "delete_everything", "arguments": {}}])
        self.assertEqual(result, [])

        # Capped at MAX_TOOL_CALLS_PER_TURN
        many = [{"name": "list_items", "arguments": {}} for _ in range(agent_loop.MAX_TOOL_CALLS_PER_TURN + 2)]
        result = agent_loop._normalize_tool_calls(many)
        self.assertEqual(len(result), agent_loop.MAX_TOOL_CALLS_PER_TURN)


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
                return SimpleNamespace(output_text='{"items":[{"type":"note","title":"Test","content":null,"status":null,"priority":null,"tags":[],"parent_title":null,"due_date":null,"start_time":null,"end_time":null,"location":null,"url":null,"read_status":null,"email":null,"phone":null,"organization":null,"recurrence_rule":null,"metadata":null}]}')

        fake_client = SimpleNamespace(responses=FakeResponses())

        with patch.object(self.analysis, "get_client", return_value=fake_client):
            result, log = self.analysis.run_transcript_analysis("capture this")

        self.assertEqual(log["status"], "success")
        self.assertEqual(result[0]["title"], "Test")
        self.assertEqual(recorded["max_output_tokens"], 1200)
        self.assertEqual(recorded["text"]["format"]["type"], "json_schema")
        self.assertEqual(recorded["text"]["format"]["name"], "transcript_items")
        type_enum = recorded["text"]["format"]["schema"]["properties"]["items"]["items"]["properties"]["type"]["enum"]
        self.assertIn("epic", type_enum)
        self.assertIn("event", type_enum)
        self.assertIn("habit", type_enum)
        properties = recorded["text"]["format"]["schema"]["properties"]["items"]["items"]["properties"]
        self.assertIn("parent_title", properties)
        self.assertNotIn("epic_title", properties)
        self.assertIn("Do not create a duplicate raw note", recorded["instructions"])
        self.assertIn("Epics are stronger categories", recorded["instructions"])
        self.assertIn("input", recorded)
        self.assertIn("instructions", recorded)
        self.assertNotIn("max_tokens", recorded)
        self.assertNotIn("max_completion_tokens", recorded)
        # recurrence_rule must use additionalProperties: false (required by OpenAI strict mode)
        recurrence_schema = properties["recurrence_rule"]
        self.assertFalse(recurrence_schema.get("additionalProperties"))
        self.assertIn("freq", recurrence_schema["properties"])
        # metadata is now a string type (open-ended objects aren't allowed in strict schemas)
        self.assertIn("string", properties["metadata"]["type"])

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
                return SimpleNamespace(output_text='{"items":[{"type":"todo","title":"Buy oat milk","content":null,"status":null,"priority":null,"tags":["errand"],"parent_title":null,"due_date":null,"start_time":null,"end_time":null,"location":null,"url":null,"read_status":null,"email":null,"phone":null,"organization":null,"recurrence_rule":null,"metadata":null}]}')

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
        self.assertIn("parent_title", messages[2]["content"])
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
                return SimpleNamespace(output_text='{"items":[{"type":"note","title":"Test","content":null,"status":null,"priority":null,"tags":[],"parent_title":null,"due_date":null,"start_time":null,"end_time":null,"location":null,"url":null,"read_status":null,"email":null,"phone":null,"organization":null,"recurrence_rule":null,"metadata":null}]}')

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
        self.assertIn(result[0]["type"], {"note", "todo", "event", "epic", "contact", "resource", "decision", "journal", "habit"})
        self.assertTrue(result[0]["title"] is None or isinstance(result[0]["title"], str))
        self.assertIsInstance(result[0]["tags"], list)
        self.assertIn("due_date", result[0])
        self.assertIn("parent_title", result[0])


if __name__ == "__main__":
    unittest.main()
