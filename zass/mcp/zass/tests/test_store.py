"""Store tests. Run: uv run --project mcp/zass pytest mcp/zass/tests -q"""

import pytest

from zass_mcp import store


@pytest.fixture(autouse=True)
def tmp_state(tmp_path, monkeypatch):
    monkeypatch.setenv("ZASS_STATE_DIR", str(tmp_path))
    return tmp_path


def test_slugify():
    assert store.slugify("Pay rent (August!)") == "pay-rent-august"
    assert store.slugify("   ") == "item"


def test_todo_lifecycle():
    meta = store.todo_add("Pay rent", due="2026-08-01", priority="high", tags=["finance"])
    assert meta["id"] == "pay-rent"

    rows = store.todo_list()
    assert len(rows) == 1
    assert rows[0]["title"] == "Pay rent"

    # collision gets a suffix
    meta2 = store.todo_add("Pay rent")
    assert meta2["id"] == "pay-rent-2"

    # filters
    assert len(store.todo_list(tag="finance")) == 1
    assert len(store.todo_list(due_within_days=365)) == 1
    assert len(store.todo_list(due_within_days=0)) == 0

    # complete
    done = store.todo_update("pay-rent", status="done")
    assert done["completed"] is not None
    assert len(store.todo_list()) == 1  # only pay-rent-2 remains open
    assert len(store.todo_list(include_done=True)) == 2


def test_todo_update_append_note():
    store.todo_add("Call plumber")
    store.todo_update("call-plumber", append_note="left voicemail")
    rows = store.todo_list()
    assert "left voicemail" in rows[0]["notes"]


def test_todo_sorting():
    store.todo_add("later", due="2026-12-01")
    store.todo_add("soon", due="2026-08-01")
    store.todo_add("no due date")
    rows = store.todo_list()
    assert [r["title"] for r in rows] == ["soon", "later", "no due date"]


def test_notes():
    meta = store.note_add("Trip ideas", "- Kyoto\n- Lisbon", tags=["travel"])
    assert meta["id"] == "trip-ideas"

    got = store.note_get("trip-ideas")
    assert got is not None
    assert "Kyoto" in got[1]

    store.note_update("trip-ideas", append="- Oaxaca")
    _, body = store.note_get("trip-ideas")
    assert "Oaxaca" in body and "Kyoto" in body

    # partial id match
    assert store.note_get("trip") is not None
    assert store.note_list(tag="travel")[0]["title"] == "Trip ideas"


def test_people_merge_semantics():
    store.person_upsert("Jane Doe", relation="friend", emails=["jane@x.com"])
    store.person_upsert("Jane Doe", emails=["jane@work.com"], phones=["+15551234567"])
    meta, _ = store.person_get("jane-doe")
    assert meta["emails"] == ["jane@x.com", "jane@work.com"]
    assert meta["relation"] == "friend"

    store.person_log("jane", "coffee at Blue Bottle")
    _, body = store.person_get("jane-doe")
    assert "coffee at Blue Bottle" in body

    assert len(store.person_list(query="friend")) == 1
    assert store.person_list(query="nobody") == []


def test_memory():
    store.memory_save("Prefers morning meetings", slug="morning-meetings", kind="preference")
    rows = store.memory_list(query="morning")
    assert len(rows) == 1
    assert rows[0][0]["type"] == "preference"

    # overwrite same slug corrects the fact
    store.memory_save("Prefers afternoon meetings", slug="morning-meetings", kind="preference")
    rows = store.memory_list()
    assert len(rows) == 1
    assert "afternoon" in rows[0][1]

    assert store.memory_forget("morning-meetings") is True
    assert store.memory_list() == []


def test_journal_and_inbox():
    store.journal_log("set up zass")
    store.journal_log("second entry")
    text = store.journal_read()
    assert "set up zass" in text and "second entry" in text

    store.capture("look into flight prices")
    assert "flight prices" in store.inbox_read()


def test_search():
    store.note_add("Recipes", "carbonara with guanciale")
    store.todo_add("buy guanciale")
    hits = store.search("guanciale")
    files = {h["file"] for h in hits}
    assert any("notes/" in f for f in files)
    assert any("todos/" in f for f in files)

    hits = store.search("guanciale", kind="notes")
    assert all("notes/" in h["file"] for h in hits)

    assert store.search("zzz-nothing") == []


def test_frontmatter_roundtrip(tmp_state):
    p = tmp_state / "doc.md"
    store.write_doc(p, {"a": 1, "tags": ["x"]}, "body **here**")
    meta, body = store.read_doc(p)
    assert meta == {"a": 1, "tags": ["x"]}
    assert body.strip() == "body **here**"
