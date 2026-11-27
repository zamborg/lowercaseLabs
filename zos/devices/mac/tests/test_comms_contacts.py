"""Contacts: fuzzy search, labels, ids, and get (fixtures only)."""

from __future__ import annotations

from conftest import JARED_ID, MacFixture
from zos_mac.contacts import clean_label


def test_clean_label() -> None:
    assert clean_label("_$!<Mobile>!$_") == "mobile"
    assert clean_label("_$!<HomeFAX>!$_") == "homefax"
    assert clean_label("iPhone") == "iPhone"
    assert clean_label("  ") is None
    assert clean_label(None) is None


def test_contacts_search_fuzzy_name_nickname_org(mac_fixture: MacFixture) -> None:
    c = mac_fixture.contacts

    def names(q: str) -> list[str]:
        return [hit.display_name for hit, _ in c.search(q)]

    assert names("Jared")[0] == "Jared Dunn"
    assert names("jar")[0] == "Jared Dunn"  # prefix
    assert names("Jarad")[0] == "Jared Dunn"  # typo
    assert names("jared dunn") == ["Jared Dunn"]  # multi-token AND
    assert names("JD") == ["Jared Dunn"]  # nickname
    assert names("Richie") == ["Richard Hendricks"]
    assert names("hooli") == ["Gavin Belson"]  # organization
    assert names("pied piper") == ["Pied Piper"]  # org-only contact
    assert names("jared hall") == []  # tokens must all match
    assert names("zzzz") == []
    # Exact handle lookups
    assert names("(650) 555-0102") == ["Jared Dunn"]
    assert names("JARED@example.com") == ["Jared Dunn"]


def test_contacts_search_ordering_and_limit(mac_fixture: MacFixture) -> None:
    hits = mac_fixture.contacts.search("h", limit=10)
    # "h" prefixes Hall, Hendricks, Hooli -> all returned, alphabetical among equal scores
    assert [h.display_name for h, _ in hits] == ["Gavin Belson", "Monica Hall", "Richard Hendricks"]
    assert len(mac_fixture.contacts.search("h", limit=1)) == 1


def test_contacts_labels_and_get(mac_fixture: MacFixture) -> None:
    c = mac_fixture.contacts
    jared = c.search("Jared")[0][0].to_dict()
    assert jared["id"] == JARED_ID
    assert jared["nickname"] == "JD"
    assert jared["phones"] == [
        {"value": "+1 (415) 555-0101", "label": "mobile"},
        {"value": "650-555-0102", "label": "work"},
    ]
    assert jared["emails"] == [{"value": "Jared@Example.com", "label": "home"}]

    gavin = c.search("Gavin")[0][0]
    assert gavin.id == "SRC-A:3"  # no ZUNIQUEID -> source tag + pk
    assert [p.label for p in gavin.phones] == ["mobile", "iPhone"]
    got = c.get("SRC-A:3")
    assert got is not None
    assert got.to_dict()["organization"] == "Hooli"
    assert c.get("nope") is None
    assert c.get("") is None
