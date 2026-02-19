from app.tagging import TAG_CATALOG, clean_tags, extract_mood_tags, extract_themes


def test_tag_catalog_has_expected_size() -> None:
    assert len(TAG_CATALOG) == 50


def test_extract_mood_tags_returns_limited_matches() -> None:
    lowered = "i feel anxious stressed overwhelmed and hopeful about tomorrow"
    tags = extract_mood_tags(lowered)

    assert tags
    assert len(tags) <= 4
    assert "anxious" in tags
    assert "hopeful" in tags


def test_clean_tags_normalizes_and_limits() -> None:
    cleaned = clean_tags(["  Work_Pressure ", "burnout", "unknown", "work pressure", "sad"], maximum=4)

    assert cleaned == ["work pressure", "burned out", "sad"]


def test_extract_themes_detects_keyword_themes() -> None:
    lowered = "work deadlines and family issues made sleep hard"
    themes = extract_themes(lowered)

    assert "work" in themes
    assert "relationships" in themes
    assert "health" in themes
