from app.llm import extract_insights
from app.social import mood_to_dot_color
from app.config import settings


def test_extract_insights_basic_shape(monkeypatch) -> None:
    monkeypatch.setattr(settings, "openai_api_key", None)

    insights = extract_insights(
        "I feel stressed and overwhelmed about work deadlines, but I am still hopeful about tomorrow."
    )

    assert "mood_score" in insights
    assert isinstance(insights["mood_tags"], list)
    assert 1 <= len(insights["mood_tags"]) <= 4
    assert isinstance(insights["summary"], str)
    assert isinstance(insights["themes"], list)
    assert isinstance(insights["signals"], dict)
    assert isinstance(insights["safety_flags"], dict)


def test_mood_to_dot_color_bins() -> None:
    assert mood_to_dot_color(-2.0).startswith("#")
    assert mood_to_dot_color(0.0).startswith("#")
    assert mood_to_dot_color(2.0).startswith("#")
