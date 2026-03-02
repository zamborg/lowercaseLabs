from __future__ import annotations

from .models import SocialPresence, User


DOT_PALETTE = [
    (-2.0, "#4A1E26"),
    (-1.0, "#7A2E3A"),
    (-0.25, "#A45F4F"),
    (0.25, "#6C6F74"),
    (1.0, "#4D8E78"),
    (1.5, "#2F9D89"),
    (2.0, "#18A999"),
]

SILENT_DOT_COLOR = "#2D313A"


def mood_to_dot_color(mood_score: float) -> str:
    for threshold, color in DOT_PALETTE:
        if mood_score <= threshold:
            return color
    return DOT_PALETTE[-1][1]


def visible_label_for_viewer(presence: SocialPresence | None, owner: User, viewer_user_id: str) -> str | None:
    _ = viewer_user_id
    if presence is None:
        return owner.display_name or owner.anonymous_handle
    return presence.display_name_override or owner.display_name or owner.anonymous_handle
