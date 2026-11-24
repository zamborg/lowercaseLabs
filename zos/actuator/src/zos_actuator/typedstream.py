"""Decode Apple NSArchiver / typedstream `attributedBody` blobs from macOS Messages chat.db."""

from __future__ import annotations

import struct

_NSSTRING_MARKERS = (b"NSMutableString", b"NSString")


def _read_typedstream_length(data: bytes, pos: int) -> tuple[int, int] | None:
    """Read an Apple typedstream variable-length integer starting at `pos`.

    Format:
      - byte < 0x80: value is that byte directly (1 byte)
      - 0x81: next 2 bytes, little-endian unsigned short
      - 0x82: next 4 bytes, little-endian unsigned int
      - 0x83: next 8 bytes, little-endian unsigned long long
    Returns (length, next_pos) or None if truncated/invalid.
    """
    if pos >= len(data):
        return None
    first = data[pos]
    if first < 0x80:
        return first, pos + 1
    if first == 0x81 and pos + 3 <= len(data):
        return struct.unpack_from("<H", data, pos + 1)[0], pos + 3
    if first == 0x82 and pos + 5 <= len(data):
        return struct.unpack_from("<I", data, pos + 1)[0], pos + 5
    if first == 0x83 and pos + 9 <= len(data):
        return struct.unpack_from("<Q", data, pos + 1)[0], pos + 9
    return None


def decode_attributed_body(blob: bytes | memoryview | None) -> str:
    """Extract the plain UTF-8 text string from a macOS `message.attributedBody` typedstream blob.

    Returns an empty string if `blob` is None, empty, or cannot be decoded.
    """
    if not blob:
        return ""
    data = bytes(blob)

    # Locate NSString / NSMutableString class marker, then find the `\x01+` string tag.
    for marker in _NSSTRING_MARKERS:
        marker_idx = data.find(marker)
        if marker_idx == -1:
            continue

        # In Apple's typedstream encoding, the NSString payload begins after the
        # class header with a '+' (0x2B) byte (usually preceded by 0x01) followed
        # immediately by the typedstream length and raw UTF-8 bytes.
        search_start = marker_idx + len(marker)
        plus_idx = data.find(b"\x01+", search_start)
        if plus_idx != -1:
            len_pos = plus_idx + 2
        else:
            plus_idx = data.find(b"+", search_start)
            if plus_idx == -1:
                continue
            len_pos = plus_idx + 1

        parsed = _read_typedstream_length(data, len_pos)
        if parsed is None:
            continue
        length, text_start = parsed
        text_end = text_start + length
        if length == 0:
            return ""
        if text_end <= len(data):
            return data[text_start:text_end].decode("utf-8", errors="replace")

    return ""


def build_typedstream_attributed_body(text: str) -> bytes:
    """Construct a valid Apple typedstream NSAttributedString/NSString blob for fixture tests."""
    encoded = text.encode("utf-8")
    n = len(encoded)
    if n < 0x80:
        len_bytes = bytes([n])
    elif n <= 0xFFFF:
        len_bytes = b"\x81" + struct.pack("<H", n)
    elif n <= 0xFFFFFFFF:
        len_bytes = b"\x82" + struct.pack("<I", n)
    else:
        len_bytes = b"\x83" + struct.pack("<Q", n)

    return (
        b"\x04\x0bstreamtyped\x81\xe8\x03\x84\x01@\x84\x84\x84"
        b"\x12NSAttributedString\x00\x84\x84\x08NSObject\x00\x85\x92\x84\x84\x84"
        b"\x08NSString\x01\x94\x84\x01+"
        + len_bytes
        + encoded
        + b"\x86\x84\x02iI\x01\x05\x92\x84\x84\x84\x0cNSDictionary\x00\x94\x84\x01i\x01\x86\x86"
    )
