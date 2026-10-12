"""Pure helpers for HTTP Range streaming and upload validation (no third-party imports)."""

from __future__ import annotations

import re
from collections.abc import Iterator

CONTENT_TYPES = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
}

_RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


class RangeNotSatisfiable(Exception):
    def __init__(self, size: int) -> None:
        super().__init__(f"range not satisfiable for resource of {size} bytes")
        self.size = size


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Parse a single-range ``Range: bytes=...`` header.

    Returns an inclusive ``(start, end)`` pair, or ``None`` when the full resource should be
    served (no header, malformed header, or multi-range, all of which RFC 9110 allows a
    server to ignore). Raises ``RangeNotSatisfiable`` when the range lies outside the file.
    """
    if not header:
        return None
    header = header.strip()
    if "," in header:
        return None
    match = _RANGE_RE.match(header)
    if not match:
        return None
    first, last = match.groups()
    if first == "" and last == "":
        return None

    if first == "":  # suffix range: the last N bytes
        n = int(last)
        if n == 0 or size == 0:
            raise RangeNotSatisfiable(size)
        return max(size - n, 0), size - 1

    start = int(first)
    if last == "":
        end = size - 1
    else:
        end = int(last)
        if end < start:  # syntactically invalid: ignore the header
            return None
    if start >= size:
        raise RangeNotSatisfiable(size)
    return start, min(end, size - 1)


def iter_file(path: str, start: int, end: int, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
    """Yield bytes ``start..end`` (inclusive) in ``chunk_size`` pieces.

    Only one chunk is ever held in memory, so peak memory is independent of file size.
    """
    remaining = end - start + 1
    with open(path, "rb") as fh:
        fh.seek(start)
        while remaining > 0:
            data = fh.read(min(chunk_size, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


def make_etag(size: int, mtime_ns: int) -> str:
    return f'"{size:x}-{mtime_ns:x}"'


def sniff_audio_extension(head: bytes) -> str | None:
    """Identify an audio container from its magic bytes so uploads can't just lie about
    their extension. Returns the canonical extension or None."""
    if head[:3] == b"ID3" or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0):
        return ".mp3"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return ".wav"
    if head[:4] == b"OggS":
        return ".ogg"
    if head[:4] == b"fLaC":
        return ".flac"
    if head[4:8] == b"ftyp":
        return ".m4a"
    return None
