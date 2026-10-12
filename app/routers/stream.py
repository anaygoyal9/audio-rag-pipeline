"""Audio streaming with HTTP Range support (RFC 9110 section 14).

* No Range header          -> 200 with the whole file, ``Accept-Ranges: bytes``
* ``Range: bytes=a-b``     -> 206 Partial Content + ``Content-Range``
* Range outside the file   -> 416 with ``Content-Range: bytes */size``

The file is read in fixed-size chunks by a generator, so memory per request is one chunk no
matter how large the track is, and a player can start after the first chunk and seek with
a new Range request instead of downloading everything.
"""

from __future__ import annotations

import os
from pathlib import Path

from anyio import to_thread
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from ..deps import get_current_user_stream, get_services
from ..errors import AppError
from ..services import Services
from ..streaming import RangeNotSatisfiable, iter_file, make_etag, parse_range
from ..utils import oid
from .history import record_play

router = APIRouter(prefix="/api/v1/stream", tags=["streaming"])


@router.api_route("/{song_id}", methods=["GET", "HEAD"])
async def stream_song(
    song_id: str,
    request: Request,
    s: Services = Depends(get_services),
    user: dict = Depends(get_current_user_stream),
):
    song = await s.db.songs.find_one({"_id": oid(song_id, "song_id")})
    if not song or not song.get("audio_path"):
        raise AppError(404, "audio_not_found", "No audio available for this song")

    root = Path(s.settings.audio_dir).resolve()
    path = (root / song["audio_path"]).resolve()
    if not path.is_relative_to(root):  # defence in depth against a tampered audio_path
        raise AppError(404, "audio_not_found", "No audio available for this song")
    try:
        st = await to_thread.run_sync(os.stat, path)
    except FileNotFoundError:
        raise AppError(404, "audio_not_found", "Audio file is missing") from None

    size = st.st_size
    etag = make_etag(size, st.st_mtime_ns)
    base_headers = {
        "Accept-Ranges": "bytes",
        "ETag": etag,
        "Cache-Control": "private, max-age=3600",
    }
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=base_headers)

    range_header = request.headers.get("range")
    if_range = request.headers.get("if-range")
    if if_range and if_range != etag:
        range_header = None  # validator mismatch: client's cached copy is stale, send it all

    try:
        byte_range = parse_range(range_header, size)
    except RangeNotSatisfiable:
        raise AppError(
            416, "range_not_satisfiable", "Requested range is outside the file", headers={"Content-Range": f"bytes */{size}"}
        ) from None

    if byte_range is None:
        status, start, end = 200, 0, size - 1
        headers = {**base_headers, "Content-Length": str(size)}
    else:
        status = 206
        start, end = byte_range
        headers = {
            **base_headers,
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Content-Length": str(end - start + 1),
        }
    content_type = song.get("audio_content_type", "application/octet-stream")

    if request.method == "HEAD":
        return Response(status_code=status, headers={**headers, "Content-Type": content_type})

    # Count a play when playback starts (first byte), not on every seek.
    background = BackgroundTask(record_play, s, user["_id"], song["_id"]) if start == 0 else None
    return StreamingResponse(
        iter_file(str(path), start, end, s.settings.stream_chunk_size),
        status_code=status,
        headers=headers,
        media_type=content_type,
        background=background,
    )
