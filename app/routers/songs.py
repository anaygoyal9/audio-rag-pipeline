from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, Query, Response, UploadFile
from pymongo.errors import DuplicateKeyError

from ..deps import get_current_user, get_services, require_admin
from ..errors import AppError
from ..rag import indexer
from ..schemas import Page, SongCreate, SongOut, SongUpdate, serialize_song
from ..services import Services
from ..streaming import CONTENT_TYPES, sniff_audio_extension
from ..utils import Pagination, oid, page_envelope, pagination, utcnow

router = APIRouter(prefix="/api/v1/songs", tags=["songs"])

SORTS = {
    "-created_at": [("created_at", -1)],
    "title": [("title", 1)],
    "-title": [("title", -1)],
    "tempo": [("tempo", 1)],
    "-tempo": [("tempo", -1)],
    "popularity": [("play_count", -1)],
}


@router.get("", response_model=Page[SongOut])
async def list_songs(
    q: str | None = Query(None, min_length=1, max_length=100, description="Full-text search"),
    genre: str | None = Query(None, max_length=50),
    mood: str | None = Query(None, max_length=50),
    min_tempo: float | None = Query(None, ge=0),
    max_tempo: float | None = Query(None, ge=0),
    sort: Literal["-created_at", "title", "-title", "tempo", "-tempo", "popularity"] = "-created_at",
    pg: Pagination = Depends(pagination),
    s: Services = Depends(get_services),
    _user: dict = Depends(get_current_user),
):
    flt: dict = {}
    if q:
        flt["$text"] = {"$search": q}
    if genre:
        flt["genre"] = genre.strip().lower()
    if mood:
        flt["mood"] = mood.strip().lower()
    if min_tempo is not None or max_tempo is not None:
        flt["tempo"] = {}
        if min_tempo is not None:
            flt["tempo"]["$gte"] = min_tempo
        if max_tempo is not None:
            flt["tempo"]["$lte"] = max_tempo

    total = await s.db.songs.count_documents(flt)
    cursor = s.db.songs.find(flt).sort(SORTS[sort] + [("_id", 1)]).skip(pg.skip).limit(pg.limit)
    items = [serialize_song(d) async for d in cursor]
    return page_envelope(items, total, pg)


@router.get("/facets")
async def facets(s: Services = Depends(get_services), _user: dict = Depends(get_current_user)):
    """Distinct genres/moods in the catalogue (handy for filter dropdowns)."""
    return {
        "genres": sorted(await s.db.songs.distinct("genre")),
        "moods": sorted(await s.db.songs.distinct("mood")),
    }


@router.get("/{song_id}", response_model=SongOut)
async def get_song(song_id: str, s: Services = Depends(get_services), _user: dict = Depends(get_current_user)):
    doc = await s.db.songs.find_one({"_id": oid(song_id, "song_id")})
    if not doc:
        raise AppError(404, "song_not_found", "Song not found")
    return serialize_song(doc)


@router.post("", response_model=SongOut, status_code=201)
async def create_song(body: SongCreate, s: Services = Depends(get_services), _admin: dict = Depends(require_admin)):
    now = utcnow()
    doc = {**body.model_dump(), "play_count": 0, "created_at": now, "updated_at": now}
    try:
        result = await s.db.songs.insert_one(doc)
    except DuplicateKeyError:
        raise AppError(409, "song_exists", "This artist already has a song with that title") from None
    doc["_id"] = result.inserted_id
    try:
        await indexer.index_songs(s, [doc])
    except Exception as exc:  # noqa: BLE001
        # Keep Mongo and the vector index consistent: undo the insert rather than leave an
        # unsearchable track behind.
        await s.db.songs.delete_one({"_id": result.inserted_id})
        raise AppError(502, "indexing_failed", "Could not index the song for search; nothing was saved") from exc
    return serialize_song(doc)


@router.patch("/{song_id}", response_model=SongOut)
async def update_song(
    song_id: str, body: SongUpdate, s: Services = Depends(get_services), _admin: dict = Depends(require_admin)
):
    changes = body.model_dump(exclude_unset=True)
    if not changes:
        raise AppError(422, "no_changes", "No fields to update")
    changes["updated_at"] = utcnow()
    try:
        doc = await s.db.songs.find_one_and_update(
            {"_id": oid(song_id, "song_id")}, {"$set": changes}, return_document=True
        )
    except DuplicateKeyError:
        raise AppError(409, "song_exists", "This artist already has a song with that title") from None
    if not doc:
        raise AppError(404, "song_not_found", "Song not found")
    await indexer.index_songs(s, [doc])  # re-embed: text/metadata may have changed
    return serialize_song(doc)


@router.delete("/{song_id}", status_code=204)
async def delete_song(song_id: str, s: Services = Depends(get_services), _admin: dict = Depends(require_admin)):
    _id = oid(song_id, "song_id")
    doc = await s.db.songs.find_one_and_delete({"_id": _id})
    if not doc:
        raise AppError(404, "song_not_found", "Song not found")
    await indexer.remove_song(s, song_id)
    await s.db.playlists.update_many({"song_ids": _id}, {"$pull": {"song_ids": _id}})
    if doc.get("audio_path"):
        try:
            (Path(s.settings.audio_dir) / doc["audio_path"]).unlink(missing_ok=True)
        except OSError:
            pass
    return Response(status_code=204)


@router.post("/{song_id}/audio", response_model=SongOut)
async def upload_audio(
    song_id: str,
    file: UploadFile = File(...),
    s: Services = Depends(get_services),
    _admin: dict = Depends(require_admin),
):
    _id = oid(song_id, "song_id")
    song = await s.db.songs.find_one({"_id": _id})
    if not song:
        raise AppError(404, "song_not_found", "Song not found")

    ext = Path(file.filename or "").suffix.lower()
    if ext not in CONTENT_TYPES:
        raise AppError(415, "unsupported_media_type", f"Allowed formats: {', '.join(sorted(CONTENT_TYPES))}")

    audio_dir = Path(s.settings.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = s.settings.max_upload_mb * 1024 * 1024

    # Stream to a temp file in chunks (bounded memory), validating size and magic bytes as we go.
    fd, tmp_name = tempfile.mkstemp(dir=audio_dir, suffix=".part")
    size = 0
    try:
        with os.fdopen(fd, "wb") as out:
            first = True
            while chunk := await file.read(1024 * 1024):
                if first:
                    sniffed = sniff_audio_extension(chunk[:16])
                    if sniffed is None:
                        raise AppError(415, "not_audio", "File content is not a recognised audio format")
                    ext = sniffed  # trust the bytes over the filename
                    first = False
                size += len(chunk)
                if size > max_bytes:
                    raise AppError(413, "file_too_large", f"Maximum upload size is {s.settings.max_upload_mb} MB")
                out.write(chunk)
        if size == 0:
            raise AppError(422, "empty_file", "Uploaded file is empty")
        final_name = f"{song_id}{ext}"
        os.replace(tmp_name, audio_dir / final_name)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise

    old = song.get("audio_path")
    if old and old != final_name:
        (audio_dir / old).unlink(missing_ok=True)

    doc = await s.db.songs.find_one_and_update(
        {"_id": _id},
        {
            "$set": {
                "audio_path": final_name,
                "audio_content_type": CONTENT_TYPES[ext],
                "audio_size": size,
                "updated_at": utcnow(),
            }
        },
        return_document=True,
    )
    return serialize_song(doc)
