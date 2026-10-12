from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Response

from ..deps import get_current_user, get_services
from ..errors import AppError
from ..schemas import (
    Page,
    PlaylistCreate,
    PlaylistOut,
    PlaylistSongAdd,
    PlaylistUpdate,
    serialize_playlist,
    serialize_song,
)
from ..services import Services
from ..utils import Pagination, oid, page_envelope, pagination, utcnow

router = APIRouter(prefix="/api/v1/playlists", tags=["playlists"])


async def _load(s: Services, playlist_id: str, user: dict, *, write: bool) -> dict:
    doc = await s.db.playlists.find_one({"_id": oid(playlist_id, "playlist_id")})
    owner = doc is not None and doc["owner_id"] == user["_id"]
    # Private playlists answer 404 (not 403) to non-owners so their existence isn't revealed.
    if doc is None or not (owner or (doc.get("is_public") and not write)):
        raise AppError(404, "playlist_not_found", "Playlist not found")
    return doc


async def _expand(s: Services, doc: dict) -> PlaylistOut:
    ids = doc.get("song_ids", [])
    by_id = {d["_id"]: d async for d in s.db.songs.find({"_id": {"$in": ids}})}
    songs = [serialize_song(by_id[i]) for i in ids if i in by_id]  # keep playlist order
    return serialize_playlist(doc, songs)


async def _validated_song_ids(s: Services, raw_ids: list[str]) -> list:
    ids = []
    for raw in raw_ids:
        _id = oid(raw, "song_id")
        if _id not in ids:
            ids.append(_id)
    if ids and await s.db.songs.count_documents({"_id": {"$in": ids}}) != len(ids):
        raise AppError(422, "unknown_song", "One or more song_ids do not exist")
    return ids


@router.get("", response_model=Page[PlaylistOut])
async def list_playlists(
    scope: str = Query("mine", pattern="^(mine|public)$"),
    pg: Pagination = Depends(pagination),
    user: dict = Depends(get_current_user),
    s: Services = Depends(get_services),
):
    flt = {"owner_id": user["_id"]} if scope == "mine" else {"is_public": True}
    total = await s.db.playlists.count_documents(flt)
    cursor = s.db.playlists.find(flt).sort([("updated_at", -1), ("_id", 1)]).skip(pg.skip).limit(pg.limit)
    return page_envelope([serialize_playlist(d) async for d in cursor], total, pg)


@router.post("", response_model=PlaylistOut, status_code=201)
async def create_playlist(body: PlaylistCreate, user: dict = Depends(get_current_user), s: Services = Depends(get_services)):
    now = utcnow()
    doc = {
        "owner_id": user["_id"],
        "name": body.name,
        "description": body.description,
        "is_public": body.is_public,
        "source": "manual",
        "song_ids": await _validated_song_ids(s, body.song_ids),
        "created_at": now,
        "updated_at": now,
    }
    doc["_id"] = (await s.db.playlists.insert_one(doc)).inserted_id
    return serialize_playlist(doc)


@router.get("/{playlist_id}", response_model=PlaylistOut)
async def get_playlist(
    playlist_id: str,
    expand: bool = Query(True, description="Include full song objects"),
    user: dict = Depends(get_current_user),
    s: Services = Depends(get_services),
):
    doc = await _load(s, playlist_id, user, write=False)
    return await _expand(s, doc) if expand else serialize_playlist(doc)


@router.patch("/{playlist_id}", response_model=PlaylistOut)
async def update_playlist(
    playlist_id: str, body: PlaylistUpdate, user: dict = Depends(get_current_user), s: Services = Depends(get_services)
):
    doc = await _load(s, playlist_id, user, write=True)
    changes = body.model_dump(exclude_unset=True)
    if not changes:
        raise AppError(422, "no_changes", "No fields to update")
    changes["updated_at"] = utcnow()
    await s.db.playlists.update_one({"_id": doc["_id"]}, {"$set": changes})
    doc.update(changes)
    return serialize_playlist(doc)


@router.delete("/{playlist_id}", status_code=204)
async def delete_playlist(playlist_id: str, user: dict = Depends(get_current_user), s: Services = Depends(get_services)):
    doc = await _load(s, playlist_id, user, write=True)
    await s.db.playlists.delete_one({"_id": doc["_id"]})
    return Response(status_code=204)


@router.post("/{playlist_id}/songs", response_model=PlaylistOut)
async def add_song(
    playlist_id: str, body: PlaylistSongAdd, user: dict = Depends(get_current_user), s: Services = Depends(get_services)
):
    doc = await _load(s, playlist_id, user, write=True)
    song_id = oid(body.song_id, "song_id")
    if not await s.db.songs.count_documents({"_id": song_id}, limit=1):
        raise AppError(404, "song_not_found", "Song not found")
    updated = await s.db.playlists.find_one_and_update(
        {"_id": doc["_id"]},
        {"$addToSet": {"song_ids": song_id}, "$set": {"updated_at": utcnow()}},
        return_document=True,
    )
    return serialize_playlist(updated)


@router.delete("/{playlist_id}/songs/{song_id}", response_model=PlaylistOut)
async def remove_song(
    playlist_id: str, song_id: str, user: dict = Depends(get_current_user), s: Services = Depends(get_services)
):
    doc = await _load(s, playlist_id, user, write=True)
    updated = await s.db.playlists.find_one_and_update(
        {"_id": doc["_id"]},
        {"$pull": {"song_ids": oid(song_id, "song_id")}, "$set": {"updated_at": utcnow()}},
        return_document=True,
    )
    return serialize_playlist(updated)
