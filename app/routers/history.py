from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ..deps import get_current_user, get_services
from ..errors import AppError
from ..schemas import HistoryCreate, HistoryOut, Page
from ..services import Services
from ..utils import Pagination, oid, page_envelope, pagination, utcnow

router = APIRouter(prefix="/api/v1/history", tags=["listening history"])
log = logging.getLogger("app.history")


async def record_play(services: Services, user_id, song_id, *, position_sec: float = 0, completed: bool = False) -> None:
    """Shared by POST /history and the streaming endpoint (which logs a play on the first byte range)."""
    try:
        await services.db.listening_history.insert_one(
            {
                "user_id": user_id,
                "song_id": song_id,
                "played_at": utcnow(),
                "position_sec": position_sec,
                "completed": completed,
            }
        )
        await services.db.songs.update_one({"_id": song_id}, {"$inc": {"play_count": 1}})
    except Exception as exc:  # noqa: BLE001 - analytics must never break playback
        log.warning("could not record play: %s", exc)


@router.post("", response_model=HistoryOut, status_code=201)
async def add_history(body: HistoryCreate, user: dict = Depends(get_current_user), s: Services = Depends(get_services)):
    song_id = oid(body.song_id, "song_id")
    song = await s.db.songs.find_one({"_id": song_id}, {"title": 1, "artist": 1})
    if not song:
        raise AppError(404, "song_not_found", "Song not found")
    now = utcnow()
    entry = {
        "user_id": user["_id"],
        "song_id": song_id,
        "played_at": now,
        "position_sec": body.position_sec,
        "completed": body.completed,
    }
    entry["_id"] = (await s.db.listening_history.insert_one(entry)).inserted_id
    await s.db.songs.update_one({"_id": song_id}, {"$inc": {"play_count": 1}})
    return HistoryOut(
        id=str(entry["_id"]),
        song_id=body.song_id,
        played_at=now,
        position_sec=body.position_sec,
        completed=body.completed,
        song_title=song["title"],
        song_artist=song["artist"],
    )


@router.get("", response_model=Page[HistoryOut])
async def list_history(
    pg: Pagination = Depends(pagination), user: dict = Depends(get_current_user), s: Services = Depends(get_services)
):
    flt = {"user_id": user["_id"]}
    total = await s.db.listening_history.count_documents(flt)
    rows = [r async for r in s.db.listening_history.find(flt).sort("played_at", -1).skip(pg.skip).limit(pg.limit)]
    songs = {d["_id"]: d async for d in s.db.songs.find({"_id": {"$in": [r["song_id"] for r in rows]}}, {"title": 1, "artist": 1})}
    items = [
        HistoryOut(
            id=str(r["_id"]),
            song_id=str(r["song_id"]),
            played_at=r["played_at"],
            position_sec=r.get("position_sec", 0),
            completed=r.get("completed", False),
            song_title=songs.get(r["song_id"], {}).get("title"),
            song_artist=songs.get(r["song_id"], {}).get("artist"),
        )
        for r in rows
    ]
    return page_envelope(items, total, pg)


@router.get("/stats")
async def listening_stats(user: dict = Depends(get_current_user), s: Services = Depends(get_services)):
    """Aggregation pipelines over the history collection: top genres and most-played tracks."""
    plays_per_song = [
        {"$match": {"user_id": user["_id"]}},
        {"$group": {"_id": "$song_id", "plays": {"$sum": 1}}},
        {"$lookup": {"from": "songs", "localField": "_id", "foreignField": "_id", "as": "song"}},
        {"$unwind": "$song"},
    ]
    top_genres = plays_per_song + [
        {"$group": {"_id": "$song.genre", "plays": {"$sum": "$plays"}, "tracks": {"$sum": 1}}},
        {"$sort": {"plays": -1, "_id": 1}},
        {"$limit": 5},
    ]
    top_tracks = plays_per_song + [
        {"$sort": {"plays": -1, "song.title": 1}},
        {"$limit": 5},
        {"$project": {"_id": 0, "song_id": {"$toString": "$_id"}, "title": "$song.title", "artist": "$song.artist", "plays": 1}},
    ]
    total_plays = await s.db.listening_history.count_documents({"user_id": user["_id"]})
    genres = [{"genre": g["_id"], "plays": g["plays"], "tracks": g["tracks"]} async for g in s.db.listening_history.aggregate(top_genres)]
    tracks = [t async for t in s.db.listening_history.aggregate(top_tracks)]
    return {"total_plays": total_plays, "top_genres": genres, "top_tracks": tracks}
