"""MongoDB index definitions. Idempotent: safe to run on every startup."""

from __future__ import annotations

import logging
from typing import Any

from pymongo import ASCENDING, DESCENDING, IndexModel

log = logging.getLogger("app.db")


async def ensure_basic_indexes(db: Any) -> None:
    """Unique and non-text indexes — safe for both real MongoDB and mongomock.

    users             unique email lookup on login
    songs             catalogue browse (genre+mood, tempo range), popularity sort,
                      duplicate guard on (artist, title)
    playlists         "my playlists" (owner, newest first) and public discovery
    listening_history per-user timeline and per-song aggregation for stats
    """
    await db.users.create_indexes([IndexModel([("email", ASCENDING)], unique=True, name="uniq_email")])
    await db.songs.create_indexes(
        [
            IndexModel([("artist", ASCENDING), ("title", ASCENDING)], unique=True, name="uniq_artist_title"),
            IndexModel([("genre", ASCENDING), ("mood", ASCENDING)], name="genre_mood"),
            IndexModel([("tempo", ASCENDING)], name="tempo"),
            IndexModel([("play_count", DESCENDING)], name="popularity"),
            IndexModel([("created_at", DESCENDING)], name="created_at_desc"),
        ]
    )
    await db.playlists.create_indexes(
        [
            IndexModel([("owner_id", ASCENDING), ("updated_at", DESCENDING)], name="owner_recent"),
            IndexModel([("is_public", ASCENDING), ("updated_at", DESCENDING)], name="public_recent"),
        ]
    )
    await db.listening_history.create_indexes(
        [
            IndexModel([("user_id", ASCENDING), ("played_at", DESCENDING)], name="user_timeline"),
            IndexModel([("song_id", ASCENDING)], name="song"),
        ]
    )


async def ensure_text_indexes(db: Any) -> None:
    """Full-text search index — requires real MongoDB (not supported by mongomock)."""
    await db.songs.create_indexes(
        [
            IndexModel(
                [("title", "text"), ("artist", "text"), ("album", "text"), ("tags", "text")],
                weights={"title": 10, "artist": 6, "album": 3, "tags": 2},
                name="song_text",
            ),
        ]
    )


async def ensure_indexes(db: Any) -> None:
    """Create all indexes (basic + text). Use on real MongoDB only."""
    await ensure_basic_indexes(db)
    await ensure_text_indexes(db)
    log.info("MongoDB indexes ensured")
