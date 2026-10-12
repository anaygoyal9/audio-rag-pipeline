"""Keeps the Pinecone index in sync with the MongoDB song catalogue (Mongo is the source of truth)."""

from __future__ import annotations

import logging
from typing import Any

from ..services import Services

log = logging.getLogger("app.indexer")


def song_document(song: dict) -> str:
    """The text that gets embedded for a track."""
    parts = [
        f"{song['title']} by {song['artist']}.",
        f"Genre: {song['genre']}. Mood: {song['mood']}.",
    ]
    if song.get("album"):
        parts.append(f"Album: {song['album']}.")
    if song.get("description"):
        parts.append(song["description"])
    if song.get("tags"):
        parts.append("Tags: " + ", ".join(song["tags"]) + ".")
    return " ".join(parts)


def song_metadata(song: dict) -> dict[str, Any]:
    """Scalar metadata stored next to the vector; this is what ``where`` filters run against."""
    meta = {
        "genre": song["genre"],
        "mood": song["mood"],
        "artist": song["artist"],
        "tempo": float(song["tempo"]),
        "energy": float(song["energy"]),
        "valence": float(song["valence"]),
        "danceability": float(song["danceability"]),
    }
    if song.get("year") is not None:
        meta["year"] = int(song["year"])
    return meta


async def index_songs(services: Services, songs: list[dict]) -> None:
    if not songs:
        return
    docs = [song_document(s) for s in songs]
    vectors = await services.embeddings.embed_documents(docs)  # the text is embedded, not stored
    await services.vectors.aupsert(
        ids=[str(s["_id"]) for s in songs],
        metadatas=[song_metadata(s) for s in songs],
        embeddings=vectors,
    )


async def remove_song(services: Services, song_id: str) -> None:
    await services.vectors.adelete([song_id])


async def reindex_all(services: Services, batch_size: int = 64) -> int:
    total = 0
    batch: list[dict] = []
    async for song in services.db.songs.find({}):
        batch.append(song)
        if len(batch) >= batch_size:
            await index_songs(services, batch)
            total += len(batch)
            batch = []
    if batch:
        await index_songs(services, batch)
        total += len(batch)
    return total
