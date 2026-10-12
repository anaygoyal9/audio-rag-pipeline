from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from ..deps import get_current_user, get_services, require_admin
from ..rag import indexer
from ..rag.service import hydrate, run_discovery
from ..schemas import DiscoverItem, DiscoverRequest, DiscoverResponse, serialize_playlist, serialize_song
from ..services import Services
from ..utils import utcnow

router = APIRouter(prefix="/api/v1/discover", tags=["RAG discovery"])


@router.post("", response_model=DiscoverResponse)
async def discover(body: DiscoverRequest, user: dict = Depends(get_current_user), s: Services = Depends(get_services)):
    """Turn a natural-language request into a ranked, filtered list of tracks (optionally saved as a playlist)."""
    outcome = await run_discovery(s, body.query, body.limit)

    t = time.perf_counter()
    pairs = await hydrate(s, outcome.hits)
    outcome.timings_ms["hydrate"] = round((time.perf_counter() - t) * 1000, 2)

    playlist = None
    if body.save_as_playlist and pairs:
        now = utcnow()
        doc = {
            "owner_id": user["_id"],
            "name": body.playlist_name or outcome.parsed.title,
            "description": f"Generated from: {body.query}"[:500],
            "is_public": False,
            "source": "rag",
            "song_ids": [song["_id"] for _, song in pairs],
            "created_at": now,
            "updated_at": now,
        }
        doc["_id"] = (await s.db.playlists.insert_one(doc)).inserted_id
        playlist = serialize_playlist(doc, [serialize_song(song) for _, song in pairs])

    return DiscoverResponse(
        query=body.query,
        interpreted={
            "semantic_text": outcome.parsed.semantic_text,
            "filters": {
                "genres": outcome.parsed.filters.genres,
                "moods": outcome.parsed.filters.moods,
                "numeric": outcome.parsed.filters.numeric,
            },
            "parser": outcome.parsed.source,
        },
        results=[DiscoverItem(score=hit.score, song=serialize_song(song)) for hit, song in pairs],
        relaxation_level=outcome.relaxation_level,
        embedding_cache_hit=outcome.embedding_cache_hit,
        timings_ms=outcome.timings_ms,
        playlist=playlist,
    )


@router.get("/stats")
async def stats(s: Services = Depends(get_services), _user: dict = Depends(get_current_user)):
    """Embedding-cache effectiveness and index size."""
    return {
        "embedding_cache": {**s.embeddings.stats.as_dict(), "backend": s.cache.backend},
        "embedder": s.embeddings.embedder.name,
        "indexed_songs": await s.vectors.acount(),
        "llm_enabled": s.llm is not None,
    }


@router.post("/reindex")
async def reindex(s: Services = Depends(get_services), _admin: dict = Depends(require_admin)):
    """Rebuild the vector index from MongoDB (e.g. after switching embedding backend)."""
    return {"reindexed": await indexer.reindex_all(s)}
