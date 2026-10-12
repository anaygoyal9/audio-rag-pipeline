"""The retrieval pipeline: parse -> embed (cached) -> filtered vector search -> hydrate."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field

from bson import ObjectId

from ..services import Services
from .embeddings import normalize_query
from .query_parser import Filters, ParsedQuery, build_where, llm_parse, matches, rule_based_parse
from .vector_store import Hit

log = logging.getLogger("app.rag")

MIN_HITS = 5  # below this many results, relax the filters and try again
POST_FILTER_OVERFETCH = 10


@dataclass
class Outcome:
    parsed: ParsedQuery
    hits: list[Hit]
    relaxation_level: int
    embedding_cache_hit: bool
    timings_ms: dict[str, float] = field(default_factory=dict)


async def get_vocab(services: Services) -> dict[str, list[str]]:
    cached = await services.cache.get_json("vocab:v1")
    if cached:
        return cached
    genres = sorted(g for g in await services.db.songs.distinct("genre") if g)
    moods = sorted(m for m in await services.db.songs.distinct("mood") if m)
    vocab = {"genres": genres, "moods": moods}
    await services.cache.set_json("vocab:v1", vocab, 300)
    return vocab


async def parse_query(services: Services, text: str, vocab: dict[str, list[str]]) -> ParsedQuery:
    key_material = json.dumps([normalize_query(text), vocab, services.llm is not None], sort_keys=True)
    key = "parse:" + hashlib.sha256(key_material.encode()).hexdigest()
    cached = await services.cache.get_json(key)
    if cached:
        return ParsedQuery.from_dict(cached)

    parsed: ParsedQuery | None = None
    if services.llm is not None:
        try:
            parsed = await llm_parse(services.llm, text, vocab)
        except Exception as exc:  # noqa: BLE001 - any LLM/network/JSON failure falls back to rules
            log.warning("LLM query parsing failed (%s); using rule-based parser", exc)
    if parsed is None:
        parsed = rule_based_parse(text, vocab)
    await services.cache.set_json(key, parsed.to_dict(), 3600)
    return parsed


async def run_discovery(
    services: Services,
    text: str,
    limit: int,
    *,
    use_cache: bool = True,
    filter_mode: str = "pre",  # "pre" (Pinecone filter), "post" (overfetch then filter in Python), "none"
) -> Outcome:
    """``use_cache`` and ``filter_mode`` exist so scripts/benchmark.py can measure each optimisation
    in isolation; the API always runs the defaults."""
    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    t = time.perf_counter()
    vocab = await get_vocab(services)
    parsed = await parse_query(services, text, vocab)
    timings["parse"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    vector, cache_hit = await services.embeddings.embed_query(parsed.semantic_text, use_cache=use_cache)
    timings["embed"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    best: tuple[int, list[Hit]] | None = None
    if filter_mode == "none":
        ladder = [(0, Filters())]
    else:
        ladder = parsed.filters.relaxation_ladder()
    for level, filters in ladder:
        if filter_mode == "post":
            raw = await services.vectors.aquery(vector, limit * POST_FILTER_OVERFETCH, None)
            hits = [h for h in raw if matches(h.metadata, filters)][:limit]
        else:
            hits = await services.vectors.aquery(vector, limit, build_where(filters))
        if best is None or len(hits) > len(best[1]):
            best = (level, hits)
        if len(hits) >= min(limit, MIN_HITS):
            best = (level, hits)
            break
    timings["vector_search"] = (time.perf_counter() - t) * 1000
    timings["total"] = (time.perf_counter() - t_start) * 1000

    level, hits = best if best else (0, [])
    return Outcome(
        parsed=parsed,
        hits=hits,
        relaxation_level=level,
        embedding_cache_hit=cache_hit,
        timings_ms={k: round(v, 2) for k, v in timings.items()},
    )


async def hydrate(services: Services, hits: list[Hit]) -> list[tuple[Hit, dict]]:
    """Load full song documents for the hits, preserving rank order and dropping any vector
    whose Mongo document has since been deleted."""
    ids = []
    for h in hits:
        try:
            ids.append(ObjectId(h.id))
        except Exception:  # noqa: BLE001
            continue
    docs = {str(d["_id"]): d async for d in services.db.songs.find({"_id": {"$in": ids}})}
    return [(h, docs[h.id]) for h in hits if h.id in docs]
