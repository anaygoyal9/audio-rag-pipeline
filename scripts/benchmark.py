"""Measure the two retrieval optimisations on YOUR data, instead of trusting a number.

    python -m scripts.benchmark [--rounds 20]

Runs the retrieval pipeline in-process (LLM disabled so network latency does not pollute the
numbers) against whatever MONGO_URI / PINECONE_INDEX / REDIS_URL / EMBEDDING_BACKEND point at, and
prints p50/p95 end-to-end latency for each variant:

  baseline          no embedding cache, filters applied in Python after an over-fetched search
  + embedding cache embedding served from Redis on repeat queries
  + pre-filtering   metadata filter pushed into the vector search (Pinecone `filter`)

Seed first (`python -m scripts.seed`). With a 24-song catalogue the vector search is trivially
fast, so expect the cache (which skips the embedding model) to dominate; filter pushdown only
shows a meaningful gap on larger collections. Use a bigger catalogue for a realistic result.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

from app.config import get_settings
from app.rag.service import run_discovery
from app.services import Services

QUERIES = [
    "upbeat electronic music for a morning workout",
    "calm piano for studying late at night",
    "sad acoustic songs for a rainy day",
    "energetic rock for a road trip",
    "chill lofi beats to relax",
    "romantic jazz for a quiet dinner",
    "happy pop for a summer party",
    "dark moody tracks for deep focus",
]


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))]


async def measure(services: Services, rounds: int, **kwargs) -> list[float]:
    samples: list[float] = []
    for _ in range(rounds):
        for q in QUERIES:
            t = time.perf_counter()
            await run_discovery(services, q, 15, **kwargs)
            samples.append((time.perf_counter() - t) * 1000)
    return samples


async def main(rounds: int) -> None:
    settings = get_settings()
    settings.llm_provider = "none"
    services = await Services.create(settings)
    services.llm = None
    if await services.vectors.acount() == 0:
        raise SystemExit("Vector index is empty. Run `python -m scripts.seed` first.")

    for q in QUERIES:  # warm: model load, parse cache, vocab cache, caches for the cached variants
        await run_discovery(services, q, 15)

    variants = [
        ("baseline (no cache, post-filter)", dict(use_cache=False, filter_mode="post")),
        ("+ embedding cache", dict(use_cache=True, filter_mode="post")),
        ("+ pre-filtering (full pipeline)", dict(use_cache=True, filter_mode="pre")),
    ]
    results = []
    for name, kwargs in variants:
        samples = await measure(services, rounds, **kwargs)
        results.append((name, statistics.median(samples), pct(samples, 0.95)))

    base = results[0][1]
    print(f"\nEmbedder: {services.embeddings.embedder.name} | cache: {services.cache.backend} | "
          f"vectors: {await services.vectors.acount()} | samples per variant: {rounds * len(QUERIES)}\n")
    print(f"{'variant':36} {'p50 ms':>9} {'p95 ms':>9} {'p50 vs baseline':>17}")
    for name, p50, p95 in results:
        print(f"{name:36} {p50:9.2f} {p95:9.2f} {(1 - p50 / base) * 100:16.1f}%")
    await services.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=20)
    asyncio.run(main(ap.parse_args().rounds))
