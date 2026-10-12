"""Embedding backends plus a Redis-backed query-embedding cache."""

from __future__ import annotations

import hashlib
import math
import re
import time
from array import array
from dataclasses import dataclass
from typing import Protocol

from ..cache import Cache

_TOKEN_RE = re.compile(r"[a-z0-9']+")


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str], kind: str = "passage") -> list[list[float]]:
        """``kind`` is "passage" (indexed text) or "query" (a search); asymmetric models need it."""


class HashEmbedder:
    """Deterministic, dependency-free feature-hashing embedder (unigrams + bigrams).

    It captures lexical overlap only, not meaning ("chill" will not match "relaxing"), so it
    exists for tests, offline demos and benchmarks. Use the Pinecone backend for real semantic
    search.
    """

    name = "hash"

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        tokens = _TOKEN_RE.findall(text.lower())
        vec = [0.0] * self.dim
        grams = [(t, 1.0) for t in tokens] + [(f"{a}_{b}", 0.5) for a, b in zip(tokens, tokens[1:])]
        for gram, weight in grams:
            h = int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big")
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[h % self.dim] += sign * weight
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    def embed(self, texts: list[str], kind: str = "passage") -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class PineconeEmbedder:
    """Pinecone Inference (hosted multilingual-e5-large, 1024-d): no local model to download or run.

    e5 is asymmetric, so indexed tracks are embedded as "passage" and searches as "query"."""

    name = "pinecone-e5"
    dim = 1024
    model = "multilingual-e5-large"
    MAX_BATCH = 96

    def __init__(self, api_key: str | None) -> None:
        if not api_key:
            raise RuntimeError(
                "PINECONE_API_KEY is required for EMBEDDING_BACKEND=pinecone "
                "(or set EMBEDDING_BACKEND=hash for offline development)."
            )
        from pinecone import Pinecone

        self._pc = Pinecone(api_key=api_key)

    def embed(self, texts: list[str], kind: str = "passage") -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.MAX_BATCH):
            res = self._pc.inference.embed(
                model=self.model,
                inputs=texts[start:start + self.MAX_BATCH],
                parameters={"input_type": kind, "truncation": "END"},
            )
            for item in res.data:
                values = item["values"] if isinstance(item, dict) else item.values
                out.append([float(x) for x in values])
        return out


def build_embedder(backend: str, dim: int = 384, api_key: str | None = None) -> Embedder:
    if backend == "hash":
        return HashEmbedder(dim)
    if backend in ("pinecone", "pinecone-e5"):
        return PineconeEmbedder(api_key)
    raise ValueError(f"unknown EMBEDDING_BACKEND '{backend}' (use 'pinecone' or 'hash')")


def normalize_query(text: str) -> str:
    return " ".join(text.lower().split())


@dataclass
class EmbeddingStats:
    hits: int = 0
    misses: int = 0
    embed_ms_total: float = 0.0  # time spent actually computing embeddings (misses only)

    def as_dict(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "avg_miss_embed_ms": round(self.embed_ms_total / self.misses, 2) if self.misses else 0.0,
        }


class EmbeddingService:
    """Embeds queries, memoising vectors in Redis so repeated/popular searches skip the model."""

    def __init__(self, embedder: Embedder, cache: Cache, ttl: int) -> None:
        self.embedder = embedder
        self.cache = cache
        self.ttl = ttl
        self.stats = EmbeddingStats()

    def _key(self, text: str) -> str:
        digest = hashlib.sha256(normalize_query(text).encode("utf-8")).hexdigest()
        return f"emb:{self.embedder.name}:{digest}"

    async def embed_query(self, text: str, *, use_cache: bool = True) -> tuple[list[float], bool]:
        """Returns (vector, cache_hit)."""
        key = self._key(text)
        if use_cache:
            raw = await self.cache.get_bytes(key)
            if raw:
                arr = array("f")
                arr.frombytes(raw)
                self.stats.hits += 1
                return arr.tolist(), True
        from anyio import to_thread

        started = time.perf_counter()
        vector = (await to_thread.run_sync(self.embedder.embed, [text], "query"))[0]
        self.stats.misses += 1
        self.stats.embed_ms_total += (time.perf_counter() - started) * 1000
        if use_cache:
            await self.cache.set_bytes(key, array("f", vector).tobytes(), self.ttl)
        return vector, False

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from anyio import to_thread

        return await to_thread.run_sync(self.embedder.embed, texts, "passage")
