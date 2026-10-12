"""Service container: one place that owns every external connection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
from anyio import to_thread

from .cache import Cache
from .config import Settings
from .rag.embeddings import Embedder, EmbeddingService, build_embedder
from .rag.llm import LLM, build_llm
from .rag.vector_store import VectorStore


@dataclass
class Services:
    settings: Settings
    db: Any  # motor AsyncIOMotorDatabase (or a compatible mock in tests)
    cache: Cache
    embeddings: EmbeddingService
    vectors: VectorStore
    llm: LLM | None
    http: httpx.AsyncClient
    mongo_client: Any = None

    @classmethod
    async def create(
        cls,
        settings: Settings,
        *,
        db: Any = None,
        cache: Cache | None = None,
        embedder: Embedder | None = None,
        vectors: VectorStore | None = None,
        llm: LLM | None = None,
    ) -> "Services":
        mongo_client = None
        if db is None:
            from motor.motor_asyncio import AsyncIOMotorClient

            mongo_client = AsyncIOMotorClient(settings.mongo_uri, serverSelectionTimeoutMS=3000)
            db = mongo_client[settings.mongo_db]

        cache = cache or Cache(settings.redis_url)
        await cache.connect()

        embedder = embedder or await to_thread.run_sync(
            build_embedder, settings.embedding_backend, settings.embedding_dim, settings.pinecone_api_key
        )
        if vectors is None:
            vectors = await to_thread.run_sync(
                lambda: VectorStore.create(
                    backend=settings.vector_backend,
                    api_key=settings.pinecone_api_key,
                    index_base=settings.pinecone_index,
                    namespace=settings.pinecone_namespace,
                    cloud=settings.pinecone_cloud,
                    region=settings.pinecone_region,
                    dimension=embedder.dim,
                    embedder_name=embedder.name,
                )
            )

        http = httpx.AsyncClient(timeout=settings.llm_timeout_seconds)
        llm = llm if llm is not None else build_llm(settings, http)
        return cls(
            settings=settings,
            db=db,
            cache=cache,
            embeddings=EmbeddingService(embedder, cache, settings.embedding_cache_ttl),
            vectors=vectors,
            llm=llm,
            http=http,
            mongo_client=mongo_client,
        )

    async def close(self) -> None:
        await self.http.aclose()
        await self.cache.close()
        if self.mongo_client is not None:
            self.mongo_client.close()
