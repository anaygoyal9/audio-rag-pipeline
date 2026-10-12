from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Settings, get_settings
from .db import ensure_indexes
from .errors import register_exception_handlers
from .middleware import RateLimitMiddleware, RequestContextMiddleware
from .routers import auth, discover, history, playlists, songs, stream
from .services import Services

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


def create_app(
    settings: Settings | None = None,
    services_factory: Callable[[], Awaitable[Services]] | None = None,
    *,
    init_indexes: bool = True,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        services = await (services_factory() if services_factory else Services.create(settings))
        if init_indexes:
            await ensure_indexes(services.db)
        app.state.services = services
        if settings.vector_backend == "memory":  # in-memory index starts empty on every boot
            from .rag.indexer import reindex_all

            await reindex_all(services)
        try:
            yield
        finally:
            await services.close()

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description=(
            "Natural-language music discovery (RAG over Pinecone) with JWT-secured REST APIs, "
            "HTTP Range audio streaming, Redis caching and rate limiting."
        ),
        lifespan=lifespan,
    )

    # add_middleware wraps outward: the last one added is the outermost layer.
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Range", "Accept-Ranges", "Content-Length", "X-Request-ID", "Retry-After"],
    )
    register_exception_handlers(app)

    for module in (auth, songs, playlists, history, stream, discover):
        app.include_router(module.router)

    @app.get("/healthz", tags=["ops"])
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"])
    async def readyz(request: Request):
        s: Services = request.app.state.services
        checks: dict[str, str] = {}
        try:
            await s.db.command("ping")
            checks["mongodb"] = "ok"
        except Exception:  # noqa: BLE001
            checks["mongodb"] = "down"
        try:
            checks["vector_store"] = f"ok ({await s.vectors.acount()} vectors)"
        except Exception:  # noqa: BLE001
            checks["vector_store"] = "down"
        checks["redis"] = "ok" if await s.cache.ping() else "degraded (in-process fallback)"
        ready = checks["mongodb"] == "ok" and checks["vector_store"].startswith("ok")
        return JSONResponse({"ready": ready, "checks": checks}, status_code=200 if ready else 503)

    return app


app = create_app()
