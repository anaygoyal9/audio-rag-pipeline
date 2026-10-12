"""Pure-ASGI middleware (BaseHTTPMiddleware buffers/wraps streaming bodies, so we avoid it)."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

log = logging.getLogger("app.access")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
_EXEMPT_PREFIXES = ("/healthz", "/readyz", "/docs", "/redoc", "/openapi.json")


class RequestContextMiddleware:
    """Assigns a request id, times the request, and writes one structured access-log line."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        supplied = Headers(scope=scope).get("x-request-id", "")
        request_id = supplied if _REQUEST_ID_RE.match(supplied) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        status = {"code": 500}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                headers["X-Process-Time-ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            log.info(
                json.dumps(
                    {
                        "request_id": request_id,
                        "method": scope["method"],
                        "path": scope["path"],
                        "status": status["code"],
                        "ms": round((time.perf_counter() - started) * 1000, 1),
                    }
                )
            )


class RateLimitMiddleware:
    """Fixed-window rate limiter keyed by client address, backed by Redis (shared across
    workers) with an in-process fallback.

    Buckets: ``auth`` (login/register, strict brute-force guard), ``stream`` (generous, because
    a single seek can issue several Range requests) and ``api`` (everything else).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    @staticmethod
    def _bucket(path: str, multiplier_out: list[int]) -> str:
        if path.startswith("/api/v1/auth/login") or path.startswith("/api/v1/auth/register"):
            return "auth"
        if path.startswith("/api/v1/stream"):
            multiplier_out.append(10)
            return "stream"
        return "api"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        services = getattr(scope["app"].state, "services", None) if scope["type"] == "http" else None
        path = scope.get("path", "")
        if services is None or scope.get("method") == "OPTIONS" or path.startswith(_EXEMPT_PREFIXES):
            await self.app(scope, receive, send)
            return

        settings = services.settings
        mult: list[int] = []
        bucket = self._bucket(path, mult)
        if bucket == "auth":
            limit = settings.auth_rate_limit_requests
        else:
            limit = settings.rate_limit_requests * (mult[0] if mult else 1)

        headers = Headers(scope=scope)
        client = scope.get("client")
        ip = client[0] if client else "unknown"
        if settings.trust_proxy_headers and headers.get("x-forwarded-for"):
            ip = headers["x-forwarded-for"].split(",")[0].strip()

        window = settings.rate_limit_window_seconds
        count, retry_after = await services.cache.incr_window(f"rl:{bucket}:{ip}", window)
        remaining = max(limit - count, 0)

        if count > limit:
            request_id = scope.get("state", {}).get("request_id")
            response = JSONResponse(
                {
                    "error": {
                        "code": "rate_limited",
                        "message": "Too many requests. Slow down and retry shortly.",
                        "request_id": request_id,
                    }
                },
                status_code=429,
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                },
            )
            await response(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                h = MutableHeaders(scope=message)
                h["X-RateLimit-Limit"] = str(limit)
                h["X-RateLimit-Remaining"] = str(remaining)
            await send(message)

        await self.app(scope, receive, send_wrapper)
