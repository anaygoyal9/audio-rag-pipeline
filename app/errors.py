"""Uniform JSON error envelope and global exception interceptors.

Every error, whether raised by us, by FastAPI validation, by Starlette routing, by MongoDB,
or by an unexpected bug, leaves the API as::

    {"error": {"code": "...", "message": "...", "request_id": "...", "details": ...}}
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pymongo.errors import DuplicateKeyError, PyMongoError
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("app.errors")


class AppError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details
        self.headers = headers


def error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: Any = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "request_id": getattr(request.state, "request_id", None),
        }
    }
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(body, status_code=status_code, headers=headers)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError):
        return error_response(request, exc.status_code, exc.code, exc.message, exc.details, exc.headers)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        message = exc.detail if isinstance(exc.detail, str) else "HTTP error"
        return error_response(
            request, exc.status_code, f"http_{exc.status_code}", message, headers=getattr(exc, "headers", None)
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        details = [
            {"loc": [str(p) for p in e.get("loc", ())], "message": e.get("msg", ""), "type": e.get("type", "")}
            for e in exc.errors()
        ]
        return error_response(request, 422, "validation_error", "Request validation failed", details)

    @app.exception_handler(DuplicateKeyError)
    async def _duplicate(request: Request, exc: DuplicateKeyError):
        return error_response(request, 409, "conflict", "A resource with these unique values already exists")

    @app.exception_handler(PyMongoError)
    async def _mongo(request: Request, exc: PyMongoError):
        log.error("database error on %s %s: %s", request.method, request.url.path, exc)
        return error_response(request, 503, "database_unavailable", "The database is temporarily unavailable")

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        # Never leak internals to the client; the request id ties this to the server log.
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        return error_response(request, 500, "internal_error", "An unexpected error occurred")
