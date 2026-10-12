from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import Query

from .errors import AppError


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def oid(value: str, name: str = "id") -> ObjectId:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        raise AppError(422, "invalid_id", f"'{name}' is not a valid identifier") from None


@dataclass
class Pagination:
    page: int
    limit: int

    @property
    def skip(self) -> int:
        return (self.page - 1) * self.limit


def pagination(page: int = Query(1, ge=1), limit: int = Query(20, ge=1, le=100)) -> Pagination:
    return Pagination(page=page, limit=limit)


def page_envelope(items: list, total: int, pg: Pagination) -> dict:
    return {
        "items": items,
        "total": total,
        "page": pg.page,
        "limit": pg.limit,
        "pages": (total + pg.limit - 1) // pg.limit if total else 0,
    }
