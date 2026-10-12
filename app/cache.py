"""Redis-backed cache with an in-process fallback.

Everything here fails open: if Redis is down the API keeps working (slower, and with
per-process rate limiting) instead of returning errors.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

log = logging.getLogger("app.cache")


class Cache:
    def __init__(self, redis_url: str | None, *, redis_client: Any = None) -> None:
        self._url = redis_url
        self._redis = redis_client
        self._mem: dict[str, tuple[float, bytes]] = {}
        self._counters: dict[str, tuple[int, float]] = {}

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "memory"

    async def connect(self) -> None:
        if self._redis is None and self._url:
            try:
                from redis.asyncio import Redis

                client = Redis.from_url(self._url, decode_responses=False, socket_connect_timeout=2)
                await client.ping()
                self._redis = client
            except Exception as exc:  # noqa: BLE001
                log.warning("Redis unavailable (%s); using in-process cache", exc)
                self._redis = None

    async def close(self) -> None:
        if self._redis is not None:
            try:
                await self._redis.aclose()
            except Exception:  # noqa: BLE001
                pass

    async def ping(self) -> bool:
        if self._redis is None:
            return False
        try:
            return bool(await self._redis.ping())
        except Exception:  # noqa: BLE001
            return False

    # ---- bytes -------------------------------------------------------------------------
    async def get_bytes(self, key: str) -> bytes | None:
        if self._redis is not None:
            try:
                return await self._redis.get(key)
            except Exception as exc:  # noqa: BLE001
                log.warning("redis get failed: %s", exc)
                return None
        item = self._mem.get(key)
        if item is None:
            return None
        expires, value = item
        if expires < time.monotonic():
            self._mem.pop(key, None)
            return None
        return value

    async def set_bytes(self, key: str, value: bytes, ttl: int) -> None:
        if self._redis is not None:
            try:
                await self._redis.set(key, value, ex=ttl)
            except Exception as exc:  # noqa: BLE001
                log.warning("redis set failed: %s", exc)
            return
        if len(self._mem) > 20000:  # crude bound so the fallback cannot grow without limit
            now = time.monotonic()
            self._mem = {k: v for k, v in self._mem.items() if v[0] > now}
            if len(self._mem) > 20000:
                self._mem.clear()
        self._mem[key] = (time.monotonic() + ttl, value)

    # ---- json --------------------------------------------------------------------------
    async def get_json(self, key: str) -> Any | None:
        raw = await self.get_bytes(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return None

    async def set_json(self, key: str, value: Any, ttl: int) -> None:
        await self.set_bytes(key, json.dumps(value).encode("utf-8"), ttl)

    # ---- rate limiting -----------------------------------------------------------------
    async def incr_window(self, key: str, window: int) -> tuple[int, int]:
        """Fixed-window counter. Returns (count_in_window, seconds_until_window_resets)."""
        now = time.time()
        bucket = int(now // window)
        retry_after = max(int(window - (now % window)), 1)
        full_key = f"{key}:{bucket}"
        if self._redis is not None:
            try:
                pipe = self._redis.pipeline(transaction=True)
                pipe.incr(full_key)
                pipe.expire(full_key, window + 1)
                count, _ = await pipe.execute()
                return int(count), retry_after
            except Exception as exc:  # noqa: BLE001
                log.warning("redis rate-limit failed, using local counter: %s", exc)
        count, expires = self._counters.get(full_key, (0, now + window + 1))
        if len(self._counters) > 50000:
            self._counters = {k: v for k, v in self._counters.items() if v[1] > now}
        self._counters[full_key] = (count + 1, expires)
        return count + 1, retry_after
