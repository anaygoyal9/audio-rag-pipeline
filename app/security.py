"""Password hashing (bcrypt) and JWT helpers.

bcrypt only looks at the first 72 bytes of its input, so passwords are pre-hashed with
SHA-256 (base64 encoded) first. This keeps long passphrases fully significant and avoids
the ValueError newer bcrypt releases raise for >72 byte inputs.
"""

from __future__ import annotations

import base64
import hashlib
import time
import uuid

import bcrypt
import jwt
from anyio import to_thread

from .config import Settings

_dummy_hash: str | None = None


def _prehash(password: str) -> bytes:
    return base64.b64encode(hashlib.sha256(password.encode("utf-8")).digest())


def _hash(password: str, rounds: int) -> str:
    return bcrypt.hashpw(_prehash(password), bcrypt.gensalt(rounds)).decode("ascii")


def _verify(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_prehash(password), hashed.encode("ascii"))
    except ValueError:
        return False


async def hash_password(password: str, rounds: int = 12) -> str:
    # bcrypt is deliberately slow and CPU-bound: keep it off the event loop.
    return await to_thread.run_sync(_hash, password, rounds)


async def verify_password(password: str, hashed: str | None, rounds: int = 12) -> bool:
    """Constant-ish time check: unknown users are verified against a dummy hash so that
    response time does not reveal whether an email is registered."""
    global _dummy_hash
    if hashed is None:
        if _dummy_hash is None:
            _dummy_hash = await to_thread.run_sync(_hash, "dummy-password", rounds)
        await to_thread.run_sync(_verify, password, _dummy_hash)
        return False
    return await to_thread.run_sync(_verify, password, hashed)


def create_access_token(settings: Settings, user_id: str, role: str) -> tuple[str, int]:
    now = int(time.time())
    ttl = settings.access_token_minutes * 60
    payload = {
        "sub": user_id,
        "role": role,
        "iat": now,
        "exp": now + ttl,
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm), ttl


def decode_access_token(settings: Settings, token: str) -> dict:
    """Raises jwt.ExpiredSignatureError / jwt.PyJWTError on failure."""
    return jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[settings.jwt_algorithm],
        options={"require": ["exp", "sub"]},
    )
