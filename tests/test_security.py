import asyncio
import time

import pytest

pytest.importorskip("bcrypt")
jwt = pytest.importorskip("jwt")
pytest.importorskip("pydantic_settings")
pytest.importorskip("anyio")

from app.config import Settings  # noqa: E402
from app.security import create_access_token, decode_access_token, hash_password, verify_password  # noqa: E402


def test_password_roundtrip_and_long_passwords():
    async def scenario():
        h = await hash_password("correct horse battery staple", rounds=4)
        long_a = "a" * 100 + "X"
        long_b = "a" * 100 + "Y"  # differs only after byte 72: plain bcrypt would treat these as equal
        hl = await hash_password(long_a, rounds=4)
        return (
            await verify_password("correct horse battery staple", h),
            await verify_password("wrong", h),
            await verify_password(long_a, hl),
            await verify_password(long_b, hl),
            await verify_password("anything", None, rounds=4),
        )

    ok, bad, long_ok, long_bad, unknown_user = asyncio.run(scenario())
    assert ok and not bad
    assert long_ok and not long_bad
    assert unknown_user is False


def test_jwt_roundtrip_expiry_and_tampering():
    settings = Settings(jwt_secret="x" * 32, access_token_minutes=1)
    token, ttl = create_access_token(settings, "abc123", "user")
    assert ttl == 60
    claims = decode_access_token(settings, token)
    assert claims["sub"] == "abc123" and claims["role"] == "user"

    with pytest.raises(jwt.PyJWTError):
        decode_access_token(Settings(jwt_secret="y" * 32), token)  # wrong key

    expired = jwt.encode({"sub": "abc", "exp": int(time.time()) - 5}, settings.jwt_secret, algorithm="HS256")
    with pytest.raises(jwt.ExpiredSignatureError):
        decode_access_token(settings, expired)

    none_alg = jwt.encode({"sub": "abc", "exp": int(time.time()) + 60}, key=None, algorithm="none")
    with pytest.raises(jwt.PyJWTError):
        decode_access_token(settings, none_alg)  # "alg: none" must be rejected


def test_default_secret_rejected_outside_dev():
    with pytest.raises(ValueError):
        Settings(environment="production")
    Settings(environment="production", jwt_secret="z" * 32)
