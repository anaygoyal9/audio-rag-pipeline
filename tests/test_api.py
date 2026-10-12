"""End-to-end API tests with in-memory fakes (mongomock-motor, in-process cache, in-memory vector index,
offline hash embedder). They exercise the real FastAPI app, middleware and routers."""

import io
import struct
import uuid
import wave

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mongomock_motor")
pytest.importorskip("httpx")

import httpx  # noqa: E402
from mongomock_motor import AsyncMongoMockClient  # noqa: E402

from app.cache import Cache  # noqa: E402
from app.config import Settings  # noqa: E402
from app.db import ensure_basic_indexes  # noqa: E402
from app.main import create_app  # noqa: E402
from app.rag.embeddings import HashEmbedder  # noqa: E402
from app.rag.vector_store import InMemoryIndex, VectorStore  # noqa: E402
from app.security import hash_password  # noqa: E402
from app.services import Services  # noqa: E402
from app.utils import utcnow  # noqa: E402

PASSWORD = "S3curePassw0rd!"


def make_wav(seconds=1.0, rate=8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", 1000) * int(rate * seconds))
    return buf.getvalue()


@pytest.fixture
async def client(tmp_path):
    settings = Settings(
        environment="dev",
        audio_dir=str(tmp_path / "audio"),
        bcrypt_rounds=4,
        embedding_backend="hash",
        rate_limit_requests=1000,
        auth_rate_limit_requests=1000,
        redis_url=None,
    )

    db = AsyncMongoMockClient()[f"test_{uuid.uuid4().hex}"]

    async def factory():
        return await Services.create(
            settings,
            db=db,
            cache=Cache(None),
            embedder=HashEmbedder(settings.embedding_dim),
            vectors=VectorStore(InMemoryIndex(), namespace=f"t_{uuid.uuid4().hex}"),
        )

    # init_indexes=False skips text indexes (unsupported by mongomock), but we
    # still need unique indexes (e.g. email) so we create those explicitly here.
    await ensure_basic_indexes(db)

    app = create_app(settings, factory, init_indexes=False)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            c.app = app
            yield c


async def login(client, email, password=PASSWORD) -> dict:
    r = await client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


async def make_admin(client) -> dict:
    db = client.app.state.services.db
    await db.users.insert_one(
        {
            "email": "admin@test.io",
            "password_hash": await hash_password(PASSWORD, 4),
            "role": "admin",
            "is_active": True,
            "created_at": utcnow(),
            "updated_at": utcnow(),
        }
    )
    return await login(client, "admin@test.io")


SONG = {
    "title": "Neon Sprint",
    "artist": "Voltage Kid",
    "genre": "Electronic",
    "mood": "energetic",
    "tempo": 140,
    "energy": 0.92,
    "valence": 0.7,
    "danceability": 0.85,
    "description": "Pounding synth bass for a hard workout",
}


async def test_auth_flow_and_error_envelope(client):
    r = await client.post("/api/v1/auth/register", json={"email": "A@Test.io", "password": PASSWORD})
    assert r.status_code == 201 and r.json()["email"] == "a@test.io" and r.json()["role"] == "user"
    assert "password" not in r.text

    dup = await client.post("/api/v1/auth/register", json={"email": "a@test.io", "password": PASSWORD})
    assert dup.status_code == 409 and dup.json()["error"]["code"] == "email_taken"

    bad = await client.post("/api/v1/auth/login", json={"email": "a@test.io", "password": "nope"})
    assert bad.status_code == 401 and bad.json()["error"]["code"] == "invalid_credentials"
    assert bad.json()["error"]["request_id"] == bad.headers["x-request-id"]

    headers = await login(client, "a@test.io")
    assert (await client.get("/api/v1/auth/me", headers=headers)).json()["email"] == "a@test.io"
    assert (await client.get("/api/v1/auth/me")).status_code == 401
    assert (await client.get("/api/v1/auth/me", headers={"Authorization": "Bearer junk"})).status_code == 401

    invalid = await client.post("/api/v1/auth/register", json={"email": "not-an-email", "password": "x"})
    assert invalid.status_code == 422 and invalid.json()["error"]["code"] == "validation_error"


async def test_admin_only_song_management(client):
    await client.post("/api/v1/auth/register", json={"email": "u@test.io", "password": PASSWORD})
    user = await login(client, "u@test.io")
    assert (await client.post("/api/v1/songs", json=SONG, headers=user)).status_code == 403

    admin = await make_admin(client)
    r = await client.post("/api/v1/songs", json=SONG, headers=admin)
    assert r.status_code == 201 and r.json()["genre"] == "electronic"
    assert (await client.get(f"/api/v1/songs/{r.json()['id']}", headers=user)).status_code == 200
    assert (await client.get("/api/v1/songs/not-an-id", headers=user)).status_code == 422


async def test_streaming_range_requests(client):
    admin = await make_admin(client)
    song = (await client.post("/api/v1/songs", json=SONG, headers=admin)).json()
    audio = make_wav()

    bad = await client.post(
        f"/api/v1/songs/{song['id']}/audio", headers=admin, files={"file": ("x.wav", b"not audio at all", "audio/wav")}
    )
    assert bad.status_code == 415

    up = await client.post(f"/api/v1/songs/{song['id']}/audio", headers=admin, files={"file": ("t.wav", audio, "audio/wav")})
    assert up.status_code == 200 and up.json()["has_audio"] and up.json()["audio_size"] == len(audio)

    url = f"/api/v1/stream/{song['id']}"
    full = await client.get(url, headers=admin)
    assert full.status_code == 200 and full.content == audio and full.headers["accept-ranges"] == "bytes"

    part = await client.get(url, headers={**admin, "Range": "bytes=10-19"})
    assert part.status_code == 206 and part.content == audio[10:20]
    assert part.headers["content-range"] == f"bytes 10-19/{len(audio)}" and part.headers["content-length"] == "10"

    tail = await client.get(url, headers={**admin, "Range": "bytes=-5"})
    assert tail.status_code == 206 and tail.content == audio[-5:]

    nope = await client.get(url, headers={**admin, "Range": f"bytes={len(audio) + 10}-"})
    assert nope.status_code == 416 and nope.headers["content-range"] == f"bytes */{len(audio)}"

    head = await client.head(url, headers=admin)
    assert head.status_code == 200 and head.headers["content-length"] == str(len(audio)) and head.content == b""

    assert (await client.get(url)).status_code == 401
    token = admin["Authorization"].split()[1]
    assert (await client.get(f"{url}?token={token}", headers={"Range": "bytes=0-3"})).status_code == 206


async def test_discovery_and_playlist_saving(client):
    admin = await make_admin(client)
    for song in (
        SONG,
        {**SONG, "title": "Lullaby", "artist": "Nimbus", "genre": "ambient", "mood": "chill", "tempo": 55,
         "energy": 0.1, "description": "slow ambient drone for sleep"},
    ):
        assert (await client.post("/api/v1/songs", json=song, headers=admin)).status_code == 201

    r = await client.post(
        "/api/v1/discover",
        json={"query": "electronic workout music", "limit": 5, "save_as_playlist": True},
        headers=admin,
    )
    body = r.json()
    assert r.status_code == 200
    assert body["interpreted"]["filters"]["genres"] == ["electronic"]
    assert body["results"][0]["song"]["title"] == "Neon Sprint"
    assert body["playlist"]["source"] == "rag" and body["playlist"]["song_count"] >= 1
    assert set(body["timings_ms"]) >= {"parse", "embed", "vector_search", "total", "hydrate"}

    again = await client.post("/api/v1/discover", json={"query": "electronic workout music"}, headers=admin)
    assert again.json()["embedding_cache_hit"] is True

    stats = (await client.get("/api/v1/discover/stats", headers=admin)).json()
    assert stats["embedding_cache"]["hits"] >= 1 and stats["indexed_songs"] == 2


async def test_rate_limiting_returns_429_with_retry_after(client):
    client.app.state.services.settings.auth_rate_limit_requests = 3
    codes = [
        (await client.post("/api/v1/auth/login", json={"email": "x@test.io", "password": "nope"})).status_code
        for _ in range(5)
    ]
    assert codes[:3] == [401, 401, 401] and codes[3:] == [429, 429]
    r = await client.post("/api/v1/auth/login", json={"email": "x@test.io", "password": "nope"})
    assert r.headers["retry-after"].isdigit() and r.json()["error"]["code"] == "rate_limited"


async def test_unknown_route_uses_error_envelope(client):
    r = await client.get("/api/v1/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "http_404"
