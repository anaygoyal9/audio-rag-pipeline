from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Generic, TypeVar

from pydantic import BaseModel, EmailStr, Field, field_validator

T = TypeVar("T")
Unit = Annotated[float, Field(ge=0, le=1)]


class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    limit: int
    pages: int


# ---- users -----------------------------------------------------------------------------
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str | None = Field(None, max_length=80)


class UserLogin(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserUpdate(BaseModel):
    display_name: str | None = Field(None, max_length=80)
    password: str | None = Field(None, min_length=8, max_length=128)


class UserOut(BaseModel):
    id: str
    email: EmailStr
    display_name: str | None = None
    role: str
    created_at: datetime


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


def serialize_user(doc: dict) -> UserOut:
    return UserOut(
        id=str(doc["_id"]),
        email=doc["email"],
        display_name=doc.get("display_name"),
        role=doc.get("role", "user"),
        created_at=doc["created_at"],
    )


# ---- songs -----------------------------------------------------------------------------
def _lower(v: str | None) -> str | None:
    return v.strip().lower() if isinstance(v, str) else v


class SongCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    artist: str = Field(min_length=1, max_length=200)
    album: str | None = Field(None, max_length=200)
    genre: str = Field(min_length=1, max_length=50)
    mood: str = Field(min_length=1, max_length=50)
    tempo: float = Field(ge=30, le=300, description="BPM")
    energy: Unit
    valence: Unit
    danceability: Unit
    duration_sec: float | None = Field(None, ge=0)
    year: int | None = Field(None, ge=1900, le=2100)
    description: str | None = Field(None, max_length=1000)
    tags: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("genre", "mood")
    @classmethod
    def _normalise(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("tags")
    @classmethod
    def _tags(cls, v: list[str]) -> list[str]:
        return [t.strip().lower() for t in v if t.strip()]


class SongUpdate(BaseModel):
    title: str | None = Field(None, min_length=1, max_length=200)
    artist: str | None = Field(None, min_length=1, max_length=200)
    album: str | None = Field(None, max_length=200)
    genre: str | None = Field(None, min_length=1, max_length=50)
    mood: str | None = Field(None, min_length=1, max_length=50)
    tempo: float | None = Field(None, ge=30, le=300)
    energy: Unit | None = None
    valence: Unit | None = None
    danceability: Unit | None = None
    duration_sec: float | None = Field(None, ge=0)
    year: int | None = Field(None, ge=1900, le=2100)
    description: str | None = Field(None, max_length=1000)
    tags: list[str] | None = Field(None, max_length=20)

    @field_validator("genre", "mood")
    @classmethod
    def _normalise(cls, v: str | None) -> str | None:
        return _lower(v)

    @field_validator("tags")
    @classmethod
    def _tags(cls, v: list[str] | None) -> list[str] | None:
        return [t.strip().lower() for t in v if t.strip()] if v is not None else v


class SongOut(SongCreate):
    id: str
    has_audio: bool = False
    audio_size: int | None = None
    play_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None


SONG_FIELDS = [
    "title", "artist", "album", "genre", "mood", "tempo", "energy", "valence",
    "danceability", "duration_sec", "year", "description", "tags",
]  # fmt: skip


def serialize_song(doc: dict) -> SongOut:
    data: dict[str, Any] = {k: doc.get(k) for k in SONG_FIELDS}
    data["tags"] = doc.get("tags") or []
    data.update(
        id=str(doc["_id"]),
        has_audio=bool(doc.get("audio_path")),
        audio_size=doc.get("audio_size"),
        play_count=doc.get("play_count", 0),
        created_at=doc.get("created_at"),
        updated_at=doc.get("updated_at"),
    )
    return SongOut.model_validate(data)


# ---- playlists -------------------------------------------------------------------------
class PlaylistCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str | None = Field(None, max_length=500)
    is_public: bool = False
    song_ids: list[str] = Field(default_factory=list, max_length=500)


class PlaylistUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = Field(None, max_length=500)
    is_public: bool | None = None


class PlaylistSongAdd(BaseModel):
    song_id: str


class PlaylistOut(BaseModel):
    id: str
    owner_id: str
    name: str
    description: str | None = None
    is_public: bool
    source: str = "manual"
    song_ids: list[str]
    song_count: int
    created_at: datetime
    updated_at: datetime
    songs: list[SongOut] | None = None


def serialize_playlist(doc: dict, songs: list[SongOut] | None = None) -> PlaylistOut:
    ids = [str(i) for i in doc.get("song_ids", [])]
    return PlaylistOut(
        id=str(doc["_id"]),
        owner_id=str(doc["owner_id"]),
        name=doc["name"],
        description=doc.get("description"),
        is_public=doc.get("is_public", False),
        source=doc.get("source", "manual"),
        song_ids=ids,
        song_count=len(ids),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
        songs=songs,
    )


# ---- listening history -----------------------------------------------------------------
class HistoryCreate(BaseModel):
    song_id: str
    position_sec: float = Field(0, ge=0)
    completed: bool = False


class HistoryOut(BaseModel):
    id: str
    song_id: str
    played_at: datetime
    position_sec: float = 0
    completed: bool = False
    song_title: str | None = None
    song_artist: str | None = None


# ---- discovery -------------------------------------------------------------------------
class DiscoverRequest(BaseModel):
    query: str = Field(min_length=3, max_length=300, examples=["upbeat electronic music for a morning workout"])
    limit: int = Field(15, ge=1, le=50)
    save_as_playlist: bool = False
    playlist_name: str | None = Field(None, min_length=1, max_length=100)


class DiscoverItem(BaseModel):
    score: float
    song: SongOut


class DiscoverResponse(BaseModel):
    query: str
    interpreted: dict[str, Any]
    results: list[DiscoverItem]
    relaxation_level: int = Field(description="0 = filters applied as parsed; higher = progressively relaxed")
    embedding_cache_hit: bool
    timings_ms: dict[str, float]
    playlist: PlaylistOut | None = None
