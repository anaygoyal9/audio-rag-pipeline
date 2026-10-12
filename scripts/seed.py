"""Seed a demo catalogue: 24 tracks with audio features, synthetic playable WAVs, demo users,
and the vector index.

    python -m scripts.seed

The audio is a short generated arpeggio per track (stdlib ``wave`` only), so streaming and
Range requests can be demonstrated without any licensed music. Safe to re-run.
"""

from __future__ import annotations

import asyncio
import math
import os
import struct
import wave
from pathlib import Path

from app.config import get_settings
from app.db import ensure_indexes
from app.rag import indexer
from app.security import hash_password
from app.services import Services
from app.utils import utcnow

# title, artist, album, genre, mood, tempo, energy, valence, danceability, year, description
CATALOG = [
    ("Midnight Library", "Paper Lanterns", "Quiet Hours", "lofi", "focused", 72, 0.25, 0.45, 0.45, 2023, "Soft dusty piano loops and vinyl crackle, ideal for studying late at night."),
    ("Rainy Window Seat", "Paper Lanterns", "Quiet Hours", "lofi", "chill", 68, 0.2, 0.4, 0.4, 2023, "Mellow beats with warm Rhodes chords and rain ambience for unwinding."),
    ("Coffee Shop Static", "Tape Hiss Club", "Slow Mornings", "lofi", "chill", 78, 0.3, 0.55, 0.5, 2022, "Laid-back jazzy lofi with a lazy drum groove, perfect background for working."),
    ("Neon Sprint", "Voltage Kid", "Overdrive", "electronic", "energetic", 140, 0.92, 0.7, 0.85, 2024, "Pounding synth bass and driving four-on-the-floor kick for a hard workout."),
    ("Pulse Engine", "Voltage Kid", "Overdrive", "electronic", "energetic", 132, 0.88, 0.65, 0.82, 2024, "Relentless arpeggiated synths and big drops, great for running and cardio."),
    ("Sunrise Protocol", "Helix Ghost", "Daybreak", "electronic", "uplifting", 124, 0.75, 0.82, 0.8, 2023, "Bright euphoric house with rising pads, an energetic morning boost."),
    ("Glass Cathedral", "Helix Ghost", "Daybreak", "ambient", "chill", 60, 0.15, 0.5, 0.2, 2022, "Slow evolving synth textures and shimmering reverb for deep relaxation and sleep."),
    ("Weightless", "Nimbus", "Float", "ambient", "focused", 55, 0.12, 0.45, 0.15, 2021, "Drifting drones and gentle pulses that fade into the background while you concentrate."),
    ("Tidal Memory", "Nimbus", "Float", "ambient", "melancholic", 50, 0.1, 0.25, 0.1, 2021, "Wistful, slow-moving soundscape evoking an empty beach at dusk."),
    ("Broken Amp", "The Static Choir", "Loud Enough", "rock", "energetic", 148, 0.9, 0.55, 0.6, 2020, "Crunchy distorted guitars and pounding drums, raw garage rock energy."),
    ("Highway Lights", "The Static Choir", "Loud Enough", "rock", "uplifting", 128, 0.78, 0.75, 0.65, 2020, "Anthemic road-trip rock with soaring guitar hooks and shout-along chorus."),
    ("Slow Burn", "Marlow & Finch", "Embers", "rock", "dark", 90, 0.6, 0.2, 0.4, 2019, "Brooding bluesy riffs and smoky vocals with a heavy, dark atmosphere."),
    ("Sundress Season", "Coral Avenue", "Sugar Rush", "pop", "happy", 118, 0.7, 0.92, 0.82, 2024, "Sparkling summery pop with handclaps and a cheerful singalong melody."),
    ("Last Text", "Coral Avenue", "Sugar Rush", "pop", "melancholic", 84, 0.45, 0.25, 0.55, 2024, "Heartbreak ballad over soft synths about a message that never got a reply."),
    ("Golden Hour Drive", "June Parade", "Postcards", "pop", "romantic", 100, 0.55, 0.8, 0.7, 2023, "Warm dreamy pop for a sunset drive with someone special."),
    ("Blue Room Sessions", "Theo Marsh Quartet", "After Hours", "jazz", "romantic", 76, 0.3, 0.6, 0.45, 2018, "Smoky late-night jazz with muted trumpet and brushed drums, intimate and romantic."),
    ("Walking Bass Weekend", "Theo Marsh Quartet", "After Hours", "jazz", "happy", 138, 0.55, 0.85, 0.7, 2018, "Swinging upright bass and sparkling piano, cheerful and playful."),
    ("Nocturne in Rain", "Elise Varga", "Etudes", "classical", "melancholic", 58, 0.18, 0.3, 0.15, 2017, "Delicate solo piano, slow and wistful, full of longing."),
    ("Morning Strings", "Elise Varga", "Etudes", "classical", "focused", 80, 0.3, 0.6, 0.25, 2017, "Gentle chamber strings with a steady pulse that supports deep concentration."),
    ("Concrete Crown", "Rook & Ember", "Block Party", "hip-hop", "energetic", 96, 0.8, 0.6, 0.85, 2022, "Heavy 808s and confident bars, hard-hitting boom bap for the gym."),
    ("Late Night Verses", "Rook & Ember", "Block Party", "hip-hop", "chill", 82, 0.4, 0.5, 0.7, 2022, "Smooth jazzy hip-hop beat with laid-back rhymes for a night drive."),
    ("Porch Light", "Willow Creek", "Hand Me Down", "acoustic", "romantic", 88, 0.3, 0.7, 0.4, 2021, "Fingerpicked guitar and warm harmonies, an intimate acoustic love song."),
    ("Dust Road", "Willow Creek", "Hand Me Down", "acoustic", "melancholic", 70, 0.28, 0.3, 0.3, 2021, "Sparse acoustic guitar and aching vocals, a quiet song about leaving home."),
    ("Bedroom Festival", "Pale Mauve", "Small Rooms", "indie", "happy", 122, 0.68, 0.78, 0.72, 2023, "Jangly guitars, bouncy bass and hand-claps, an upbeat indie dance-around-the-room track."),
]  # fmt: skip

SCALE = [220.0, 246.94, 277.18, 293.66, 329.63, 369.99, 415.30, 440.0]  # A major
RATE = 16000


def write_demo_wav(path: Path, seed: int, seconds: float = 6.0) -> int:
    """A gentle arpeggio, different per track, as a 16-bit mono WAV. Returns duration seconds."""
    frames = bytearray()
    note_len = int(RATE * 0.25)
    total = int(RATE * seconds)
    for i in range(total):
        note = (i // note_len + seed) % len(SCALE)
        freq = SCALE[note] * (1 if (seed + i // (note_len * 8)) % 2 == 0 else 0.75)
        t = (i % note_len) / RATE
        env = min(1.0, (i % note_len) / (RATE * 0.01)) * math.exp(-4.0 * t)
        sample = 0.35 * env * math.sin(2 * math.pi * freq * (i / RATE))
        frames += struct.pack("<h", int(sample * 32767))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(bytes(frames))
    return int(seconds)


async def main() -> None:
    settings = get_settings()
    services = await Services.create(settings)
    await ensure_indexes(services.db)
    audio_dir = Path(settings.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    songs = []
    for n, (title, artist, album, genre, mood, tempo, energy, valence, dance, year, desc) in enumerate(CATALOG):
        now = utcnow()
        fields = {
            "title": title, "artist": artist, "album": album, "genre": genre, "mood": mood,
            "tempo": float(tempo), "energy": energy, "valence": valence, "danceability": dance,
            "year": year, "description": desc, "tags": [genre, mood], "updated_at": now,
        }  # fmt: skip
        doc = await services.db.songs.find_one_and_update(
            {"artist": artist, "title": title},
            {"$set": fields, "$setOnInsert": {"created_at": now, "play_count": 0}},
            upsert=True,
            return_document=True,
        )
        filename = f"{doc['_id']}.wav"
        duration = write_demo_wav(audio_dir / filename, seed=n)
        doc = await services.db.songs.find_one_and_update(
            {"_id": doc["_id"]},
            {"$set": {
                "audio_path": filename, "audio_content_type": "audio/wav",
                "audio_size": os.path.getsize(audio_dir / filename), "duration_sec": float(duration),
            }},
            return_document=True,
        )  # fmt: skip
        songs.append(doc)

    await indexer.index_songs(services, songs)

    demo_password = os.environ.get("SEED_PASSWORD", "DemoPass123!")
    for email, role, name in (("admin@example.com", "admin", "Demo Admin"), ("demo@example.com", "user", "Demo User")):
        now = utcnow()
        await services.db.users.update_one(
            {"email": email},
            {
                "$set": {"role": role, "display_name": name, "is_active": True, "updated_at": now},
                "$setOnInsert": {
                    "password_hash": await hash_password(demo_password, settings.bcrypt_rounds),
                    "created_at": now,
                },
            },
            upsert=True,
        )

    print(f"Seeded {len(songs)} songs, built the vector index ({services.embeddings.embedder.name}), created demo users.")
    print(f"  admin@example.com / {demo_password}   (role: admin)")
    print(f"  demo@example.com  / {demo_password}   (role: user)")
    print("Demo credentials are for local development only; change or delete them anywhere else.")
    await services.close()


if __name__ == "__main__":
    asyncio.run(main())
