import asyncio
import math

from app.cache import Cache
from app.rag.embeddings import EmbeddingService, HashEmbedder, normalize_query


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_hash_embedder_is_deterministic_and_normalised():
    e = HashEmbedder(128)
    a, b = e.embed(["calm piano music"])[0], e.embed(["calm piano music"])[0]
    assert a == b
    assert len(a) == 128
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-9)


def test_hash_embedder_ranks_overlap_higher():
    e = HashEmbedder(384)
    q, near, far = e.embed(["energetic electronic workout", "electronic workout synths", "quiet acoustic lullaby"])
    assert cosine(q, near) > cosine(q, far)


def test_empty_text_does_not_crash():
    assert HashEmbedder(16).embed([""])[0] == [0.0] * 16


def test_normalize_query():
    assert normalize_query("  Chill   LOFI  ") == "chill lofi"


class CountingEmbedder:
    name = "counting"
    dim = 32

    def __init__(self):
        self.calls = 0
        self.inner = HashEmbedder(32)

    def embed(self, texts, kind="passage"):
        self.calls += len(texts)
        return self.inner.embed(texts)


def test_query_embedding_cache_hits_skip_the_model():
    async def scenario():
        embedder = CountingEmbedder()
        svc = EmbeddingService(embedder, Cache(None), ttl=60)
        v1, hit1 = await svc.embed_query("Chill  lofi")
        v2, hit2 = await svc.embed_query("chill lofi")  # same after normalisation
        v3, hit3 = await svc.embed_query("chill lofi", use_cache=False)
        return embedder.calls, (hit1, hit2, hit3), svc.stats.as_dict(), v1, v2, v3

    calls, hits, stats, v1, v2, v3 = asyncio.run(scenario())
    assert hits == (False, True, False)
    assert calls == 2  # the middle call never reached the embedder
    assert stats["hits"] == 1 and stats["misses"] == 2
    assert all(abs(a - b) < 1e-6 for a, b in zip(v1, v2))  # float32 round-trip through the cache
    assert v1 == v3
