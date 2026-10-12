"""Vector layer tests: no network, no services."""

import asyncio

from app.rag.query_parser import Filters, build_where, matches
from app.rag.vector_store import InMemoryIndex, VectorStore, index_name_for, metadata_matches


def make_store():
    store = VectorStore(InMemoryIndex(), namespace="t")
    store.upsert(
        ids=["a", "b", "c"],
        metadatas=[
            {"genre": "jazz", "mood": "calm", "tempo": 80.0, "energy": 0.2},
            {"genre": "rock", "mood": "intense", "tempo": 150.0, "energy": 0.9},
            {"genre": "jazz", "mood": "happy", "tempo": 120.0, "energy": 0.6},
        ],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
    )
    return store


def test_index_name_is_pinecone_safe():
    assert index_name_for("Audio_RAG", "pinecone-e5") == "audio-rag-pinecone-e5"
    long = index_name_for("x" * 60, "hash")
    assert len(long) <= 45 and not long.endswith("-")


def test_query_ranks_by_cosine_and_reports_score():
    hits = make_store().query([1.0, 0.0], 3)
    assert [h.id for h in hits] == ["a", "c", "b"]
    assert hits[0].score == 1.0 and hits[2].score == 0.0
    assert hits[0].metadata["genre"] == "jazz"


def test_filter_is_applied_inside_the_search():
    store = make_store()
    where = build_where(Filters(genres=["jazz"], numeric={"tempo": {"$gte": 100}}))
    assert [h.id for h in store.query([1.0, 0.0], 3, where)] == ["c"]
    assert store.query([1.0, 0.0], 3, {"genre": "classical"}) == []


def test_build_where_agrees_with_python_side_matches():
    f = Filters(genres=["jazz", "rock"], moods=["calm"], numeric={"energy": {"$lte": 0.5}, "tempo": {"$gte": 60}})
    where = build_where(f)
    for meta in (
        {"genre": "jazz", "mood": "calm", "energy": 0.2, "tempo": 80.0},
        {"genre": "rock", "mood": "calm", "energy": 0.9, "tempo": 80.0},
        {"genre": "jazz", "mood": "happy", "energy": 0.2, "tempo": 80.0},
        {"genre": "jazz", "mood": "calm", "energy": 0.2},  # missing tempo never matches
    ):
        assert metadata_matches(meta, where) == matches(meta, f)


def test_upsert_overwrites_delete_removes_and_count_follows():
    store = make_store()
    assert store.count() == 3
    store.upsert(["a"], [{"genre": "blues"}], [[0.0, 1.0]])
    assert store.count() == 3
    assert store.query([0.0, 1.0], 1, {"genre": "blues"})[0].id == "a"
    store.delete(["a", "missing"])
    assert store.count() == 2


def test_namespaces_are_isolated_and_empty_query_is_safe():
    index = InMemoryIndex()
    one, two = VectorStore(index, "one"), VectorStore(index, "two")
    one.upsert(["x"], [{}], [[1.0]])
    assert one.count() == 1 and two.count() == 0
    assert two.query([1.0], 5) == [] and one.query([1.0], 0) == []


def test_async_wrappers():
    async def scenario():
        store = VectorStore(InMemoryIndex(), "t")
        await store.aupsert(["a"], [{"genre": "jazz"}], [[1.0, 0.0]])
        hits = await store.aquery([1.0, 0.0], 3, {"genre": "jazz"})
        n = await store.acount()
        await store.adelete(["a"])
        return hits, n, await store.acount()

    hits, n, after = asyncio.run(scenario())
    assert [h.id for h in hits] == ["a"] and n == 1 and after == 0
