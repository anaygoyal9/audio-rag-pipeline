"""Thin Pinecone wrapper. Vectors are supplied by us (so queries can be cached in Redis).

Pinecone applies metadata filters *during* the ANN search (single-stage filtering), so a ``filter``
on genre / mood / tempo ranges narrows the candidate set before similarity ranking.

``VECTOR_BACKEND=memory`` swaps in :class:`InMemoryIndex`, a brute-force cosine index with the same
interface, for offline development and tests (no API key, nothing persisted).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from functools import partial
from types import SimpleNamespace
from typing import Any

UPSERT_BATCH = 100
DELETE_BATCH = 1000


@dataclass
class Hit:
    id: str
    score: float  # cosine similarity in [-1, 1]; higher is closer
    metadata: dict[str, Any] = field(default_factory=dict)


def index_name_for(base: str, embedder_name: str) -> str:
    """Pinecone index names: lowercase letters, digits, hyphens, max 45 chars.
    One index per embedding backend, so vectors from different models (and dimensions) never mix."""
    name = re.sub(r"[^a-z0-9-]+", "-", f"{base}-{embedder_name}".lower()).strip("-")
    return name[:45].rstrip("-")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from either an SDK response object or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class VectorStore:
    def __init__(self, index: Any, namespace: str = "songs") -> None:
        self._index = index
        self._ns = namespace

    @classmethod
    def create(
        cls,
        *,
        backend: str,
        api_key: str | None,
        index_base: str,
        namespace: str,
        cloud: str,
        region: str,
        dimension: int,
        embedder_name: str,
    ) -> "VectorStore":
        if backend == "memory":
            return cls(InMemoryIndex(), namespace)
        if backend != "pinecone":
            raise ValueError(f"unknown VECTOR_BACKEND '{backend}' (use 'pinecone' or 'memory')")
        if not api_key:
            raise RuntimeError(
                "PINECONE_API_KEY is not set. Create a free key at https://app.pinecone.io, "
                "or set VECTOR_BACKEND=memory for offline development."
            )

        from pinecone import Pinecone, ServerlessSpec

        pc = Pinecone(api_key=api_key)
        name = index_name_for(index_base, embedder_name)
        if not pc.has_index(name):
            pc.create_index(
                name=name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )  # blocks until the index is ready
        desc = pc.describe_index(name)
        if int(_get(desc, "dimension")) != dimension:
            raise RuntimeError(
                f"Pinecone index '{name}' has dimension {_get(desc, 'dimension')} but the "
                f"'{embedder_name}' embedder produces {dimension}. Use a different PINECONE_INDEX name."
            )
        return cls(pc.Index(host=_get(desc, "host")), namespace)

    # ---- sync primitives (run in a worker thread by the async wrappers) ----------------
    def count(self) -> int:
        stats = self._index.describe_index_stats()
        entry = (_get(stats, "namespaces") or {}).get(self._ns)
        return int(_get(entry, "vector_count", 0)) if entry is not None else 0

    def upsert(self, ids: list[str], metadatas: list[dict], embeddings: list[list[float]]) -> None:
        records = [{"id": i, "values": v, "metadata": m} for i, v, m in zip(ids, embeddings, metadatas)]
        for start in range(0, len(records), UPSERT_BATCH):
            self._index.upsert(vectors=records[start:start + UPSERT_BATCH], namespace=self._ns)

    def delete(self, ids: list[str]) -> None:
        for start in range(0, len(ids), DELETE_BATCH):
            self._index.delete(ids=ids[start:start + DELETE_BATCH], namespace=self._ns)

    def query(self, embedding: list[float], n: int, where: dict | None = None) -> list[Hit]:
        if n <= 0:
            return []
        res = self._index.query(
            vector=embedding,
            top_k=min(n, 1000),  # Pinecone caps top_k at 1000 when metadata is included
            filter=where or None,
            include_metadata=True,
            namespace=self._ns,
        )
        return [
            Hit(id=_get(m, "id"), score=round(float(_get(m, "score")), 4), metadata=dict(_get(m, "metadata") or {}))
            for m in _get(res, "matches", [])
        ]

    # ---- async wrappers ----------------------------------------------------------------
    async def aquery(self, embedding: list[float], n: int, where: dict | None = None) -> list[Hit]:
        from anyio import to_thread

        return await to_thread.run_sync(partial(self.query, embedding, n, where))

    async def aupsert(self, ids, metadatas, embeddings) -> None:  # noqa: ANN001
        from anyio import to_thread

        await to_thread.run_sync(partial(self.upsert, ids, metadatas, embeddings))

    async def adelete(self, ids: list[str]) -> None:
        from anyio import to_thread

        await to_thread.run_sync(self.delete, ids)

    async def acount(self) -> int:
        from anyio import to_thread

        return await to_thread.run_sync(self.count)


# ------------------------------------------------------------------------------------------
# In-memory stand-in with Pinecone's call shape and filter operators.
# ------------------------------------------------------------------------------------------
def _cond_ok(value: Any, cond: Any) -> bool:
    if not isinstance(cond, dict):
        return value == cond  # bare value means $eq
    for op, target in cond.items():
        if op == "$eq" and not value == target:
            return False
        if op == "$ne" and not value != target:
            return False
        if op == "$in" and value not in target:
            return False
        if op == "$nin" and value in target:
            return False
        if op in ("$gt", "$gte", "$lt", "$lte"):
            if value is None:
                return False
            if (op == "$gt" and not value > target) or (op == "$gte" and not value >= target):
                return False
            if (op == "$lt" and not value < target) or (op == "$lte" and not value <= target):
                return False
    return True


def metadata_matches(metadata: dict, flt: dict | None) -> bool:
    if not flt:
        return True
    for key, cond in flt.items():
        if key == "$and":
            if not all(metadata_matches(metadata, sub) for sub in cond):
                return False
        elif key == "$or":
            if not any(metadata_matches(metadata, sub) for sub in cond):
                return False
        elif key not in metadata or not _cond_ok(metadata[key], cond):
            return False
    return True


class InMemoryIndex:
    def __init__(self) -> None:
        self._data: dict[str, dict[str, tuple[list[float], dict]]] = {}

    def upsert(self, vectors: list[dict], namespace: str = "") -> None:
        ns = self._data.setdefault(namespace, {})
        for v in vectors:
            ns[v["id"]] = (list(v["values"]), dict(v.get("metadata") or {}))

    def delete(self, ids: list[str], namespace: str = "") -> None:
        for i in ids:
            self._data.get(namespace, {}).pop(i, None)

    def query(self, vector, top_k, filter=None, include_metadata=True, namespace=""):  # noqa: A002
        qn = math.sqrt(sum(x * x for x in vector)) or 1.0
        scored = []
        for vid, (values, meta) in self._data.get(namespace, {}).items():
            if not metadata_matches(meta, filter):
                continue
            vn = math.sqrt(sum(x * x for x in values)) or 1.0
            scored.append((sum(a * b for a, b in zip(vector, values)) / (qn * vn), vid, meta))
        scored.sort(key=lambda t: t[0], reverse=True)
        matches = [SimpleNamespace(id=vid, score=s, metadata=meta if include_metadata else None)
                   for s, vid, meta in scored[:top_k]]
        return SimpleNamespace(matches=matches)

    def describe_index_stats(self) -> dict:
        namespaces = {ns: {"vector_count": len(rows)} for ns, rows in self._data.items() if rows}
        return {"namespaces": namespaces, "total_vector_count": sum(len(r) for r in self._data.values())}
