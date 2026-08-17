"""Tests for the public ``fetch_all`` listing API on accessors."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from serviette.config.schema import DuckDbConfig
from serviette.server.accessors.duckdb import DuckDbAccessor

DOCS = ["alpha document about cats", "beta report on dogs", "gamma notes on birds"]


def _seed(path: Path) -> None:
    """Populate a store directly, mirroring the indexer sink schema.

    Self-contained copy of the ``write_duckdb_rows`` pattern from
    ``tests/conftest.py`` — importing ``tests.conftest`` is ambiguous in the
    uv workspace (multiple ``tests`` packages).
    """

    import duckdb

    conn = duckdb.connect(str(path))
    conn.execute(
        "CREATE TABLE serviette_embeddings ("
        "  chunk_id VARCHAR PRIMARY KEY, text VARCHAR, metadata VARCHAR,"
        "  embedding DOUBLE[])"
    )
    for i, text in enumerate(DOCS):
        conn.execute(
            "INSERT INTO serviette_embeddings VALUES (?, ?, ?, ?)",
            [str(i), text, json.dumps({"path": f"/docs/{i}.txt"}), [0.1, 0.2]],
        )
    conn.close()


@pytest.fixture
def store_path(tmp_path):
    path = tmp_path / "store.duckdb"
    _seed(path)
    return path


def test_fetch_all_returns_text_and_metadata(store_path):
    accessor = DuckDbAccessor(DuckDbConfig(type="duckdb", path=str(store_path)))
    hits = asyncio.run(accessor.fetch_all())
    assert sorted(h["metadata"]["path"] for h in hits) == [
        f"/docs/{i}.txt" for i in range(len(DOCS))
    ]
    assert all("text" in h and "embedding" not in h for h in hits)


def test_fetch_all_with_embeddings(store_path):
    accessor = DuckDbAccessor(DuckDbConfig(type="duckdb", path=str(store_path)))
    hits = asyncio.run(accessor.fetch_all(with_embeddings=True))
    assert all(isinstance(h["embedding"], list) for h in hits)


def test_fetch_all_default_raises_on_plain_accessor():
    from serviette.server.accessors.abstract import AsyncVectorAccessor

    class _Minimal(AsyncVectorAccessor):
        async def retrieve(self, embedding, k):
            return []

        async def close(self):
            pass

    with pytest.raises(NotImplementedError):
        asyncio.run(_Minimal().fetch_all())
