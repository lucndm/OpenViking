# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Full-text, hybrid, and grep-integration tests for the LanceDB backend.

Covers opt-in content persistence, BM25 keyword search with filters and
pagination, update/delete visibility in FTS, restart reuse, multilingual
behavior (English / Vietnamese diacritics / CJK), dense+lexical hybrid
fusion, and the grep engine capability gate.  Skipped when the optional
``lancedb`` package is missing.
"""

import json
import os
import shutil
import tempfile
import unittest

import pytest

from openviking.storage.collection_schemas import CollectionSchemas
from openviking.storage.expr import Eq
from openviking.storage.vectordb_adapters.factory import create_collection_adapter
from openviking_cli.utils.config import OpenVikingConfigSingleton, get_openviking_config

pytest.importorskip("lancedb")

DIM = 8


def _record(
    record_id: str,
    account_id: str,
    uri: str,
    vector: list[float],
    content: str,
    level: int = 2,
) -> dict:
    return {
        "id": record_id,
        "account_id": account_id,
        "uri": uri,
        "context_type": "resource",
        "level": level,
        "name": os.path.basename(uri),
        "vector": vector,
        "content": content,
    }


def _init_config(tmp_dir: str, lancedb_extra: dict | None = None, dimension: int = DIM) -> None:
    section: dict = {"uri": os.path.join(tmp_dir, "lance")}
    if lancedb_extra:
        section.update(lancedb_extra)
    path = os.path.join(tmp_dir, "ov.conf")
    with open(path, "w") as f:
        json.dump(
            {
                "storage": {
                    "vectordb": {"backend": "lancedb", "name": "context", "lancedb": section}
                },
                "embedding": {
                    "dense": {
                        "provider": "openai",
                        "model": "text-embedding-3-small",
                        "api_key": "mock-key",
                        "dimension": dimension,
                    }
                },
            },
            f,
        )
    OpenVikingConfigSingleton.initialize(config_path=path)


class TestLanceDBFTS(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.schema = CollectionSchemas.context_collection("context", DIM)

    def tearDown(self):
        OpenVikingConfigSingleton.reset_instance()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _new_adapter(self, lancedb_extra: dict | None = None):
        _init_config(self.test_dir, lancedb_extra)
        adapter = create_collection_adapter(get_openviking_config().storage.vectordb)
        adapter.create_collection(
            "context", self.schema, distance="cosine", sparse_weight=0.0, index_name="default"
        )
        return adapter

    # -- capability gating ------------------------------------------------------

    def test_fts_disabled_reports_not_implemented(self):
        adapter = self._new_adapter()
        try:
            assert adapter.USE_CONTENT_FIELD is False
            with pytest.raises(NotImplementedError):
                adapter.search_by_keywords(query="anything")
            meta = adapter.get_collection_info() or {}
            assert "FullText" not in meta
        finally:
            adapter.close()

    def test_fts_requires_store_content_validation(self):
        with pytest.raises(ValueError):
            self._new_adapter({"fts": {"columns": ["content"]}})

    def test_hybrid_requires_fts_validation(self):
        with pytest.raises(ValueError):
            self._new_adapter({"store_content": True, "hybrid": {"method": "rrf"}})

    # -- keyword search -----------------------------------------------------------

    def test_keyword_search_filters_and_pagination(self):
        adapter = self._new_adapter({"store_content": True, "fts": {}})
        try:
            assert adapter.USE_CONTENT_FIELD is True
            adapter.upsert(
                [
                    _record(
                        "a",
                        "acct-a",
                        "/resources/a.md",
                        [0.5] * DIM,
                        "kubernetes operator patterns",
                    ),
                    _record(
                        "b", "acct-a", "/resources/b.md", [0.5] * DIM, "docker compose networking"
                    ),
                    _record(
                        "c", "acct-b", "/resources/c.md", [0.5] * DIM, "kubernetes storage classes"
                    ),
                ]
            )
            hits = adapter.search_by_keywords(query="kubernetes", limit=10)
            assert {h["id"] for h in hits} == {"a", "c"}
            assert all(h["_score"] > 0 for h in hits)

            scoped = adapter.search_by_keywords(
                query="kubernetes", limit=10, filter=Eq("account_id", "acct-a")
            )
            assert {h["id"] for h in scoped} == {"a"}

            paged = adapter.search_by_keywords(query="kubernetes", limit=1, offset=1)
            # BM25 ties ('a' and 'c' score equally), so only the page size and
            # membership are deterministic.
            assert len(paged) == 1
            assert paged[0]["id"] in {"a", "c"}
        finally:
            adapter.close()

    def test_fts_reflects_updates_and_deletes(self):
        adapter = self._new_adapter({"store_content": True, "fts": {}})
        try:
            adapter.upsert(
                [_record("a", "acct-a", "/r/a.md", [0.5] * DIM, "initial draft about redis")]
            )
            adapter.upsert(
                [_record("a", "acct-a", "/r/a.md", [0.5] * DIM, "rewritten to talk about postgres")]
            )
            assert {h["id"] for h in adapter.search_by_keywords(query="postgres", limit=5)} == {"a"}
            assert adapter.search_by_keywords(query="redis", limit=5) == []

            adapter.delete(ids=["a"])
            assert adapter.search_by_keywords(query="postgres", limit=5) == []
        finally:
            adapter.close()

    def test_fts_reused_after_restart(self):
        adapter = self._new_adapter({"store_content": True, "fts": {}})
        try:
            adapter.upsert(
                [_record("a", "acct-a", "/r/a.md", [0.5] * DIM, "durable full text content")]
            )
        finally:
            adapter.close()

        adapter = create_collection_adapter(get_openviking_config().storage.vectordb)
        try:
            assert adapter.USE_CONTENT_FIELD is True
            meta = adapter.get_collection_info() or {}
            assert any(ft.get("Field") == "content" for ft in meta.get("FullText", []))
            hits = adapter.search_by_keywords(query="durable", limit=5)
            assert {h["id"] for h in hits} == {"a"}
        finally:
            adapter.close()

    def test_large_content_is_not_truncated(self):
        adapter = self._new_adapter({"store_content": True, "fts": {}})
        try:
            big = "openviking " * 400_000  # ~4.4 MB, above VikingDB-style limits
            adapter.upsert([_record("big", "acct-a", "/r/big.md", [0.5] * DIM, big)])
            stored = adapter.get(["big"])[0]["content"]
            assert stored == big
        finally:
            adapter.close()

    # -- multilingual ---------------------------------------------------------------

    def test_multilingual_content(self):
        adapter = self._new_adapter({"store_content": True, "fts": {}})
        try:
            adapter.upsert(
                [
                    _record("en", "acct-a", "/r/en.md", [0.5] * DIM, "the quick brown fox jumps"),
                    _record("vi", "acct-a", "/r/vi.md", [0.5] * DIM, "xin chào thế giới mở"),
                    _record("zh", "acct-a", "/r/zh.md", [0.5] * DIM, "开源上下文数据库"),
                    _record(
                        "mix", "acct-a", "/r/mix.md", [0.5] * DIM, "deploy Vietnamese modelモデル"
                    ),
                ]
            )
            assert adapter.search_by_keywords(query="fox", limit=5)[0]["id"] == "en"
            # Vietnamese with diacritics.
            assert adapter.search_by_keywords(query="chào", limit=5)[0]["id"] == "vi"
            assert len(adapter.search_by_keywords(query="Vietnamese", limit=5)) == 1
            # Documented tokenizer limitation: the whitespace tokenizer keeps
            # a CJK run as one token, so whole-token queries match while
            # sub-word CJK queries do not.  Sub-word matching stays the job
            # of the filesystem grep re-check.
            assert adapter.search_by_keywords(query="开源上下文数据库", limit=5)[0]["id"] == "zh"
            assert adapter.search_by_keywords(query="数据库", limit=5) == []
        finally:
            adapter.close()

    # -- hybrid ------------------------------------------------------------------

    def test_hybrid_fusion_respects_weights_and_filters(self):
        adapter = self._new_adapter(
            {
                "store_content": True,
                "fts": {},
                "hybrid": {"method": "weighted", "dense_weight": 0.9, "lexical_weight": 0.1},
            }
        )
        try:
            # 'target' is lexically strong AND close to the query vector;
            # 'decoy' only matches lexically.
            adapter.upsert(
                [
                    _record(
                        "target",
                        "acct-a",
                        "/r/t.md",
                        [1.0] + [0.0] * (DIM - 1),
                        "vector database benchmark",
                    ),
                    _record(
                        "decoy",
                        "acct-a",
                        "/r/d.md",
                        [-1.0] + [0.0] * (DIM - 1),
                        "vector database alternatives",
                    ),
                    _record(
                        "other",
                        "acct-b",
                        "/r/o.md",
                        [1.0] + [0.0] * (DIM - 1),
                        "vector database notes",
                    ),
                ]
            )
            query_vector = [1.0] + [0.0] * (DIM - 1)
            fused = adapter.search_by_keywords(
                query="vector database", query_vector=query_vector, limit=10
            )
            ids = [h["id"] for h in fused]
            assert "target" in ids and "decoy" in ids
            # Dense-dominant weights must rank the vector-close row first.
            assert ids[0] == "target"

            # The security filter applies to both candidate branches.
            scoped = adapter.search_by_keywords(
                query="vector database",
                query_vector=query_vector,
                filter=Eq("account_id", "acct-b"),
                limit=10,
            )
            assert {h["id"] for h in scoped} == {"other"}
        finally:
            adapter.close()

    def test_hybrid_rrf_method(self):
        adapter = self._new_adapter(
            {"store_content": True, "fts": {}, "hybrid": {"method": "rrf", "rrf_k": 60}}
        )
        try:
            adapter.upsert(
                [
                    _record("t1", "acct-a", "/r/t1.md", [1.0] + [0.0] * (DIM - 1), "vector search"),
                    _record(
                        "t2", "acct-a", "/r/t2.md", [0.9] + [0.1] * (DIM - 1), "keyword search"
                    ),
                ]
            )
            fused = adapter.search_by_keywords(
                query="search", query_vector=[1.0] + [0.0] * (DIM - 1), limit=10
            )
            assert {h["id"] for h in fused} == {"t1", "t2"}
            assert all(h["_score"] > 0 for h in fused)
        finally:
            adapter.close()

    def test_hybrid_without_config_raises(self):
        adapter = self._new_adapter({"store_content": True, "fts": {}})
        try:
            with pytest.raises(NotImplementedError):
                adapter.search_by_keywords(query="anything", query_vector=[0.5] * DIM)
        finally:
            adapter.close()


class TestGrepEngineResolution(unittest.TestCase):
    """The grep resolver must treat LanceDB like other full-text backends."""

    def test_lancedb_backend_resolves_to_remote_grep(self):
        import asyncio

        from openviking.storage.viking_fs._grep import _GrepMixin

        mixin = _GrepMixin.__new__(_GrepMixin)
        mixin._fulltext_available = None  # type: ignore[assignment]

        class _Store:
            _backend_type = "lancedb"

        class _OtherStore:
            _backend_type = "local"

        mixin._get_vector_store = lambda: _Store()  # type: ignore[assignment]
        mixin._collection_has_fulltext = lambda vector_store, ctx: asyncio.sleep(0, result=True)
        mixin._get_cached_count = lambda uri, ctx: asyncio.sleep(0, result=500_000)

        resolved = asyncio.run(
            mixin._resolve_grep_engine("auto", "viking://resources", None, 10000)
        )
        assert resolved == "vikingdb_then_fs"

    def test_lancedb_without_fulltext_falls_back_to_fs(self):
        import asyncio

        from openviking.storage.viking_fs._grep import _GrepMixin

        mixin = _GrepMixin.__new__(_GrepMixin)
        mixin._fulltext_available = None  # type: ignore[assignment]

        class _Store:
            _backend_type = "lancedb"

        mixin._get_vector_store = lambda: _Store()  # type: ignore[assignment]
        mixin._collection_has_fulltext = lambda vector_store, ctx: asyncio.sleep(0, result=False)

        resolved = asyncio.run(mixin._resolve_grep_engine("auto", "viking://resources", None, 0))
        assert resolved == "fs"
