# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Indexing tests for the LanceDB backend (ADR Phase 2).

Covers Lance-native ANN + scalar indexes, the incremental-visibility
invariant (build index -> append -> search still sees new rows), coalesced
bulk-ingest maintenance, deferred index builds, in-place upgrades of
Phase-1 tables, and index reuse across restarts.  Skipped when the optional
``lancedb`` package is missing.
"""

import json
import os
import shutil
import tempfile
import unittest

import pytest

from openviking.storage.collection_schemas import CollectionSchemas
from openviking.storage.expr import And, Eq, In, PathScope
from openviking.storage.vectordb_adapters.factory import create_collection_adapter
from openviking_cli.utils.config import OpenVikingConfigSingleton, get_openviking_config

pytest.importorskip("lancedb")

DIM = 8
INDEX_ROWS = 60


def _record(record_id: str, account_id: str, uri: str, vector: list[float], level: int = 2):
    return {
        "id": record_id,
        "account_id": account_id,
        "uri": uri,
        "context_type": "resource",
        "level": level,
        "name": os.path.basename(uri),
        "vector": vector,
        "search_tags": [f"tag-{level}"],
    }


def _row(i: int, account_id: str = "acct-a") -> dict:
    vector = [0.0] * DIM
    vector[0] = 1.0 if i % 2 == 0 else 0.5
    vector[1] = 0.1 * (i % 5)
    return _record(f"r{i}", account_id, f"/resources/p/r{i}.md", vector, level=i % 3)


def _lancedb_cfg(tmp_dir: str, extra: dict | None = None) -> dict:
    section: dict = {"uri": os.path.join(tmp_dir, "lance")}
    if extra:
        section.update(extra)
    return {
        "backend": "lancedb",
        "name": "context",
        "lancedb": section,
    }


def _init_config(tmp_dir: str, lancedb_extra: dict | None = None) -> None:
    path = os.path.join(tmp_dir, "ov.conf")
    with open(path, "w") as f:
        json.dump(
            {
                "storage": {"vectordb": _lancedb_cfg(tmp_dir, lancedb_extra)},
                "embedding": {
                    "dense": {
                        "provider": "openai",
                        "model": "text-embedding-3-small",
                        "api_key": "mock-key",
                        "dimension": DIM,
                    }
                },
            },
            f,
        )
    OpenVikingConfigSingleton.initialize(config_path=path)


class TestLanceDBIndexing(unittest.TestCase):
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

    def _inner(self, adapter):
        return adapter._collection._Collection__collection

    # -- incremental visibility ------------------------------------------------

    def test_appended_rows_stay_visible_after_index_build(self):
        adapter = self._new_adapter(
            {
                "vector_index": {
                    "index_type": "IVF_PQ",
                    "num_partitions": 2,
                    "num_sub_vectors": 2,
                    "min_rows_to_build": 10,
                }
            }
        )
        try:
            # IVF_PQ training needs at least 256 rows.
            adapter.upsert([_row(i) for i in range(300)])
            query = [1.0] + [0.0] * (DIM - 1)

            before = adapter.query(query_vector=query, limit=5)
            assert len(before) >= 5

            # The critical invariant: build index -> append -> search -> the
            # new state must remain visible (no silent misses).
            adapter.upsert([_row(10_000)])
            after = adapter.query(query_vector=query, limit=INDEX_ROWS + 10)
            ids = {r["id"] for r in after}
            assert "r10000" in ids, "appended row disappeared after index build"
        finally:
            adapter.close()

    def test_combined_ann_metadata_acl_query(self):
        adapter = self._new_adapter(
            {
                "vector_index": {
                    "index_type": "IVF_FLAT",
                    "num_partitions": 2,
                    "min_rows_to_build": 10,
                },
                "scalar_indexes": {"level": "BTREE", "search_tags": "LABEL_LIST"},
            }
        )
        try:
            adapter.upsert([_row(i) for i in range(INDEX_ROWS)])
            adapter.upsert([_row(i, account_id="acct-b") for i in range(100, 110)])
            query = [1.0] + [0.0] * (DIM - 1)
            combined = adapter.query(
                query_vector=query,
                filter=And(
                    [
                        Eq("account_id", "acct-a"),
                        In("level", [0, 1]),
                        PathScope("uri", "viking://resources", depth=-1),
                    ]
                ),
                limit=200,
            )
            assert combined, "indexed query returned nothing"
            for row in combined:
                assert row["account_id"] == "acct-a"
                assert row["level"] in (0, 1)
            # With an ANN index the candidate set must not be truncated by
            # an undersized top-K: every matching row stays retrievable.
            assert len(combined) >= 20
        finally:
            adapter.close()

    # -- bulk ingest coalescing -------------------------------------------------

    def test_bulk_ingest_coalesces_maintenance(self):
        adapter = self._new_adapter({"optimize_after_bulk_ingest": True})
        inner = self._inner(adapter)
        try:
            adapter.begin_bulk_ingest()
            for batch in range(5):
                adapter.upsert([_row(batch * 10 + i) for i in range(10)])
            # No maintenance may run while the scope is open.
            assert inner.maintenance_runs == 0
            adapter.end_bulk_ingest()
            # N batches -> exactly one coalesced maintenance action.
            assert inner.maintenance_runs == 1
        finally:
            adapter.close()

    def test_bulk_ingest_without_optimize_config_is_noop(self):
        adapter = self._new_adapter({"optimize_after_bulk_ingest": False})
        inner = self._inner(adapter)
        try:
            adapter.begin_bulk_ingest()
            adapter.upsert([_row(i) for i in range(10)])
            adapter.end_bulk_ingest()
            assert inner.maintenance_runs == 0
        finally:
            adapter.close()

    # -- deferred build + explicit optimize ---------------------------------------

    def test_vector_index_defers_until_min_rows(self):
        adapter = self._new_adapter(
            {
                "vector_index": {
                    "index_type": "IVF_FLAT",
                    "num_partitions": 2,
                    "min_rows_to_build": 1000,
                }
            }
        )
        inner = self._inner(adapter)
        try:
            adapter.upsert([_row(i) for i in range(20)])
            meta = inner.get_index_meta_data("default")
            assert meta.get("state") == "pending"
            assert "default" not in inner._native_index_names()

            # Explicit optimize forces the build even below min_rows.
            inner.optimize_index("default")
            meta = inner.get_index_meta_data("default")
            assert meta.get("state") == "built"
            assert meta.get("num_indexed_rows", 0) >= 20
            assert "default" in inner._native_index_names()
        finally:
            adapter.close()

    # -- phase-1 upgrade + restart reuse ------------------------------------------

    def test_phase1_table_upgrades_in_place(self):
        # Phase-1 table: created without any native index configuration.
        adapter = self._new_adapter()
        try:
            adapter.upsert([_row(i) for i in range(INDEX_ROWS)])
        finally:
            adapter.close()

        # Reopen with index configuration and build indexes in place.
        _init_config(
            self.test_dir,
            {
                "vector_index": {
                    "index_type": "IVF_FLAT",
                    "num_partitions": 2,
                    "min_rows_to_build": 10,
                },
                "scalar_indexes": {"level": "BTREE"},
            },
        )
        adapter = create_collection_adapter(get_openviking_config().storage.vectordb)
        try:
            assert adapter.collection_exists()
            assert adapter.count() == INDEX_ROWS, "upgrade must not lose data"
            coll = adapter.get_collection()
            coll.create_index(
                "default",
                {
                    "IndexName": "default",
                    "VectorIndex": {"IndexType": "ivf_pq", "Distance": "cosine"},
                    "ScalarIndex": ["level"],
                },
            )
            inner = self._inner(adapter)
            names = inner._native_index_names()
            assert "default" in names
            assert any(name.startswith("level") for name in names)

            results = adapter.query(query_vector=[1.0] + [0.0] * (DIM - 1), limit=10)
            assert len(results) == 10
        finally:
            adapter.close()

    def test_restart_reuses_native_index_without_rebuild(self):
        adapter = self._new_adapter(
            {
                "vector_index": {
                    "index_type": "IVF_FLAT",
                    "num_partitions": 2,
                    "min_rows_to_build": 10,
                }
            }
        )
        try:
            adapter.upsert([_row(i) for i in range(INDEX_ROWS)])
            inner = self._inner(adapter)
            assert "default" in inner._native_index_names()
        finally:
            adapter.close()

        adapter = create_collection_adapter(get_openviking_config().storage.vectordb)
        try:
            assert adapter.collection_exists()
            inner = self._inner(adapter)
            assert "default" in inner._native_index_names()
            meta = inner.get_index_meta_data("default")
            assert meta.get("state") == "built"
            # Data survived; no re-ingestion happened.
            assert adapter.count() == INDEX_ROWS
        finally:
            adapter.close()

    def test_drop_index_removes_native_index(self):
        adapter = self._new_adapter(
            {
                "vector_index": {
                    "index_type": "IVF_FLAT",
                    "num_partitions": 2,
                    "min_rows_to_build": 10,
                }
            }
        )
        try:
            adapter.upsert([_row(i) for i in range(INDEX_ROWS)])
            inner = self._inner(adapter)
            assert "default" in inner._native_index_names()
            inner.drop_index("default")
            assert "default" not in inner._native_index_names()
            # Search keeps working via flat scan.
            results = adapter.query(query_vector=[1.0] + [0.0] * (DIM - 1), limit=5)
            assert len(results) == 5
        finally:
            adapter.close()
