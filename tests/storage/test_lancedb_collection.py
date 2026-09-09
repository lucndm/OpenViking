# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Integration tests for the LanceDB collection backend.

These tests exercise lifecycle, CRUD, dense search, filters, and ACL
isolation against a real LanceDB table on local disk.  They are skipped when
the optional ``lancedb`` package is not installed.
"""

import json
import os
import shutil
import tempfile
import unittest

import pytest

from openviking.storage.collection_schemas import CollectionSchemas
from openviking.storage.expr import And, Eq, In, PathScope, RawDSL
from openviking.storage.vectordb_adapters.factory import create_collection_adapter
from openviking_cli.utils.config import OpenVikingConfigSingleton, get_openviking_config

lancedb = pytest.importorskip("lancedb")

DIM = 8
ACL_MODE_FIELD = "acl_mode"


def _record(
    record_id: str,
    account_id: str,
    uri: str,
    vector: list[float],
    level: int = 2,
    acl_mode: str | None = None,
    grants: list[str] | None = None,
    created_at: str = "2026-01-01T00:00:00+00:00",
) -> dict:
    return {
        "id": record_id,
        "account_id": account_id,
        "uri": uri,
        "context_type": "resource",
        "level": level,
        "name": os.path.basename(uri),
        "vector": vector,
        "created_at": created_at,
        "updated_at": created_at,
        "active_count": 1,
        "search_tags": grants or [],
        "acl_mode": acl_mode,
        "acl_direct_grants": grants or [],
    }


class TestLanceDBCollectionIntegration(unittest.TestCase):
    def setUp(self):
        pytest.importorskip("lancedb")
        self.test_dir = tempfile.mkdtemp()
        self.lance_uri = os.path.join(self.test_dir, "lance")
        config_data = {
            "storage": {
                "vectordb": {
                    "backend": "lancedb",
                    "name": "context",
                    "lancedb": {"uri": self.lance_uri},
                }
            },
            "embedding": {
                "dense": {
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                    "api_key": "mock-key",
                    "dimension": DIM,
                }
            },
        }
        self.config_path = os.path.join(self.test_dir, "ov.conf")
        with open(self.config_path, "w") as f:
            json.dump(config_data, f)
        OpenVikingConfigSingleton.initialize(config_path=self.config_path)
        self.adapter = create_collection_adapter(get_openviking_config().storage.vectordb)
        self.schema = CollectionSchemas.context_collection("context", DIM)

    def tearDown(self):
        self.adapter.close()
        OpenVikingConfigSingleton.reset_instance()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create(self):
        created = self.adapter.create_collection(
            "context",
            self.schema,
            distance="cosine",
            sparse_weight=0.0,
            index_name="default",
        )
        assert created is True

    # -- lifecycle ----------------------------------------------------------

    def test_lifecycle_create_exists_drop(self):
        assert not self.adapter.collection_exists()
        self._create()
        assert self.adapter.collection_exists()
        # Second create must not create a shadow table.
        assert not self.adapter.create_collection(
            "context", self.schema, distance="cosine", sparse_weight=0.0, index_name="default"
        )
        db = lancedb.connect(self.lance_uri)
        assert db.table_names() == ["context"]
        assert self.adapter.drop_collection() is True
        assert not self.adapter.collection_exists()
        assert db.table_names() == []

    def test_restart_preserves_data_without_reingest(self):
        self._create()
        self.adapter.upsert(
            [
                _record(
                    "r1",
                    "acct-a",
                    "/resources/proj/a.md",
                    [0.1] * DIM,
                )
            ]
        )
        self.adapter.close()

        adapter2 = create_collection_adapter(get_openviking_config().storage.vectordb)
        try:
            assert adapter2.collection_exists()
            results = adapter2.query(query_vector=[0.1] * DIM, limit=5)
            assert len(results) == 1
            assert results[0]["id"] == "r1"
            assert results[0]["uri"] == "viking://resources/proj/a.md"
        finally:
            adapter2.close()

    def test_dimension_mismatch_is_rejected_on_open(self):
        self._create()
        self.adapter.upsert([_record("r1", "acct-a", "/resources/a.md", [0.1] * DIM)])
        self.adapter.close()

        mismatched = {
            "storage": {
                "vectordb": {
                    "backend": "lancedb",
                    "name": "context",
                    "lancedb": {"uri": self.lance_uri},
                }
            },
            "embedding": {
                "dense": {
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                    "api_key": "mock-key",
                    "dimension": DIM + 2,
                }
            },
        }
        path2 = os.path.join(self.test_dir, "ov2.conf")
        with open(path2, "w") as f:
            json.dump(mismatched, f)
        OpenVikingConfigSingleton.reset_instance()
        OpenVikingConfigSingleton.initialize(config_path=path2)
        adapter2 = create_collection_adapter(get_openviking_config().storage.vectordb)
        with pytest.raises(ValueError, match="dimension"):
            adapter2.collection_exists()
        adapter2.close()

    # -- CRUD ---------------------------------------------------------------

    def test_upsert_get_delete_count(self):
        self._create()
        ids = self.adapter.upsert(
            [
                _record("a", "acct-a", "/resources/a.md", [1.0] + [0.0] * (DIM - 1)),
                _record("b", "acct-a", "/resources/b.md", [0.0, 1.0] + [0.0] * (DIM - 2)),
                _record("c", "acct-b", "/resources/c.md", [0.0] * (DIM - 1) + [1.0]),
            ]
        )
        assert ids == ["a", "b", "c"]
        assert self.adapter.count() == 3

        # Upsert with duplicate id replaces instead of duplicating.
        self.adapter.upsert([_record("a", "acct-a", "/resources/a2.md", [1.0] + [0.0] * (DIM - 1))])
        assert self.adapter.count() == 3
        fetched = self.adapter.get(["a"])
        assert len(fetched) == 1
        assert fetched[0]["uri"] == "viking://resources/a2.md"

        assert self.adapter.delete(ids=["c"]) == 1
        assert self.adapter.count() == 2
        assert self.adapter.get(["c"]) == []

        assert self.adapter.clear() is True
        assert self.adapter.count() == 0

    def test_update_preserves_identity(self):
        self._create()
        self.adapter.upsert([_record("a", "acct-a", "/resources/a.md", [0.5] * DIM)])
        coll = self.adapter.get_collection()
        coll.update_data([{"id": "a", "name": "renamed"}])
        fetched = self.adapter.get(["a"])[0]
        assert fetched["name"] == "renamed"
        assert fetched["uri"] == "viking://resources/a.md"
        assert self.adapter.count() == 1

    def test_delete_isolation_between_accounts(self):
        self._create()
        self.adapter.upsert(
            [
                _record("a", "acct-a", "/user/u1/resources/a.md", [0.9] * DIM),
                _record("b", "acct-b", "/user/u2/resources/b.md", [0.9] * DIM),
            ]
        )
        deleted = self.adapter.delete(filter=Eq("account_id", "acct-a"))
        assert deleted == 1
        remaining = self.adapter.query(query_vector=[0.9] * DIM, limit=10)
        assert [r["account_id"] for r in remaining] == ["acct-b"]

    # -- dense search ---------------------------------------------------------

    def test_dense_search_ordering_limit_offset_projection(self):
        self._create()
        base = [0.0] * DIM
        rows = []
        for i in range(5):
            vector = list(base)
            vector[0] = 1.0 - i * 0.1
            rows.append(_record(f"r{i}", "acct-a", f"/resources/r{i}.md", vector))
        self.adapter.upsert(rows)

        query = [1.0] + [0.0] * (DIM - 1)
        results = self.adapter.query(query_vector=query, limit=3)
        assert [r["id"] for r in results] == ["r0", "r1", "r2"]
        assert results[0]["_score"] >= results[1]["_score"] >= results[2]["_score"]

        paged = self.adapter.query(query_vector=query, limit=2, offset=1)
        assert [r["id"] for r in paged] == ["r1", "r2"]

        projected = self.adapter.query(query_vector=query, limit=1, output_fields=["name"])
        assert set(projected[0].keys()) == {"id", "_score", "name"}

    def test_dense_search_dimension_mismatch_fails_fast(self):
        self._create()
        with pytest.raises(ValueError, match="dimension"):
            self.adapter.query(query_vector=[0.1, 0.2], limit=3)

    def test_filters_eq_in_range_and_or(self):
        self._create()
        self.adapter.upsert(
            [
                _record("l0", "acct-a", "/resources/x/a.md", [0.5] * DIM, level=0),
                _record("l1", "acct-a", "/resources/x/b.md", [0.5] * DIM, level=1),
                _record("l2", "acct-b", "/resources/c.md", [0.5] * DIM, level=2),
            ]
        )
        query = [0.5] * DIM

        eq = self.adapter.query(query_vector=query, filter=Eq("account_id", "acct-a"), limit=10)
        assert {r["id"] for r in eq} == {"l0", "l1"}

        multi = self.adapter.query(query_vector=query, filter=In("level", [0, 2]), limit=10)
        assert {r["id"] for r in multi} == {"l0", "l2"}

        combined = self.adapter.query(
            query_vector=query,
            filter=And([Eq("account_id", "acct-a"), In("level", [0, 1])]),
            limit=10,
        )
        assert {r["id"] for r in combined} == {"l0", "l1"}

    def test_path_scope_depth(self):
        self._create()
        self.adapter.upsert(
            [
                _record("root", "acct-a", "/res", [0.5] * DIM),
                _record("d1", "acct-a", "/res/docs", [0.5] * DIM),
                _record("d2", "acct-a", "/res/docs/api.md", [0.5] * DIM),
                _record("d3", "acct-a", "/res/docs/api/v2.md", [0.5] * DIM),
                _record("out", "acct-a", "/other", [0.5] * DIM),
            ]
        )
        query = [0.5] * DIM
        any_depth = self.adapter.query(
            query_vector=query, filter=PathScope("uri", "viking://res", depth=-1), limit=10
        )
        assert {r["id"] for r in any_depth} == {"root", "d1", "d2", "d3"}

        depth1 = self.adapter.query(
            query_vector=query, filter=PathScope("uri", "viking://res", depth=1), limit=10
        )
        assert {r["id"] for r in depth1} == {"root", "d1"}

        exact = self.adapter.query(
            query_vector=query, filter=PathScope("uri", "viking://res/docs", depth=0), limit=10
        )
        assert {r["id"] for r in exact} == {"d1"}

    # -- ACL semantics (integration, per ADR-0075) ----------------------------

    def test_acl_filter_shapes_match_semantics(self):
        self._create()
        self.adapter.upsert(
            [
                _record("plain", "acct-a", "/user/u1/resources/p.md", [0.5] * DIM),
                _record(
                    "shared",
                    "acct-a",
                    "/resources/s.md",
                    [0.5] * DIM,
                    acl_mode="restricted",
                    grants=["acct-a:read"],
                ),
                _record(
                    "inherit",
                    "acct-a",
                    "/resources/i.md",
                    [0.5] * DIM,
                    acl_mode="inherit",
                    grants=["acct-b:read"],
                ),
            ]
        )
        query = [0.5] * DIM
        # Rows without ACL mode (null) must remain visible under must_not.
        uncontrolled = self.adapter.query(
            query_vector=query,
            filter=And(
                [
                    Eq("account_id", "acct-a"),
                    RawDSL(
                        {
                            "op": "must_not",
                            "field": ACL_MODE_FIELD,
                            "conds": ["inherit", "restricted"],
                        }
                    ),
                ]
            ),
            limit=10,
        )
        assert {r["id"] for r in uncontrolled} == {"plain"}

        granted = self.adapter.query(
            query_vector=query,
            filter=And(
                [
                    Eq("account_id", "acct-a"),
                    In("acl_mode", ["inherit", "restricted"]),
                    In("acl_direct_grants", ["acct-a:read"]),
                ]
            ),
            limit=10,
        )
        assert {r["id"] for r in granted} == {"shared"}

    def test_account_isolation_in_query_path(self):
        self._create()
        self.adapter.upsert(
            [
                _record("mine", "acct-a", "/user/u1/resources/m.md", [0.9] * DIM),
                _record("theirs", "acct-b", "/user/u2/resources/t.md", [0.9] * DIM),
            ]
        )
        visible = self.adapter.query(
            query_vector=[0.9] * DIM, filter=Eq("account_id", "acct-a"), limit=10
        )
        assert {r["id"] for r in visible} == {"mine"}

    # -- unsupported capability ------------------------------------------------

    def test_sparse_query_fails_fast(self):
        self._create()
        coll = self.adapter.get_collection()
        with pytest.raises(NotImplementedError):
            coll.search_by_vector(
                "default",
                dense_vector=[0.5] * DIM,
                sparse_vector={"term": 1.0},
            )

    def test_keyword_search_reports_not_enabled(self):
        self._create()
        with pytest.raises(NotImplementedError):
            self.adapter.search_by_keywords(query="anything")

    # -- scalar sort + aggregate -----------------------------------------------

    def test_scalar_sort_and_count(self):
        self._create()
        self.adapter.upsert(
            [
                _record("new", "acct-a", "/resources/n.md", [0.5] * DIM, level=2),
                _record("mid", "acct-a", "/resources/m.md", [0.5] * DIM, level=1),
                _record("old", "acct-a", "/resources/o.md", [0.5] * DIM, level=0),
            ]
        )
        newest = self.adapter.query(order_by="level", order_desc=True, limit=2)
        assert [r["id"] for r in newest] == ["new", "mid"]
        assert self.adapter.count(filter=Eq("account_id", "acct-a")) == 3
