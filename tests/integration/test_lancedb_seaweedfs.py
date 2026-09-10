# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""SeaweedFS integration tests for the LanceDB backend (ADR-0075/0076 gates).

Two connection modes are covered:

- ``s3`` (default): the table lives on the SeaweedFS S3 gateway, opened via a
  plain ``s3://bucket/prefix`` URI with no catalog.  This works on any
  SeaweedFS deployment with S3 enabled.
- ``namespace``: tables are declared through the Lance Namespace REST catalog
  (SeaweedFS Lake, ``-s3.port.lance``).  Requires a Lance table bucket.

Tests are skipped unless ``OV_IT_SEAWEEDFS=1`` (plus mode-specific variables
and S3 credentials) is set.

Runbook (homelab)::

    # Credentials: use the shared test identity (see the weed skill:
    # /mnt/user/appdata/seaweedfs/s3.json, identity 'test-apps', scope *:appdata)
    export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
    export AWS_REGION=us-east-1

    # S3-direct mode (works today, no Lance namespace needed):
    export OV_IT_SEAWEEDFS=1
    export OV_IT_SEAWEEDFS_S3=http://100.68.251.84:8333      # or http://seaweedfs-s3:8333
    pytest tests/integration/test_lancedb_seaweedfs.py -v

    # Namespace mode (optional; enable the Lance catalog and create a table
    # bucket first):
    #   weed server ... -s3 -s3.port.lance=9101
    #   weed shell 's3tables.bucket -create -name vectors -format LANCE -account 000000000000'
    export OV_IT_SEAWEEDFS_LANCE=http://seaweedfs:9101
    export OV_IT_SEAWEEDFS_BUCKET=vectors

Tables are created under unique names and dropped during teardown.
"""

import json
import os
import shutil
import tempfile
import uuid

import pytest

from openviking.storage.collection_schemas import CollectionSchemas
from openviking.storage.expr import And, Eq, In, PathScope
from openviking.storage.vectordb_adapters.factory import create_collection_adapter
from openviking_cli.utils.config import OpenVikingConfigSingleton, get_openviking_config

lancedb = pytest.importorskip("lancedb")

RUN = os.environ.get("OV_IT_SEAWEEDFS") == "1"
S3_ENDPOINT = os.environ.get("OV_IT_SEAWEEDFS_S3", "http://100.68.251.84:8333")
LANCE_ENDPOINT = os.environ.get("OV_IT_SEAWEEDFS_LANCE")
S3_PREFIX = os.environ.get("OV_IT_SEAWEEDFS_PREFIX", "appdata/test-openviking-lance")
NAMESPACE = os.environ.get("OV_IT_SEAWEEDFS_NAMESPACE", "openviking_it")
BUCKET = os.environ.get("OV_IT_SEAWEEDFS_BUCKET", "vectors")
REGION = os.environ.get("AWS_REGION", "us-east-1")
HAS_CREDENTIALS = bool(
    os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")
)

pytestmark = pytest.mark.skipif(
    not RUN or not HAS_CREDENTIALS,
    reason=(
        "Set OV_IT_SEAWEEDFS=1 plus AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY "
        "(test-apps identity) to run SeaweedFS integration tests"
    ),
)

DIM = 8


def _storage_options() -> dict[str, str]:
    return {
        "aws_endpoint": S3_ENDPOINT,
        "aws_region": REGION,
        "allow_http": "true",
    }


def _record(record_id, account_id, uri, vector, level=2):
    return {
        "id": record_id,
        "account_id": account_id,
        "uri": uri,
        "context_type": "resource",
        "level": level,
        "name": uri.rsplit("/", 1)[-1],
        "vector": vector,
        "created_at": "2026-01-01T00:00:00+00:00",
    }


MODES = ["s3"] + (["namespace"] if LANCE_ENDPOINT else [])


class _SeaweedFsEnv:
    """Binds the config singleton to a SeaweedFS-backed LanceDB backend."""

    def __init__(self, tmp_dir: str, table_name: str, mode: str, lancedb_extra: dict | None = None):
        self.tmp_dir = tmp_dir
        self.table_name = table_name
        self.mode = mode
        if mode == "namespace":
            section = {
                "namespace_uri": LANCE_ENDPOINT,
                "namespace_path": [BUCKET, NAMESPACE],
                "storage_options": _storage_options(),
            }
        else:
            section = {
                "uri": f"s3://{S3_PREFIX}",
                "storage_options": _storage_options(),
            }
        if lancedb_extra:
            section.update(lancedb_extra)
        config_data = {
            "storage": {"vectordb": {"backend": "lancedb", "name": table_name, "lancedb": section}},
            "embedding": {
                "dense": {
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                    "api_key": "mock-key",
                    "dimension": DIM,
                }
            },
        }
        path = os.path.join(tmp_dir, "ov.conf")
        with open(path, "w") as f:
            json.dump(config_data, f)
        OpenVikingConfigSingleton.initialize(config_path=path)

    def new_adapter(self):
        adapter = create_collection_adapter(get_openviking_config().storage.vectordb)
        schema = CollectionSchemas.context_collection(self.table_name, DIM)
        adapter.create_collection(
            self.table_name,
            schema,
            distance="cosine",
            sparse_weight=0.0,
            index_name="default",
        )
        return adapter

    def direct_db(self):
        """Open the same data without the catalog (s3:// prefix is a valid
        Lance directory), proving data ownership is independent of catalogs."""
        return lancedb.connect(f"s3://{S3_PREFIX}", storage_options=_storage_options())

    def close(self):
        OpenVikingConfigSingleton.reset_instance()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)


def _make_env(mode, lancedb_extra=None):
    return _SeaweedFsEnv(
        tempfile.mkdtemp(prefix="ov-lance-it-"),
        table_name=f"context_it_{mode}_{uuid.uuid4().hex[:8]}",
        mode=mode,
        lancedb_extra=lancedb_extra,
    )


def _teardown(env):
    try:
        adapter = env.new_adapter()
        if adapter.collection_exists():
            adapter.drop_collection()
        adapter.close()
    except Exception:
        pass
    env.close()


@pytest.fixture(params=MODES)
def env(request):
    e = _make_env(request.param)
    yield e
    _teardown(e)


# -- ADR-0075: persistence & lifecycle -----------------------------------------


def test_table_lifecycle_and_reconnect(env):
    """Create/open through the adapter, then reconnect with a fresh client."""
    adapter = env.new_adapter()
    try:
        assert adapter.collection_exists()
        adapter.upsert([_record("boot", "acct-a", "/resources/it/boot.md", [0.5] * DIM)])
    finally:
        adapter.close()

    adapter = env.new_adapter()
    try:
        assert adapter.collection_exists()
        assert adapter.count() == 1
    finally:
        adapter.close()


def test_direct_lance_uri_access_without_catalog(env):
    """Data ownership lives in S3: the dataset opens with no catalog path."""
    adapter = env.new_adapter()
    try:
        adapter.upsert(
            [
                _record("a", "acct-a", "/resources/it/a.md", [0.5] * DIM),
                _record("b", "acct-a", "/resources/it/b.md", [0.4] * DIM),
            ]
        )
        direct = env.direct_db().open_table(env.table_name)
        assert direct.count_rows() == 2
        rows = direct.search([0.5] * DIM).metric("cosine").limit(1).to_list()
        assert rows[0]["id"] == "a"
    finally:
        adapter.close()


def test_persistence_after_reconnect_without_reingest(env):
    """Recreating the client must preserve searchable data (no re-embedding)."""
    adapter = env.new_adapter()
    try:
        adapter.upsert([_record("keep", "acct-a", "/resources/it/keep.md", [0.9] * DIM)])
    finally:
        adapter.close()

    adapter = env.new_adapter()
    try:
        assert adapter.count() == 1
        hits = adapter.query(query_vector=[0.9] * DIM, limit=5)
        assert [h["id"] for h in hits] == ["keep"]
    finally:
        adapter.close()


def test_crud_filters_and_acl_isolation(env):
    adapter = env.new_adapter()
    try:
        adapter.upsert(
            [
                _record(
                    "a", "acct-a", "/user/u1/resources/a.md", [1.0] + [0.0] * (DIM - 1), level=0
                ),
                _record(
                    "b",
                    "acct-a",
                    "/user/u1/resources/sub/b.md",
                    [0.0, 1.0] + [0.0] * (DIM - 2),
                    level=1,
                ),
                _record(
                    "c", "acct-b", "/user/u2/resources/c.md", [0.0] * (DIM - 1) + [1.0], level=2
                ),
            ]
        )
        query = [1.0] + [0.0] * (DIM - 1)

        scoped = adapter.query(
            query_vector=query,
            filter=And(
                [
                    Eq("account_id", "acct-a"),
                    PathScope("uri", "viking://user/u1/resources", depth=-1),
                    In("level", [0, 1]),
                ]
            ),
            limit=10,
        )
        assert {h["id"] for h in scoped} == {"a", "b"}

        isolated = adapter.query(query_vector=query, filter=Eq("account_id", "acct-b"), limit=10)
        assert {h["id"] for h in isolated} == {"c"}

        assert adapter.delete(ids=["c"]) == 1
        assert adapter.count() == 2
    finally:
        adapter.close()


# -- ADR-0076: incremental visibility on object storage --------------------------


def test_ann_incremental_visibility_on_object_store(env):
    """The critical invariant: build index -> append -> search still sees rows.

    If this fails, appended rows are silently dropped from search and the
    backend must not ship.
    """
    adapter = env.new_adapter(
        lancedb_extra={
            "vector_index": {
                "index_type": "IVF_FLAT",
                "num_partitions": 2,
                "min_rows_to_build": 10,
            }
        }
    )
    try:
        rows = []
        for i in range(30):
            vector = [0.0] * DIM
            vector[0] = 1.0 - (i % 5) * 0.05
            rows.append(_record(f"r{i}", "acct-a", f"/resources/it/r{i}.md", vector))
        adapter.upsert(rows)

        query = [1.0] + [0.0] * (DIM - 1)
        assert adapter.query(query_vector=query, limit=5)

        adapter.upsert(
            [_record("late", "acct-a", "/resources/it/late.md", [1.0] + [0.0] * (DIM - 1))]
        )
        hits = adapter.query(query_vector=query, limit=100)
        ids = {h["id"] for h in hits}
        assert "late" in ids, "row appended after index build disappeared from search"
    finally:
        adapter.close()


# -- namespace-catalog specific (only when the Lance endpoint is provided) --------


@pytest.mark.skipif(not LANCE_ENDPOINT, reason="OV_IT_SEAWEEDFS_LANCE not set")
def test_namespace_catalog_lifecycle():
    """Tables are declared and listed through the Lance namespace catalog."""
    env_ns = _make_env("namespace")
    try:
        adapter = env_ns.new_adapter()
        try:
            assert adapter.collection_exists()
            db = lancedb.connect_namespace(
                "rest", {"uri": LANCE_ENDPOINT}, storage_options=_storage_options()
            )
            names = db.table_names(namespace_path=[BUCKET, NAMESPACE])
            assert env_ns.table_name in [n.rsplit("$", 1)[-1] for n in names]
        finally:
            adapter.close()
    finally:
        _teardown(env_ns)
