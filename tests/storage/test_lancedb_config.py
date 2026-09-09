# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Configuration and factory tests for the LanceDB VectorDB backend."""

import json
import os
import shutil
import tempfile
import unittest

import pytest

from openviking.storage.vectordb_adapters.factory import create_collection_adapter
from openviking_cli.utils.config import OpenVikingConfigSingleton, get_openviking_config


def _write_config(tmp_dir: str, vectordb: dict, extra: dict | None = None) -> str:
    config_data = {
        "storage": {"vectordb": vectordb},
        "embedding": {
            "dense": {
                "provider": "openai",
                "model": "text-embedding-3-small",
                "api_key": "mock-key",
                "dimension": 8,
            }
        },
    }
    if extra:
        config_data.update(extra)
    path = os.path.join(tmp_dir, "ov.conf")
    with open(path, "w") as f:
        json.dump(config_data, f)
    return path


class TestLanceDBConfig(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        OpenVikingConfigSingleton.reset_instance()

    def _load(self, vectordb: dict):
        path = _write_config(self.test_dir, vectordb)
        OpenVikingConfigSingleton.initialize(config_path=path)
        return get_openviking_config().storage.vectordb

    def test_lancedb_requires_uri(self):
        with pytest.raises(ValueError):
            self._load({"backend": "lancedb"})

    def test_lancedb_rejects_sparse_weight(self):
        with pytest.raises(ValueError):
            self._load(
                {
                    "backend": "lancedb",
                    "lancedb": {"uri": "/tmp/x"},
                    "sparse_weight": 0.5,
                }
            )

    def test_lancedb_rejects_unknown_metric(self):
        with pytest.raises(ValueError):
            self._load(
                {
                    "backend": "lancedb",
                    "lancedb": {"uri": "/tmp/x"},
                    "distance_metric": "manhattan",
                }
            )

    def test_lancedb_accepts_valid_config(self):
        cfg = self._load(
            {
                "backend": "lancedb",
                "name": "context",
                "lancedb": {
                    "uri": "/tmp/lance-store",
                    "storage_options": {"aws_endpoint": "http://localhost:8333"},
                },
            }
        )
        assert cfg.backend == "lancedb"
        assert cfg.lancedb is not None
        assert cfg.lancedb.uri == "/tmp/lance-store"
        assert cfg.lancedb.storage_options["aws_endpoint"] == "http://localhost:8333"


class TestLanceDBFactory(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        OpenVikingConfigSingleton.reset_instance()

    def test_factory_routes_to_lancedb_adapter(self):
        pytest.importorskip("lancedb")
        path = _write_config(
            self.test_dir,
            {
                "backend": "lancedb",
                "name": "context",
                "lancedb": {"uri": os.path.join(self.test_dir, "lance")},
            },
        )
        OpenVikingConfigSingleton.initialize(config_path=path)
        config = get_openviking_config().storage.vectordb
        adapter = create_collection_adapter(config)
        assert adapter.mode == "lancedb"
        assert adapter.collection_name == "context"
        adapter.close()

    def test_factory_lists_lancedb_in_error(self):
        with pytest.raises(ValueError) as excinfo:
            create_collection_adapter(
                type(
                    "Cfg",
                    (),
                    {
                        "backend": "does-not-exist",
                        "custom_params": {},
                    },
                )()
            )
        assert "lancedb" in str(excinfo.value)
