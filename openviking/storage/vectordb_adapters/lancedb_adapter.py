# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""LanceDB backend collection adapter (dense-retrieval MVP).

The adapter binds a single OpenViking collection to a LanceDB table.  The
table can live on a local directory or on object storage (for example a
SeaweedFS S3 endpoint with a Lance namespace catalog).  Only dense retrieval
is configured in this phase; sparse or hybrid settings fail fast.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

from openviking.storage.vectordb.collection.collection import Collection
from openviking.storage.vectordb.collection.lancedb_collection import (
    LANCEDB_AVAILABLE,
    create_lancedb_collection,
)

from .base import CollectionAdapter

_ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def _expand_env_placeholders(storage_options: Dict[str, str]) -> Dict[str, str]:
    """Expand ``${VAR}`` placeholders so permanent keys stay out of config files."""
    expanded: Dict[str, str] = {}
    for key, value in storage_options.items():
        match = _ENV_PATTERN.match(str(value))
        if match:
            resolved = os.environ.get(match.group(1))
            if resolved is None:
                raise ValueError(
                    f"LanceDB storage option {key!r} references environment variable "
                    f"{match.group(1)!r}, which is not set"
                )
            expanded[key] = resolved
        else:
            expanded[key] = value
    return expanded


class LanceDBCollectionAdapter(CollectionAdapter):
    """Adapter binding OpenViking's collection contract to a LanceDB table."""

    mode = "lancedb"

    def __init__(
        self,
        *,
        uri: str,
        table_name: str,
        index_name: str,
        storage_options: Optional[Dict[str, str]] = None,
        read_consistency_interval_ms: Optional[float] = None,
    ):
        super().__init__(collection_name=table_name, index_name=index_name)
        if not LANCEDB_AVAILABLE:
            raise ImportError(
                "The 'lancedb' package is required for the LanceDB backend. "
                "Install it with: pip install lancedb"
            )
        self._uri = uri
        self._storage_options = dict(storage_options or {})
        self._read_consistency_interval_ms = read_consistency_interval_ms
        self._db: Any = None
        self._collection: Optional[Collection] = None

    # -- construction -------------------------------------------------------

    @classmethod
    def from_config(cls, config: Any) -> "LanceDBCollectionAdapter":
        cfg = getattr(config, "lancedb", None)
        if cfg is None or not getattr(cfg, "uri", None):
            raise ValueError(
                "LanceDB backend requires 'storage.vectordb.lancedb.uri' to be set "
                "(local path or s3:// object-store prefix)"
            )
        sparse_weight = float(getattr(config, "sparse_weight", 0.0) or 0.0)
        if sparse_weight > 0.0:
            raise ValueError(
                "The LanceDB backend is dense-only: 'sparse_weight' must be 0. "
                "Sparse/hybrid retrieval is rejected instead of silently ignored."
            )
        storage_options = _expand_env_placeholders(getattr(cfg, "storage_options", {}) or {})
        return cls(
            uri=cfg.uri,
            table_name=config.name or "context",
            index_name=config.index_name or "default",
            storage_options=storage_options,
            read_consistency_interval_ms=getattr(cfg, "read_consistency_interval_ms", None),
        )

    # -- backend connection ---------------------------------------------------

    def _connect_db(self) -> Any:
        if self._db is None:
            import lancedb

            kwargs: Dict[str, Any] = {}
            if self._storage_options:
                kwargs["storage_options"] = self._storage_options
            if self._read_consistency_interval_ms is not None:
                kwargs["read_consistency_interval"] = self._read_consistency_interval_ms / 1000.0
            self._db = lancedb.connect(self._uri, **kwargs)
        return self._db

    def _schema_meta(self) -> Dict[str, Any]:
        from openviking.storage.collection_schemas import CollectionSchemas
        from openviking_cli.utils.config import get_openviking_config

        config = get_openviking_config()
        return CollectionSchemas.context_collection(
            self._collection_name,
            config.embedding.dimension,
        )

    def _open_collection(self) -> Optional[Collection]:
        db = self._connect_db()
        from openviking.storage.vectordb.collection.lancedb_collection import table_exists

        if not table_exists(db, self._collection_name):
            return None
        from openviking_cli.utils.config import get_openviking_config

        config = get_openviking_config()
        collection, _created = create_lancedb_collection(
            db=db,
            table_name=self._collection_name,
            meta_data={},
            dimension=config.embedding.dimension,
            metric=vectordb_metric(config),
        )
        return Collection(collection)

    # -- CollectionAdapter hooks -----------------------------------------------

    def _load_existing_collection_if_needed(self) -> None:
        if self._collection is not None:
            return
        self._collection = self._open_collection()

    def _create_backend_collection(self, meta: Dict[str, Any]) -> Collection:
        db = self._connect_db()
        from openviking_cli.utils.config import get_openviking_config

        config = get_openviking_config()
        collection, created = create_lancedb_collection(
            db=db,
            table_name=self._collection_name,
            meta_data=meta,
            dimension=config.embedding.dimension,
            metric=vectordb_metric(config),
        )
        if not created:
            logger = self._get_logger()
            logger.info(
                "Reusing existing LanceDB table %r instead of creating a duplicate",
                self._collection_name,
            )
        return Collection(collection)

    @staticmethod
    def _get_logger():
        from openviking_cli.utils import get_logger

        return get_logger(__name__)


def vectordb_metric(config: Any) -> str:
    metric = str(getattr(config.storage.vectordb, "distance_metric", "cosine") or "cosine")
    from openviking.storage.vectordb.collection.lancedb_collection import LANCEDB_METRICS

    if metric not in LANCEDB_METRICS:
        raise ValueError(
            f"LanceDB backend does not support distance metric {metric!r}; "
            f"supported metrics: {sorted(LANCEDB_METRICS)}"
        )
    return metric
