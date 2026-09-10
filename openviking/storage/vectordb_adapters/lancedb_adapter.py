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
from datetime import timedelta
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
        uri: Optional[str],
        table_name: str,
        index_name: str,
        storage_options: Optional[Dict[str, str]] = None,
        read_consistency_interval_ms: Optional[float] = None,
        vector_index: Optional[Dict[str, Any]] = None,
        scalar_indexes: Optional[Dict[str, str]] = None,
        optimize_after_bulk_ingest: bool = True,
        fts: Optional[Dict[str, Any]] = None,
        hybrid: Optional[Dict[str, Any]] = None,
        store_content: bool = False,
        namespace_uri: Optional[str] = None,
        namespace_path: Optional[list[str]] = None,
    ):
        super().__init__(collection_name=table_name, index_name=index_name)
        if not LANCEDB_AVAILABLE:
            raise ImportError(
                "The 'lancedb' package is required for the LanceDB backend. "
                "Install it with: pip install lancedb"
            )
        # ``content`` is persisted only when explicitly opted in; this drives
        # the upstream write path (see CollectionAdapter.USE_CONTENT_FIELD)
        # and the grep engine capability check.
        self.USE_CONTENT_FIELD = store_content
        self._uri = uri
        self._namespace_uri = namespace_uri
        self._namespace_path = list(namespace_path or [])
        self._storage_options = dict(storage_options or {})
        self._read_consistency_interval_ms = read_consistency_interval_ms
        self._vector_index = vector_index
        self._scalar_indexes = dict(scalar_indexes or {})
        self._optimize_after_bulk_ingest = optimize_after_bulk_ingest
        self._fts = fts
        self._hybrid = hybrid
        self._store_content = store_content
        self._db: Any = None
        self._collection: Optional[Collection] = None

    # -- construction -------------------------------------------------------

    @classmethod
    def from_config(cls, config: Any) -> "LanceDBCollectionAdapter":
        cfg = getattr(config, "lancedb", None)
        if cfg is None or not (getattr(cfg, "uri", None) or getattr(cfg, "namespace_uri", None)):
            raise ValueError(
                "LanceDB backend requires 'storage.vectordb.lancedb.uri' "
                "(or 'namespace_uri' + 'namespace_path') to be set"
            )
        if getattr(cfg, "namespace_uri", None) and not getattr(cfg, "namespace_path", None):
            raise ValueError(
                "LanceDB 'namespace_uri' requires 'namespace_path' "
                "(the first segment is the Lance table bucket)"
            )
        sparse_weight = float(getattr(config, "sparse_weight", 0.0) or 0.0)
        if sparse_weight > 0.0:
            raise ValueError(
                "The LanceDB backend is dense-only: 'sparse_weight' must be 0. "
                "Sparse/hybrid retrieval is rejected instead of silently ignored."
            )
        storage_options = _expand_env_placeholders(getattr(cfg, "storage_options", {}) or {})
        vector_index_cfg = getattr(cfg, "vector_index", None)
        fts_cfg = getattr(cfg, "fts", None)
        hybrid_cfg = getattr(cfg, "hybrid", None)
        return cls(
            uri=cfg.uri,
            table_name=config.name or "context",
            index_name=config.index_name or "default",
            storage_options=storage_options,
            read_consistency_interval_ms=getattr(cfg, "read_consistency_interval_ms", None),
            vector_index=vector_index_cfg.model_dump() if vector_index_cfg else None,
            scalar_indexes=getattr(cfg, "scalar_indexes", {}) or {},
            optimize_after_bulk_ingest=bool(getattr(cfg, "optimize_after_bulk_ingest", True)),
            fts=fts_cfg.model_dump() if fts_cfg else None,
            hybrid=hybrid_cfg.model_dump() if hybrid_cfg else None,
            store_content=bool(getattr(cfg, "store_content", False)),
            namespace_uri=getattr(cfg, "namespace_uri", None),
            namespace_path=getattr(cfg, "namespace_path", None) or [],
        )

    # -- backend connection ---------------------------------------------------

    def _connect_db(self) -> Any:
        if self._db is None:
            import lancedb

            kwargs: Dict[str, Any] = {}
            if self._storage_options:
                kwargs["storage_options"] = self._storage_options
            if self._read_consistency_interval_ms is not None:
                # lancedb expects a timedelta; a bare float fails deep in the
                # client with "'float' object has no attribute 'total_seconds'"
                # and the collection silently degrades to empty searches.
                kwargs["read_consistency_interval"] = timedelta(
                    milliseconds=self._read_consistency_interval_ms
                )
            if self._namespace_uri:
                # Lance Namespace REST catalog (e.g. SeaweedFS Lake):
                # tables are declared through the catalog while data is
                # written straight to the object store.
                kwargs.pop("read_consistency_interval", None)
                self._db = lancedb.connect_namespace(
                    "rest",
                    {"uri": self._namespace_uri},
                    **kwargs,
                )
            else:
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

        if not table_exists(db, self._collection_name, self._namespace_path):
            return None
        from openviking_cli.utils.config import get_openviking_config

        config = get_openviking_config()
        collection, _created = create_lancedb_collection(
            db=db,
            table_name=self._collection_name,
            meta_data={},
            dimension=config.embedding.dimension,
            metric=vectordb_metric(config),
            vector_index=self._vector_index,
            scalar_indexes=self._scalar_indexes,
            optimize_after_bulk_ingest=self._optimize_after_bulk_ingest,
            fts=self._fts,
            hybrid=self._hybrid,
            store_content=self._store_content,
            namespace_path=self._namespace_path,
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
            vector_index=self._vector_index,
            scalar_indexes=self._scalar_indexes,
            optimize_after_bulk_ingest=self._optimize_after_bulk_ingest,
            fts=self._fts,
            hybrid=self._hybrid,
            store_content=self._store_content,
            namespace_path=self._namespace_path,
        )
        if not created:
            logger = self._get_logger()
            logger.info(
                "Reusing existing LanceDB table %r instead of creating a duplicate",
                self._collection_name,
            )
        return Collection(collection)

    def begin_bulk_ingest(self) -> None:
        if self._collection is not None:
            self._collection.begin_bulk_ingest()

    def end_bulk_ingest(self) -> None:
        if self._collection is not None:
            self._collection.end_bulk_ingest()

    def search_by_keywords(
        self,
        keywords: Optional[list[str]] = None,
        query: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
        filter=None,
        output_fields: Optional[list[str]] = None,
        query_vector: Optional[list[float]] = None,
    ) -> list[dict[str, Any]]:
        """Keyword (BM25) search with optional dense+lexical hybrid fusion.

        ``query_vector`` is a LanceDB capability extension: when provided and
        hybrid search is configured, dense and lexical candidates are fused
        through the configured reranker.  Both branches share the same
        security filter.
        """
        coll = self.get_collection()
        compiled_filter = self._compile_filter(filter)
        result = coll.search_by_keywords(
            self._index_name,
            keywords=keywords,
            query=query,
            limit=limit,
            offset=offset,
            filters=compiled_filter,
            output_fields=output_fields,
            dense_vector=query_vector,
        )
        records: list[dict[str, Any]] = []
        for item in result.data:
            record = dict(item.fields) if item.fields else {}
            record["id"] = item.id
            raw_score = item.score if item.score is not None else 0.0
            if raw_score != raw_score:  # NaN guard
                raw_score = 0.0
            record["_score"] = raw_score
            records.append(self._normalize_record_for_read(record))
        return records

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
