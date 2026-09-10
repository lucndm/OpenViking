# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""LanceDB-backed ``ICollection`` implementation for OpenViking.

This module maps OpenViking's collection contract (schema DSL, filter DSL,
CRUD, and dense search) onto a LanceDB table.  Dense retrieval is the only
search mode in this phase; sparse/hybrid configuration must fail fast rather
than be silently ignored.

Filter translation converts the OpenViking filter DSL (``and``/``or``/
``must``/``must_not``/``range``/``range_out``/``contains`` plus path depth
parameters) into DataFusion SQL predicates.  The translator is deliberately
pure so it can be unit-tested without a LanceDB installation.
"""

from __future__ import annotations

import json
import random
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openviking.storage.vectordb.collection.collection import ICollection
from openviking.storage.vectordb.collection.result import (
    AggregateResult,
    DataItem,
    FetchDataInCollectionResult,
    SearchItemResult,
    SearchResult,
)
from openviking_cli.utils import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - exercised implicitly by import-guard tests
    import lancedb  # type: ignore
    from lancedb.query import ColumnOrdering  # type: ignore

    LANCEDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    lancedb = None  # type: ignore
    ColumnOrdering = None  # type: ignore
    LANCEDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VECTOR_FIELD_NAME = "vector"
ID_FIELD_NAME = "id"

# OpenViking distance metric -> LanceDB metric.
LANCEDB_METRICS: Dict[str, str] = {"cosine": "cosine", "l2": "l2", "ip": "dot"}

# Durable collection metadata is stored as field metadata on the ``id`` column
# so it survives restarts and object-store backends without sidecar files.
_META_COLUMN = ID_FIELD_NAME
_META_DESCRIPTION = "ov.description"
_META_INDEX = "ov.index"
_META_DIMENSION = "ov.dim"
_META_METRIC = "ov.metric"
_META_FIELD_TYPES = "ov.field_types"
_META_SCALAR_INDEX = "ov.scalar_index"
_META_FTS = "ov.fts"
_META_HYBRID = "ov.hybrid"
_META_STORE_CONTENT = "ov.store_content"

_PATH_DEPTH_RE = re.compile(r"^\s*-d=(-?\d+)\s*$")


# ---------------------------------------------------------------------------
# SQL literal helpers (pure functions, unit-testable without lancedb)
# ---------------------------------------------------------------------------


def sql_string_literal(value: Any) -> str:
    """Render *value* as a single-quoted SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def sql_string_array(values: Sequence[Any]) -> str:
    """Render *values* as a SQL array of string literals."""
    return "[" + ", ".join(sql_string_literal(v) for v in values) + "]"


def _escape_like(substring: str) -> str:
    """Escape LIKE wildcards inside *substring* so it matches literally."""
    escaped = substring.replace("\\", "\\\\")
    escaped = escaped.replace("%", "\\%").replace("_", "\\_")
    return escaped


def _iso_to_timestamp_literal(value: Any) -> str:
    """Normalize an ISO-8601 string / datetime into a SQL TIMESTAMP literal."""
    if isinstance(value, datetime):
        normalized = value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    else:
        text = str(value).strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid date_time filter value: {value!r}") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        normalized = parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"TIMESTAMP '{normalized}'"


def _sql_literal(field_type: str, value: Any) -> str:
    """Render *value* as a SQL literal appropriate for *field_type*."""
    if field_type == "date_time":
        return _iso_to_timestamp_literal(value)
    if field_type in {"int64", "int32", "float", "double"} and isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return sql_string_literal(value)


def _regex_escape(text: str) -> str:
    """Escape *text* for safe interpolation into a RE2 regular expression."""
    return re.escape(text).replace("\\:", ":")


def _path_scope_sql(field: str, conds: Sequence[Any], depth: Optional[int], field_type: str) -> str:
    """Translate a path-scope predicate into SQL.

    Path values are stored as ``/a/b`` strings.  ``depth`` semantics match
    ``PathScope``: ``-1`` (or None) matches any depth, ``0`` matches the exact
    path only, ``N > 0`` additionally matches up to *N* path components below
    the prefix.
    """
    column = _sql_column(field)
    clauses: List[str] = []
    for cond in conds:
        prefix = str(cond)
        if field_type == "path" and not prefix.startswith("/"):
            prefix = "/" + prefix.lstrip()
        prefix = prefix.rstrip("/") or "/"
        exact = f"{column} = {sql_string_literal(prefix)}"
        if depth is None or depth < 0:
            clauses.append(
                f"({exact} OR starts_with({column}, {sql_string_literal(prefix + '/')}))"
            )
        elif depth == 0:
            clauses.append(f"({exact})")
        else:
            # Bounded repetition ``{0,N}`` is unreliable in the bundled regex
            # engine, so allowed depths are expanded into an explicit
            # alternation (exact, +1 component, ..., +depth components).
            escaped = _regex_escape(prefix)
            parts = [escaped] + [escaped + "/[^/]+" * i for i in range(1, depth + 1)]
            pattern = "^(" + "|".join(parts) + ")$"
            # ``regexp_match`` yields an empty array (not NULL) on no-match,
            # so truthiness must go through array_length.
            clauses.append(
                f"(array_length(regexp_match({column}, {sql_string_literal(pattern)}), 1) > 0)"
            )
    if not clauses:
        return "false"
    return "(" + " OR ".join(clauses) + ")"


def _sql_column(field: str) -> str:
    """Quote a column identifier for SQL."""
    return '"' + field.replace('"', '""') + '"'


def translate_filter_to_sql(
    node: Optional[Dict[str, Any]],
    field_types: Dict[str, str],
) -> Optional[str]:
    """Translate the OpenViking filter DSL into a DataFusion SQL predicate.

    Returns ``None`` when the filter is empty (no restriction).  Raises
    ``ValueError`` for malformed nodes and ``NotImplementedError`` for
    operators outside the supported subset.
    """
    if not node:
        return None
    if not isinstance(node, dict):
        raise ValueError(f"Filter node must be an object: {node!r}")

    if "filter" in node and len(node) == 1:
        return translate_filter_to_sql(node.get("filter"), field_types)

    op = str(node.get("op", "")).lower()

    if op in {"and", "or"}:
        children = node.get("conds", [])
        if not isinstance(children, list):
            raise ValueError(f"'{op}' filter conds must be a list")
        translated = [
            translate_filter_to_sql(child, field_types)
            for child in children
            if child not in (None, {}, "")
        ]
        translated = [part for part in translated if part]
        if not translated:
            return None
        joiner = " AND " if op == "and" else " OR "
        if len(translated) == 1:
            return translated[0]
        return "(" + joiner.join(translated) + ")"

    field = node.get("field")
    if not isinstance(field, str):
        raise ValueError(f"Filter field must be a string: {node!r}")
    field_type = str(field_types.get(field, "string")).lower()
    column = _sql_column(field)
    is_list = field_type.startswith("list<")

    if op in {"must", "must_not"}:
        conds = node.get("conds", [])
        if not isinstance(conds, list) or not conds:
            raise ValueError(f"'{op}' filter conds must be a non-empty list: {node!r}")

        if field_type == "path":
            depth = _parse_path_depth(node.get("para"))
            matched = _path_scope_sql(field, conds, depth, field_type)
        else:
            if node.get("para") not in (None, ""):
                raise ValueError(f"Filter parameters are only supported for path fields: {node!r}")
            if is_list:
                matched = f"array_has_any({column}, {sql_string_array(conds)})"
            elif len(conds) == 1:
                matched = f"{column} = {_sql_literal(field_type, conds[0])}"
            else:
                literals = ", ".join(_sql_literal(field_type, c) for c in conds)
                matched = f"{column} IN ({literals})"
        if op == "must":
            return f"({matched})"
        # ``must_not`` keeps rows with absent/null values visible, matching
        # the reference semantics used for ACL mode filters.
        return f"({column} IS NULL OR NOT ({matched}))"

    if op == "contains":
        substring = node.get("substring")
        if not isinstance(substring, str):
            raise ValueError(f"'contains' filter substring must be a string: {node!r}")
        pattern = sql_string_literal("%" + _escape_like(substring) + "%")
        if is_list:
            body = f"array_to_string({column}, {sql_string_literal(chr(1))}) LIKE {pattern}"
        else:
            body = f"{column} LIKE {pattern}"
        return f"({body})"

    if op in {"range", "range_out"}:
        clauses: List[str] = []
        for key, sql_op in (("gt", ">"), ("gte", ">="), ("lt", "<"), ("lte", "<=")):
            if node.get(key) is not None:
                clauses.append(f"{column} {sql_op} {_sql_literal(field_type, node[key])}")
        if not clauses:
            return None
        body = " AND ".join(clauses)
        if op == "range":
            return f"({body})"
        return f"(NOT ({body}))"

    raise NotImplementedError(f"Unsupported filter operation for LanceDB: {op!r}")


def _parse_path_depth(para: Any) -> Optional[int]:
    if para in (None, ""):
        return None
    if isinstance(para, int):
        return para
    match = _PATH_DEPTH_RE.match(str(para))
    if not match:
        raise ValueError(f"Unsupported path filter parameter: {para!r}")
    return int(match.group(1))


# ---------------------------------------------------------------------------
# Schema translation
# ---------------------------------------------------------------------------

_ARROW_TYPES: Dict[str, Any] = {}


def build_arrow_schema(meta: Dict[str, Any]):
    """Translate an OpenViking collection schema dict into a pyarrow schema.

    ``sparse_vector`` fields are dropped: this backend is dense-only and a
    sparse column would waste storage while implying unsupported capability.
    """
    import pyarrow as pa

    fields: List[Any] = []
    for field in meta.get("Fields", []):
        name = field.get("FieldName")
        ftype = str(field.get("FieldType", "string")).lower()
        if ftype == "sparse_vector":
            continue
        nullable = not field.get("IsPrimaryKey")
        if ftype == "vector":
            dim = field.get("Dim")
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(f"Vector field {name!r} requires a positive 'Dim'")
            arrow_type = pa.list_(pa.float32(), dim)
        elif ftype in {"string", "text", "path"}:
            arrow_type = pa.string()
        elif ftype in {"int64", "int"}:
            arrow_type = pa.int64()
        elif ftype in {"int32"}:
            arrow_type = pa.int32()
        elif ftype in {"float", "double"}:
            arrow_type = pa.float64()
        elif ftype == "bool":
            arrow_type = pa.bool_()
        elif ftype.startswith("list<"):
            arrow_type = pa.list_(pa.string())
        elif ftype == "date_time":
            arrow_type = pa.timestamp("ms")
        else:
            raise ValueError(
                f"Field {name!r} has unsupported type {ftype!r} for the LanceDB backend"
            )
        fields.append(pa.field(name, arrow_type, nullable=nullable))
    if not any(field.name == ID_FIELD_NAME for field in fields):
        raise ValueError("LanceDB collection schema requires an 'id' field")
    return pa.schema(fields)


def field_types_from_meta(meta: Dict[str, Any]) -> Dict[str, str]:
    """Extract ``field -> OpenViking FieldType`` from a collection schema dict."""
    return {
        str(field.get("FieldName")): str(field.get("FieldType", "string")).lower()
        for field in meta.get("Fields", [])
        if field.get("FieldName")
    }


# ---------------------------------------------------------------------------
# Collection implementation
# ---------------------------------------------------------------------------


class LanceDBCollection(ICollection):
    """``ICollection`` implementation backed by a single LanceDB table."""

    def __init__(
        self,
        *,
        db: Any,
        table_name: str,
        meta_data: Optional[Dict[str, Any]] = None,
        dimension: Optional[int] = None,
        metric: str = "cosine",
        vector_index: Optional[Dict[str, Any]] = None,
        scalar_indexes: Optional[Dict[str, str]] = None,
        optimize_after_bulk_ingest: bool = True,
        fts: Optional[Dict[str, Any]] = None,
        hybrid: Optional[Dict[str, Any]] = None,
        store_content: bool = False,
        namespace_path: Optional[List[str]] = None,
    ):
        super().__init__()
        if not LANCEDB_AVAILABLE:  # pragma: no cover - defensive
            raise ImportError(
                "The 'lancedb' package is required for the LanceDB backend. "
                "Install it with: pip install lancedb"
            )
        self._db = db
        self._table_name = table_name
        self._dimension = dimension
        self._metric = LANCEDB_METRICS.get(metric, "cosine")
        self._ov_metric = metric
        self._table = None
        self._field_types: Dict[str, str] = {}
        self._index_metas: Dict[str, Dict[str, Any]] = {}
        self._meta_data: Dict[str, Any] = dict(meta_data or {})
        self._vector_index_cfg: Optional[Dict[str, Any]] = (
            dict(vector_index) if vector_index else None
        )
        self._scalar_index_cfg: Dict[str, str] = dict(scalar_indexes or {})
        self._optimize_after_bulk_ingest = optimize_after_bulk_ingest
        self._fts_cfg: Optional[Dict[str, Any]] = dict(fts) if fts else None
        self._hybrid_cfg: Optional[Dict[str, Any]] = dict(hybrid) if hybrid else None
        self._store_content = store_content
        self._namespace_path: List[str] = list(namespace_path or [])
        self._bulk_ingest_depth = 0
        self._maintenance_runs = 0
        self._table: Any = None
        self._load_or_create_table(create=False)

    # -- table lifecycle ---------------------------------------------------

    @staticmethod
    def _read_field_metadata(table: Any) -> Dict[str, str]:
        try:
            stored = table.schema.field(_META_COLUMN).metadata or {}
        except (KeyError, AttributeError, TypeError):
            return {}
        return {key.decode("utf-8"): value.decode("utf-8") for key, value in stored.items()}

    def _table_open_kwargs(self) -> Dict[str, Any]:
        if self._namespace_path:
            return {"namespace_path": self._namespace_path}
        return {}

    def _load_or_create_table(self, create: bool, arrow_schema: Any = None) -> None:
        kwargs = self._table_open_kwargs()
        if table_exists(self._db, self._table_name, self._namespace_path):
            self._table = self._db.open_table(self._table_name, **kwargs)
            self._restore_metadata()
            self._validate_dimension()
        elif create:
            if arrow_schema is None:
                raise ValueError("Creating a LanceDB table requires a schema")
            self._table = self._db.create_table(self._table_name, schema=arrow_schema, **kwargs)
            self._persist_metadata()
        else:
            self._table = None

    def _restore_metadata(self) -> None:
        stored = self._read_field_metadata(self._table)
        self._field_types = (
            json.loads(stored[_META_FIELD_TYPES])
            if _META_FIELD_TYPES in stored
            else self._infer_field_types_from_arrow()
        )
        if _META_INDEX in stored:
            try:
                self._index_metas = json.loads(stored[_META_INDEX])
            except json.JSONDecodeError:
                logger.warning("LanceDB index metadata is corrupt; ignoring it")
                self._index_metas = {}
        if _META_DIMENSION in stored:
            # Adopt the stored dimension only when none was configured; an
            # explicit configured dimension is validated against the stored
            # schema below instead of being silently overwritten.
            if self._dimension is None:
                self._dimension = int(stored[_META_DIMENSION])
        if _META_METRIC in stored:
            stored_metric = stored[_META_METRIC]
            if stored_metric in LANCEDB_METRICS:
                self._metric = LANCEDB_METRICS[stored_metric]
                self._ov_metric = stored_metric
        if _META_FTS in stored and self._fts_cfg is None:
            try:
                self._fts_cfg = json.loads(stored[_META_FTS])
            except json.JSONDecodeError:
                logger.warning("LanceDB FTS metadata is corrupt; ignoring it")
        if _META_HYBRID in stored and self._hybrid_cfg is None:
            try:
                self._hybrid_cfg = json.loads(stored[_META_HYBRID])
            except json.JSONDecodeError:
                logger.warning("LanceDB hybrid metadata is corrupt; ignoring it")
        if _META_STORE_CONTENT in stored:
            self._store_content = stored[_META_STORE_CONTENT] == "true"

    def _infer_field_types_from_arrow(self) -> Dict[str, str]:
        import pyarrow as pa

        types: Dict[str, str] = {}
        for field in self._table.schema:
            arrow_type = field.type
            if field.name == VECTOR_FIELD_NAME:
                types[field.name] = "vector"
            elif pa.types.is_list(arrow_type):
                types[field.name] = "list<string>"
            elif pa.types.is_timestamp(arrow_type):
                types[field.name] = "date_time"
            elif pa.types.is_integer(arrow_type):
                types[field.name] = "int64"
            elif pa.types.is_floating(arrow_type):
                types[field.name] = "float"
            elif pa.types.is_boolean(arrow_type):
                types[field.name] = "bool"
            else:
                types[field.name] = "string"
        return types

    def _persist_metadata(self) -> None:
        if self._table is None:
            return
        payload: Dict[str, str] = {
            _META_FIELD_TYPES: json.dumps(self._field_types),
            _META_METRIC: self._ov_metric,
        }
        if self._dimension is not None:
            payload[_META_DIMENSION] = str(self._dimension)
        if self._meta_data.get("Description"):
            payload[_META_DESCRIPTION] = str(self._meta_data["Description"])
        if self._index_metas:
            payload[_META_INDEX] = json.dumps(self._index_metas)
        scalar_index = self._meta_data.get("ScalarIndex")
        if scalar_index:
            payload[_META_SCALAR_INDEX] = json.dumps(list(scalar_index))
        if self._fts_cfg:
            payload[_META_FTS] = json.dumps(self._fts_cfg)
        if self._hybrid_cfg:
            payload[_META_HYBRID] = json.dumps(self._hybrid_cfg)
        if self._store_content:
            payload[_META_STORE_CONTENT] = "true"
        try:
            self._table.replace_field_metadata(_META_COLUMN, payload)
        except Exception as exc:  # pragma: no cover - backend dependent
            logger.warning("Failed to persist LanceDB collection metadata: %s", exc)

    def _validate_dimension(self) -> None:
        vector_field = None
        for field in self._table.schema:
            if field.name == VECTOR_FIELD_NAME:
                vector_field = field
                break
        if vector_field is None:
            raise ValueError(
                f"LanceDB table {self._table_name!r} has no '{VECTOR_FIELD_NAME}' column; "
                "it is not an OpenViking collection table"
            )
        stored_dim = getattr(vector_field.type, "list_size", -1)
        if self._dimension is not None and stored_dim != self._dimension:
            raise ValueError(
                f"LanceDB table {self._table_name!r} was created with vector dimension "
                f"{stored_dim}, but dimension {self._dimension} is configured. "
                "Recreate the collection or reindex with the current embedding model."
            )

    # -- ICollection: metadata ---------------------------------------------

    def get_meta_data(self) -> Dict[str, Any]:
        fields: List[Dict[str, Any]] = []
        for field in self._table.schema:
            arrow_type = field.type
            if field.name == VECTOR_FIELD_NAME:
                fields.append(
                    {
                        "FieldName": field.name,
                        "FieldType": "vector",
                        "Dim": getattr(arrow_type, "list_size", self._dimension or 0),
                    }
                )
                continue
            ov_type = self._field_types.get(field.name, "string")
            entry: Dict[str, Any] = {"FieldName": field.name, "FieldType": ov_type}
            if field.name == ID_FIELD_NAME:
                entry["IsPrimaryKey"] = True
            fields.append(entry)
        meta: Dict[str, Any] = {
            "CollectionName": self._table_name,
            "Fields": fields,
        }
        stored = self._read_field_metadata(self._table)
        if _META_DESCRIPTION in stored:
            meta["Description"] = stored[_META_DESCRIPTION]
        scalar_index = (
            json.loads(stored[_META_SCALAR_INDEX])
            if _META_SCALAR_INDEX in stored
            else self._meta_data.get("ScalarIndex", [])
        )
        if scalar_index:
            meta["ScalarIndex"] = scalar_index
        if self._fts_cfg:
            # Report the same FullText shape the grep engine resolver uses
            # to decide whether server-side full-text grep is available.
            meta["FullText"] = [
                {"Field": column, "Analyzer": {"Tokenizer": "standard"}}
                for column in self._fts_cfg.get("columns", ["content"])
            ]
        return meta

    def update(self, fields: Optional[Dict[str, Any]] = None, description: Optional[str] = None):
        if description is not None:
            self._meta_data["Description"] = description
        self._persist_metadata()

    def close(self):
        self._table = None

    def drop(self):
        self._db.drop_table(self._table_name, **self._table_open_kwargs())
        self._table = None

    # -- ICollection: index lifecycle ----------------------------------------

    def create_index(self, index_name: str, meta_data: Dict[str, Any]) -> Any:
        vector_index = meta_data.get("VectorIndex", {}) if isinstance(meta_data, dict) else {}
        if vector_index.get("EnableSparse"):
            raise NotImplementedError(
                "The LanceDB backend is dense-only; sparse/hybrid index "
                "configuration is not supported and must not be silently ignored. "
                "Set storage.vectordb.sparse_weight to 0."
            )
        index_type = str(vector_index.get("IndexType", "flat")).lower()
        if index_type not in {"flat", "flat_hybrid", "ivf_pq", "hnsw", "ivf"}:
            raise NotImplementedError(
                f"Index type {index_type!r} is not supported by the LanceDB backend"
            )
        self._index_metas[index_name] = dict(meta_data)
        self._persist_metadata()
        self._apply_native_indexes(index_name)
        return self.get_index(index_name)

    # -- native Lance index management ---------------------------------------

    def _native_index_names(self) -> List[str]:
        table = self._ensure_table()
        try:
            return [idx.name for idx in (table.list_indices() or [])]
        except Exception as exc:
            logger.debug("LanceDB list_indices failed: %s", exc)
            return []

    def _native_index_stats(self, name: str) -> Dict[str, Any]:
        table = self._ensure_table()
        try:
            stats = table.index_stats(name)
        except Exception as exc:
            logger.debug("LanceDB index_stats(%r) failed: %s", name, exc)
            return {}
        if stats is None:
            return {}
        payload: Dict[str, Any] = {"state": "built"}
        for attr in ("num_indexed_rows", "num_unindexed_rows", "index_type", "distance_type"):
            value = getattr(stats, attr, None)
            if value is not None:
                payload[attr] = value
        size = getattr(stats, "size_bytes", None)
        if size is not None:
            payload["size_bytes"] = int(size)
        return payload

    def _scalar_index_factory(self, index_type: str) -> Any:
        from lancedb.index import Bitmap, BTree, LabelList

        factories = {"BTREE": BTree, "BITMAP": Bitmap, "LABEL_LIST": LabelList}
        factory = factories.get(index_type.upper())
        if factory is None:
            raise ValueError(
                f"Unsupported LanceDB scalar index type {index_type!r}; "
                f"supported: {sorted(factories)}"
            )
        return factory()

    def _apply_native_indexes(self, index_name: str) -> None:
        """Create configured native indexes for the current table state.

        Vector (ANN) training is deferred until the table reaches the
        configured row count; deferred indexes are reported as ``pending``
        and built by bulk-ingest completion or an explicit optimize call.
        Search remains correct in the meantime because unindexed fragments
        are scanned flat.
        """
        table = self._ensure_table()

        for field, index_type in self._scalar_index_cfg.items():
            if field not in self._field_types:
                logger.warning("LanceDB scalar index on unknown field %r is skipped", field)
                continue
            config = self._scalar_index_factory(index_type)
            table.create_index(field, config=config)
            logger.info("LanceDB scalar index created on %s (%s)", field, index_type)

        if self._fts_cfg:
            self._ensure_fts_index()

        if self._vector_index_cfg:
            self._maybe_build_vector_index(index_name, force=False)

    def _ensure_fts_index(self) -> None:
        """Create FTS indexes once; later writes stay visible via flat scans."""
        table = self._ensure_table()
        cfg = self._fts_cfg or {}
        columns = list(cfg.get("columns") or ["content"])
        for column in columns:
            if column not in self._field_types:
                raise ValueError(
                    f"LanceDB FTS column {column!r} is not part of the collection schema"
                )
        existing_fts = {
            idx.name
            for idx in (table.list_indices() or [])
            if str(getattr(idx, "index_type", "")).upper() == "FTS"
        }
        if existing_fts:
            return
        for column in columns:
            table.create_fts_index(
                column,
                language=cfg.get("language", "English"),
                with_position=bool(cfg.get("with_position", False)),
                stem=bool(cfg.get("stem", True)),
                remove_stop_words=bool(cfg.get("remove_stop_words", True)),
                ascii_folding=bool(cfg.get("ascii_folding", True)),
                custom_stop_words=cfg.get("custom_stop_words"),
                replace=False,
            )
        logger.info("LanceDB FTS index created on %s", columns)

    def _maybe_build_vector_index(self, index_name: str, *, force: bool) -> bool:
        from lancedb.index import (
            IvfFlat,
            IvfHnswFlat,
            IvfHnswPq,
            IvfHnswSq,
            IvfPq,
            IvfSq,
        )

        table = self._ensure_table()
        native_names = set(self._native_index_names())
        if index_name in native_names and not force:
            return False

        cfg = self._vector_index_cfg or {}
        min_rows = int(cfg.get("min_rows_to_build", 256))
        row_count = table.count_rows()
        if not force and row_count < min_rows:
            logger.info(
                "LanceDB vector index %r deferred: %d rows < %d required to train; "
                "flat scan keeps search correct meanwhile",
                index_name,
                row_count,
                min_rows,
            )
            return False

        factories = {
            "IVF_PQ": IvfPq,
            "IVF_FLAT": IvfFlat,
            "IVF_SQ": IvfSq,
            "IVF_HNSW_PQ": IvfHnswPq,
            "IVF_HNSW_SQ": IvfHnswSq,
            "IVF_HNSW_FLAT": IvfHnswFlat,
        }
        index_type = str(cfg.get("index_type", "IVF_PQ")).upper()
        factory = factories.get(index_type)
        if factory is None:
            raise ValueError(
                f"Unsupported LanceDB vector index type {index_type!r}; "
                f"supported: {sorted(factories)}"
            )
        kwargs: Dict[str, Any] = {"distance_type": self._metric}
        for key in ("num_partitions", "num_sub_vectors"):
            if cfg.get(key) is not None:
                kwargs[key] = cfg[key]
        if index_type.endswith("PQ") and cfg.get("num_bits") is not None:
            kwargs["num_bits"] = cfg["num_bits"]
        table.create_index(
            VECTOR_FIELD_NAME, config=factory(**kwargs), name=index_name, replace=True
        )
        logger.info(
            "LanceDB vector index %r built (%s, %d rows)",
            index_name,
            index_type,
            row_count,
        )
        return True

    def optimize_index(self, index_name: Optional[str] = None) -> Dict[str, Any]:
        """Explicitly rebuild/refresh native indexes and compact fragments."""
        table = self._ensure_table()
        target = index_name or next(iter(self._index_metas), None)
        report: Dict[str, Any] = {}
        if target and self._vector_index_cfg:
            report["vector_index_built"] = self._maybe_build_vector_index(target, force=True)
        table.optimize()
        report["maintenance_runs"] = self._maintenance_runs + 1
        self._maintenance_runs += 1
        return report

    def has_index(self, index_name: str) -> bool:
        return index_name in self._index_metas or index_name in set(self._native_index_names())

    def get_index(self, index_name: str) -> Optional[Any]:
        if not self.has_index(index_name):
            return None
        return self._index_metas.get(index_name, {"IndexName": index_name})

    def get_index_meta_data(self, index_name: str) -> Dict[str, Any]:
        meta = dict(self._index_metas.get(index_name, {}))
        native_names = set(self._native_index_names())
        if index_name in native_names:
            meta.update(self._native_index_stats(index_name))
            meta.setdefault("state", "built")
        elif meta:
            meta["state"] = "pending"
        for field in self._scalar_index_cfg:
            if f"{field}_idx" in native_names:
                meta.setdefault("ScalarIndex", []).append(field)
        return meta

    def list_indexes(self) -> List[str]:
        names = set(self._index_metas.keys())
        names.update(self._native_index_names())
        return sorted(names)

    def drop_index(self, index_name: str):
        table = self._ensure_table()
        if index_name in self._native_index_names():
            table.drop_index(index_name)
        self._index_metas.pop(index_name, None)
        self._persist_metadata()

    def update_index(
        self,
        index_name: str,
        scalar_index: Optional[List[str]] = None,
        description: Optional[str] = None,
    ):
        meta = self._index_metas.setdefault(index_name, {})
        if scalar_index is not None:
            meta["ScalarIndex"] = list(scalar_index)
        if description is not None:
            meta["Description"] = description
        self._persist_metadata()

    # -- bulk ingest maintenance ----------------------------------------------

    def begin_bulk_ingest(self) -> None:
        """Suspend index maintenance until the matching end_bulk_ingest."""
        self._bulk_ingest_depth += 1

    def end_bulk_ingest(self) -> None:
        """Resume index maintenance, coalescing the whole scope into one pass.

        N batches inside one scope must not trigger N maintenance actions:
        compaction and index optimization run exactly once at scope exit.
        """
        if self._bulk_ingest_depth > 0:
            self._bulk_ingest_depth -= 1
        if self._bulk_ingest_depth > 0:
            return
        if not self._optimize_after_bulk_ingest or self._table is None:
            return
        self._run_coalesced_maintenance()

    def _run_coalesced_maintenance(self) -> None:
        table = self._ensure_table()
        if self._vector_index_cfg:
            target = next(iter(self._index_metas), None)
            if target:
                self._maybe_build_vector_index(target, force=False)
        table.optimize()
        self._maintenance_runs += 1
        logger.info("LanceDB coalesced bulk-ingest maintenance completed")

    @property
    def maintenance_runs(self) -> int:
        """Number of coalesced maintenance passes executed (observability)."""
        return self._maintenance_runs

    # -- data write path -----------------------------------------------------

    def _coerce_row_for_write(self, record: Dict[str, Any]) -> Dict[str, Any]:
        row: Dict[str, Any] = {}
        for key, value in record.items():
            if not self._field_types or key in self._field_types:
                ftype = self._field_types.get(key)
            else:
                # Unknown fields are dropped so writes cannot blow past the
                # stored schema; this mirrors the local backend behavior.
                continue
            if ftype == "sparse_vector" or key == "sparse_vector":
                continue
            if value is None:
                continue
            if ftype == "date_time":
                if isinstance(value, str):
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    value = parsed
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    value = datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
            row[key] = value
        if ID_FIELD_NAME not in row or not row[ID_FIELD_NAME]:
            raise ValueError("LanceDB upsert requires a non-empty 'id' field")
        return row

    def _ensure_table(self) -> Any:
        if self._table is None:
            raise RuntimeError(
                f"LanceDB table {self._table_name!r} is not open; the collection may have been dropped"
            )
        return self._table

    def upsert_data(self, data_list: List[Dict[str, Any]], ttl=0):
        table = self._ensure_table()
        rows = [self._coerce_row_for_write(record) for record in data_list]
        if not rows:
            return
        if ttl:
            logger.warning("LanceDB backend ignores upsert ttl=%s; TTL is not supported", ttl)
        rows = self._align_row_columns(rows)
        table.merge_insert(
            ID_FIELD_NAME
        ).when_matched_update_all().when_not_matched_insert_all().execute(rows)
        self._maybe_autobuild_vector_index()

    def _maybe_autobuild_vector_index(self) -> None:
        """Build a deferred ANN index once the row threshold is first reached.

        Outside bulk scopes a pending index is built at most once: after the
        native index exists this returns without rebuilding.  Inside a bulk
        scope the coalesced maintenance pass at scope exit owns the build.
        """
        if self._bulk_ingest_depth > 0 or not self._vector_index_cfg:
            return
        target = next(iter(self._index_metas), None)
        if target:
            self._maybe_build_vector_index(target, force=False)

    @staticmethod
    def _align_row_columns(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Give every row the same key set.

        ``merge_insert`` can drop columns that are absent from the first row
        of a heterogeneous batch, so missing keys are filled with explicit
        ``None`` before the merge.
        """
        if len(rows) <= 1:
            return rows
        keys: Dict[str, None] = {}
        for row in rows:
            keys.update(dict.fromkeys(row.keys()))
        return [{key: row.get(key) for key in keys} for row in rows]

    def update_data(self, data_list: List[Dict[str, Any]]):
        table = self._ensure_table()
        ids = [record.get(ID_FIELD_NAME) for record in data_list if record.get(ID_FIELD_NAME)]
        if not ids:
            return {"updated": 0, "primary_keys": []}
        existing = {row[ID_FIELD_NAME]: row for row in self._fetch_rows(table, ids)}
        merged_rows: List[Dict[str, Any]] = []
        for record in data_list:
            record_id = record.get(ID_FIELD_NAME)
            if record_id is None or record_id not in existing:
                continue
            merged = dict(existing[record_id])
            merged.update(record)
            merged_rows.append(self._coerce_row_for_write(merged))
        if merged_rows:
            table.merge_insert(ID_FIELD_NAME).when_matched_update_all().execute(merged_rows)
        return {"updated": len(merged_rows), "primary_keys": ids}

    def _fetch_rows(self, table: Any, ids: Sequence[Any]) -> List[Dict[str, Any]]:
        literals = ", ".join(sql_string_literal(value) for value in ids)
        rows = (
            table.search()
            .where(f"{_sql_column(ID_FIELD_NAME)} IN ({literals})", prefilter=True)
            .limit(len(ids))
            .to_list()
        )
        return self._normalize_rows(rows)

    def fetch_data(self, primary_keys: List[Any]) -> FetchDataInCollectionResult:
        table = self._ensure_table()
        if not primary_keys:
            return FetchDataInCollectionResult()
        rows = self._fetch_rows(table, primary_keys)
        found = {row[ID_FIELD_NAME]: row for row in rows}
        items: List[DataItem] = []
        missing: List[Any] = []
        for key in primary_keys:
            if key in found:
                fields = {k: v for k, v in found[key].items() if k != ID_FIELD_NAME}
                items.append(DataItem(id=key, fields=fields))
            else:
                missing.append(key)
        return FetchDataInCollectionResult(items=items, ids_not_exist=missing)

    def delete_data(self, primary_keys: List[Any]):
        table = self._ensure_table()
        if not primary_keys:
            return
        literals = ", ".join(sql_string_literal(value) for value in primary_keys)
        table.delete(f"{_sql_column(ID_FIELD_NAME)} IN ({literals})")

    def delete_all_data(self):
        table = self._ensure_table()
        table.delete("true")

    # -- search helpers ------------------------------------------------------

    def _normalize_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for row in rows:
            item = {}
            for key, value in row.items():
                if key.startswith("_"):
                    continue
                if isinstance(value, datetime):
                    value = value.astimezone(timezone.utc).isoformat()
                item[key] = value
            normalized.append(item)
        return normalized

    def _validate_query_vector(self, dense_vector: Optional[List[float]]) -> List[float]:
        if not dense_vector:
            raise ValueError("LanceDB dense search requires a query vector")
        dim = self._dimension or getattr(
            next(
                (f.type for f in self._table.schema if f.name == VECTOR_FIELD_NAME),
                None,
            ),
            "list_size",
            None,
        )
        if dim is not None and len(dense_vector) != dim:
            raise ValueError(
                f"Query vector dimension {len(dense_vector)} does not match collection "
                f"dimension {dim}; refusing to search with mismatched embeddings"
            )
        return [float(v) for v in dense_vector]

    def _score_from_distance(self, distance: float) -> float:
        # Higher score means more similar, matching other OpenViking backends.
        if self._metric == "l2":
            return -float(distance)
        return 1.0 - float(distance)

    def _rows_to_search_result(
        self,
        rows: List[Dict[str, Any]],
        output_fields: Optional[List[str]],
        score_getter: Any = None,
    ) -> SearchResult:
        result = SearchResult()
        for row in rows:
            fields = {k: v for k, v in row.items() if not k.startswith("_") and k != ID_FIELD_NAME}
            if output_fields is not None:
                fields = {k: v for k, v in fields.items() if k in output_fields}
            score = score_getter(row) if score_getter else 0.0
            result.data.append(
                SearchItemResult(id=row.get(ID_FIELD_NAME), fields=fields, score=score)
            )
        return result

    def _filter_sql(self, filters: Optional[Dict[str, Any]]) -> Optional[str]:
        return translate_filter_to_sql(filters, self._field_types)

    # -- ICollection: search --------------------------------------------------

    def search_by_vector(
        self,
        index_name: str,
        dense_vector: Optional[List[float]] = None,
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        sparse_vector: Optional[Dict[str, float]] = None,
        output_fields: Optional[List[str]] = None,
    ) -> SearchResult:
        table = self._ensure_table()
        if sparse_vector:
            raise NotImplementedError(
                "The LanceDB backend is dense-only; sparse query vectors are not supported"
            )
        query = self._validate_query_vector(dense_vector)
        where = self._filter_sql(filters)
        builder = (
            table.search(query, vector_column_name=VECTOR_FIELD_NAME)
            .metric(self._metric)
            .where(where, prefilter=True)
            .limit(limit)
        )
        if offset:
            builder = builder.offset(offset)
        if output_fields is not None:
            projection = [ID_FIELD_NAME, *output_fields]
            builder = builder.select(list(dict.fromkeys(projection)))
        rows = builder.to_list()

        def _score(row: Dict[str, Any]) -> float:
            return self._score_from_distance(row.get("_distance", 0.0))

        return self._rows_to_search_result(rows, output_fields, _score)

    def search_by_keywords(
        self,
        index_name: str,
        keywords: Optional[List[str]] = None,
        query: Optional[str] = None,
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
        dense_vector: Optional[List[float]] = None,
    ) -> SearchResult:
        """Full-text (BM25) search with optional dense+lexical hybrid fusion.

        Both branches (FTS and vector) are constrained by the same SQL
        predicate before fusion, so account/ACL/path filters cannot be
        bypassed by either candidate source.
        """
        if not self._fts_cfg:
            raise NotImplementedError(
                "LanceDB keyword/full-text search is disabled; enable "
                "'lancedb.fts' (requires 'store_content: true') or rely on the "
                "filesystem grep fallback"
            )
        table = self._ensure_table()
        query_text = (
            query or " ".join(kw.strip() for kw in (keywords or []) if kw.strip())
        ).strip()
        if not query_text:
            return SearchResult()
        columns = list(self._fts_cfg.get("columns") or ["content"])
        where = self._filter_sql(filters)

        use_hybrid = dense_vector is not None and self._hybrid_cfg is not None
        if use_hybrid and dense_vector is not None:
            builder = (
                table.search(query_type="hybrid", vector_column_name=VECTOR_FIELD_NAME)
                .vector(self._validate_query_vector(dense_vector))
                .text(query_text)
                .where(where, prefilter=True)
                .rerank(self._hybrid_reranker())
                .limit(limit)
            )
        else:
            if dense_vector is not None:
                raise NotImplementedError(
                    "LanceDB hybrid search is disabled; enable 'lancedb.hybrid' "
                    "to fuse dense and lexical branches"
                )
            builder = (
                table.search(query_text, query_type="fts", fts_columns=columns)
                .where(where, prefilter=True)
                .limit(limit)
            )
        if offset:
            builder = builder.offset(offset)
        if output_fields is not None:
            projection = [ID_FIELD_NAME, *output_fields]
            builder = builder.select(list(dict.fromkeys(projection)))
        rows = builder.to_list()

        def _score(row: Dict[str, Any]) -> float:
            for key in ("_relevance_score", "_score"):
                value = row.get(key)
                if isinstance(value, (int, float)) and value == value:
                    return float(value)
            return 0.0

        return self._rows_to_search_result(rows, output_fields, _score)

    def _hybrid_reranker(self) -> Any:
        from lancedb.rerankers import RRFReranker

        cfg = self._hybrid_cfg or {}
        method = str(cfg.get("method", "rrf")).lower()
        if method == "rrf":
            return RRFReranker(K=int(cfg.get("rrf_k", 60)), return_score="relevance")
        if method == "weighted":
            from lancedb.rerankers import LinearCombinationReranker

            dense_weight = float(cfg.get("dense_weight", 0.7))
            lexical_weight = float(cfg.get("lexical_weight", 0.3))
            total = dense_weight + lexical_weight
            if total <= 0:
                raise ValueError("LanceDB hybrid weights must sum to a positive value")
            # LinearCombinationReranker weights the vector score; the lexical
            # weight is its complement after normalization.
            return LinearCombinationReranker(weight=dense_weight / total, return_score="relevance")
        raise ValueError(f"Unsupported LanceDB hybrid method {method!r}")

    def search_by_id(
        self,
        index_name: str,
        id: Any,
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
    ) -> SearchResult:
        table = self._ensure_table()
        rows = self._fetch_rows(table, [id])
        if not rows or not rows[0].get(VECTOR_FIELD_NAME):
            return SearchResult()
        dense_vector = [float(v) for v in rows[0][VECTOR_FIELD_NAME]]
        return self.search_by_vector(
            index_name,
            dense_vector=dense_vector,
            limit=limit,
            offset=offset,
            filters=filters,
            output_fields=output_fields,
        )

    def search_by_multimodal(
        self,
        index_name: str,
        text: Optional[str],
        image: Optional[Any],
        video: Optional[Any],
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
    ) -> SearchResult:
        raise NotImplementedError("LanceDB backend does not support multimodal search")

    def search_by_random(
        self,
        index_name: str,
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
    ) -> SearchResult:
        self._ensure_table()
        if self._dimension is None:
            raise ValueError("Random search requires a known vector dimension")
        random_vector = [random.uniform(-1, 1) for _ in range(self._dimension)]
        return self.search_by_vector(
            index_name,
            dense_vector=random_vector,
            limit=limit,
            offset=offset,
            filters=filters,
            output_fields=output_fields,
        )

    def search_by_scalar(
        self,
        index_name: str,
        field: str,
        order: Optional[str] = "desc",
        limit: int = 10,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        output_fields: Optional[List[str]] = None,
    ) -> SearchResult:
        table = self._ensure_table()
        where = self._filter_sql(filters)
        ascending = str(order or "desc").lower() != "desc"
        ordering_factory: Any = ColumnOrdering
        ordering = ordering_factory(column_name=field, ascending=ascending)
        builder = table.search().where(where, prefilter=True).order_by([ordering]).limit(limit)
        if offset:
            builder = builder.offset(offset)
        if output_fields is not None:
            projection = [ID_FIELD_NAME, *output_fields]
            builder = builder.select(list(dict.fromkeys(projection)))
        rows = self._normalize_rows(builder.to_list())

        def _score(row: Dict[str, Any]) -> float:
            value = row.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            return 0.0

        return self._rows_to_search_result(rows, output_fields, _score)

    # -- ICollection: aggregation ---------------------------------------------

    def aggregate_data(
        self,
        index_name: str,
        op: str = "count",
        field: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        cond: Optional[Dict[str, Any]] = None,
    ) -> AggregateResult:
        table = self._ensure_table()
        if op != "count":
            raise NotImplementedError(f"Aggregation op {op!r} is not supported by LanceDB backend")
        if field is not None:
            raise NotImplementedError("Grouped aggregation is not supported by the LanceDB backend")
        where = self._filter_sql(filters)
        total = table.count_rows(where)
        return AggregateResult(agg={"_total": total}, op="count", field=None)


# ---------------------------------------------------------------------------
# Table bootstrap helper
# ---------------------------------------------------------------------------


def table_exists(db: Any, table_name: str, namespace_path: Optional[List[str]] = None) -> bool:
    """Return whether *table_name* exists in *db*.

    Prefers listing table names (works for plain URI connections and
    namespace catalogs alike); falls back to the namespace-aware
    ``DB.table_exists`` with the fully qualified path.

    A connection- or client-level failure must propagate: swallowing it
    here made callers treat a broken backend as "table absent", which
    silently degraded every downstream search to empty results.
    """
    try:
        names = (
            db.table_names(namespace_path=namespace_path) if namespace_path else db.table_names()
        )
        return table_name in (names or [])
    except Exception as listing_err:
        try:
            return bool(db.table_exists([*(namespace_path or []), table_name]))
        except Exception:
            raise listing_err


def create_lancedb_collection(
    *,
    db: Any,
    table_name: str,
    meta_data: Dict[str, Any],
    dimension: Optional[int] = None,
    metric: str = "cosine",
    vector_index: Optional[Dict[str, Any]] = None,
    scalar_indexes: Optional[Dict[str, str]] = None,
    optimize_after_bulk_ingest: bool = True,
    fts: Optional[Dict[str, Any]] = None,
    hybrid: Optional[Dict[str, Any]] = None,
    store_content: bool = False,
    namespace_path: Optional[List[str]] = None,
) -> Tuple[LanceDBCollection, bool]:
    """Open or create a LanceDB-backed collection.

    Returns ``(collection, created)`` where *created* reports whether the
    underlying table was newly created.  Existing tables are reused in place;
    incompatible vector dimensions raise a clear error instead of creating a
    shadow table.
    """
    field_types = field_types_from_meta(meta_data)
    existed = table_exists(db, table_name, namespace_path)
    arrow_schema = None
    if not existed:
        arrow_schema = build_arrow_schema(meta_data)
    collection = LanceDBCollection(
        db=db,
        table_name=table_name,
        meta_data=meta_data,
        dimension=dimension,
        metric=metric,
        vector_index=vector_index,
        scalar_indexes=scalar_indexes,
        optimize_after_bulk_ingest=optimize_after_bulk_ingest,
        fts=fts,
        hybrid=hybrid,
        store_content=store_content,
        namespace_path=namespace_path,
    )
    if existed:
        return collection, False
    collection._field_types = field_types
    collection._load_or_create_table(create=True, arrow_schema=arrow_schema)
    return collection, True
