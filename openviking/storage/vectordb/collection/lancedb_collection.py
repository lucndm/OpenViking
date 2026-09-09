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

    def _load_or_create_table(self, create: bool, arrow_schema: Any = None) -> None:
        if table_exists(self._db, self._table_name):
            self._table = self._db.open_table(self._table_name)
            self._restore_metadata()
            self._validate_dimension()
        elif create:
            if arrow_schema is None:
                raise ValueError("Creating a LanceDB table requires a schema")
            self._table = self._db.create_table(self._table_name, schema=arrow_schema)
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
        return meta

    def update(self, fields: Optional[Dict[str, Any]] = None, description: Optional[str] = None):
        if description is not None:
            self._meta_data["Description"] = description
        self._persist_metadata()

    def close(self):
        self._table = None

    def drop(self):
        self._db.drop_table(self._table_name)
        self._table = None

    # -- ICollection: index lifecycle ----------------------------------------

    def create_index(self, index_name: str, meta_data: Dict[str, Any]) -> Any:
        vector_index = meta_data.get("VectorIndex", {}) if isinstance(meta_data, dict) else {}
        if vector_index.get("EnableSparse"):
            raise NotImplementedError(
                "The LanceDB backend is dense-only in this phase; sparse/hybrid index "
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
        return self.get_index(index_name)

    def has_index(self, index_name: str) -> bool:
        return index_name in self._index_metas

    def get_index(self, index_name: str) -> Optional[Any]:
        if index_name not in self._index_metas:
            return None
        return self._index_metas[index_name]

    def get_index_meta_data(self, index_name: str) -> Dict[str, Any]:
        return dict(self._index_metas.get(index_name, {}))

    def list_indexes(self) -> List[str]:
        return list(self._index_metas.keys())

    def drop_index(self, index_name: str):
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
    ) -> SearchResult:
        raise NotImplementedError(
            "LanceDB keyword/full-text search is not enabled for this backend; "
            "grep falls back to the filesystem engine"
        )

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


def table_exists(db: Any, table_name: str) -> bool:
    """Return whether *table_name* exists in *db*.

    ``DB.table_exists`` only supports namespace connections, so plain URI
    connections (local directories, object stores) fall back to listing table
    names.
    """
    try:
        return table_name in (db.table_names() or [])
    except Exception:
        try:
            return bool(db.table_exists([table_name]))
        except Exception:
            return False


def create_lancedb_collection(
    *,
    db: Any,
    table_name: str,
    meta_data: Dict[str, Any],
    dimension: Optional[int] = None,
    metric: str = "cosine",
) -> Tuple[LanceDBCollection, bool]:
    """Open or create a LanceDB-backed collection.

    Returns ``(collection, created)`` where *created* reports whether the
    underlying table was newly created.  Existing tables are reused in place;
    incompatible vector dimensions raise a clear error instead of creating a
    shadow table.
    """
    field_types = field_types_from_meta(meta_data)
    existed = table_exists(db, table_name)
    arrow_schema = None
    if not existed:
        arrow_schema = build_arrow_schema(meta_data)
    collection = LanceDBCollection(
        db=db,
        table_name=table_name,
        meta_data=meta_data,
        dimension=dimension,
        metric=metric,
    )
    if existed:
        return collection, False
    collection._field_types = field_types
    collection._load_or_create_table(create=True, arrow_schema=arrow_schema)
    return collection, True
