# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from openviking_cli.utils.logger import get_logger

COLLECTION_NAME = "context"
DEFAULT_PROJECT_NAME = "default"
DEFAULT_INDEX_NAME = "default"
logger = get_logger(__name__)


class VolcengineConfig(BaseModel):
    """Configuration for Volcengine VikingDB."""

    ak: Optional[str] = Field(default=None, description="Volcengine Access Key")
    sk: Optional[str] = Field(default=None, description="Volcengine Secret Key")
    api_key: Optional[str] = Field(
        default=None,
        description="Optional VikingDB Data API key for data-plane-only access",
    )
    session_token: Optional[str] = Field(
        default=None,
        description="Optional Volcengine STS security token for temporary credentials",
    )
    region: Optional[str] = Field(
        default=None, description="Volcengine region (e.g., 'cn-beijing')"
    )
    host: Optional[str] = Field(
        default=None,
        description=(
            "Optional VikingDB data API host. "
            "Used together with `api_key` for data-plane-only access."
        ),
    )

    model_config = {"extra": "forbid"}


class VikingDBConfig(BaseModel):
    """Configuration for VikingDB private deployment."""

    host: Optional[str] = Field(default=None, description="VikingDB service host")
    headers: Optional[Dict[str, str]] = Field(
        default_factory=dict, description="Custom headers for requests"
    )

    model_config = {"extra": "forbid"}


class CuVSConfig(BaseModel):
    """Configuration for GPU dense-vector search through NVIDIA cuVS."""

    dtype: Literal["float32", "float16"] = Field(
        default="float32",
        description=(
            "GPU dataset and query dtype. float16 is an opt-in direct cast and "
            "must be benchmarked for recall; it does not change native CPU quantization."
        ),
    )
    algorithm: Literal["brute_force", "cagra"] = Field(
        default="brute_force",
        description=(
            "cuVS index algorithm. Start with brute_force for functional validation; "
            "use cagra for approximate search at larger scale."
        ),
    )
    build_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments passed to cuVS CAGRA IndexParams.",
    )
    search_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments passed to cuVS CAGRA SearchParams.",
    )
    fallback_to_native: bool = Field(
        default=True,
        description=(
            "Use OpenViking's native local index for sparse/hybrid search or other "
            "operations outside cuVS dense top-k."
        ),
    )
    auto_enable: bool = Field(
        default=False,
        description=(
            "When the VectorDB backend is 'local', automatically use cuVS dense search "
            "only when a visible GPU has enough free memory. The default is disabled."
        ),
    )
    auto_memory_reserve_mb: int = Field(
        default=1024,
        ge=0,
        description=("Free GPU memory kept outside the cuVS auto-admission budget, in MiB."),
    )
    auto_memory_safety_factor: float = Field(
        default=2.0,
        ge=1.0,
        description=(
            "Multiplier applied to the estimated cuVS vector, graph, build, and filter "
            "memory before auto-enabling GPU search."
        ),
    )
    auto_filter_native_threshold: int = Field(
        default=2000,
        ge=0,
        description=(
            "In cuVS auto mode, route filtered queries with at most this many "
            "eligible vectors to the native index. Set to zero to disable "
            "latency-aware filter routing."
        ),
    )
    auto_path_filter_native_threshold: int = Field(
        default=200,
        ge=0,
        description=(
            "In cuVS auto mode, use this lower native-routing threshold for path "
            "filters, whose native Trie/bitmap construction cost can dominate wider "
            "subtree queries. Set to zero to keep all path filters on cuVS."
        ),
    )
    filter_cache_size: int = Field(
        default=16,
        ge=0,
        description=(
            "Maximum number of repeated scalar-filter bitsets retained on the GPU. "
            "Set to zero to disable caching."
        ),
    )
    max_concurrent_gpu_searches: int = Field(
        default=1,
        ge=1,
        description=(
            "Maximum in-flight cuVS GPU search calls per index. Host-side filter and "
            "snapshot work remains concurrent; increase only after hardware-specific tuning."
        ),
    )
    micro_batching_enabled: bool = Field(
        default=False,
        description=(
            "Coalesce compatible concurrent cuVS dense queries into one matrix-search call. "
            "This OpenViking scheduler is opt-in and distinct from cuVS Dynamic Batching."
        ),
    )
    micro_batching_max_batch_size: int = Field(
        default=8,
        ge=1,
        le=8,
        description="Maximum compatible queries submitted in one cuVS search call.",
    )
    micro_batching_max_wait_ms: float = Field(
        default=1.0,
        ge=0.0,
        le=100.0,
        allow_inf_nan=False,
        description=(
            "Maximum collection window for a compatible cuVS micro-batch, in milliseconds. "
            "Zero performs opportunistic batching without an intentional wait."
        ),
    )
    auto_background_rebuild: bool = Field(
        default=False,
        description=(
            "Build dirty auto-cuVS snapshots in a coalescing background worker. "
            "Queries use the native index until the new GPU snapshot is committed."
        ),
    )
    auto_rebuild_debounce_ms: int = Field(
        default=500,
        ge=0,
        description=(
            "Quiet period used to coalesce consecutive mutations before an auto-cuVS "
            "background rebuild."
        ),
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_micro_batching(self):
        if not self.micro_batching_enabled:
            return self
        if self.algorithm != "brute_force":
            raise ValueError("cuVS micro-batching currently supports algorithm='brute_force' only")
        if self.max_concurrent_gpu_searches != 1:
            raise ValueError("cuVS micro-batching currently requires max_concurrent_gpu_searches=1")
        return self


class LanceVectorIndexConfig(BaseModel):
    """Native Lance vector (ANN) index configuration."""

    index_type: Literal[
        "IVF_PQ",
        "IVF_FLAT",
        "IVF_SQ",
        "IVF_HNSW_PQ",
        "IVF_HNSW_SQ",
        "IVF_HNSW_FLAT",
    ] = Field(
        default="IVF_PQ",
        description="Lance ANN index type; IVF_PQ is the production default.",
    )
    num_partitions: Optional[int] = Field(
        default=None,
        ge=1,
        description="IVF partition count; None lets Lance derive it from table size.",
    )
    num_sub_vectors: Optional[int] = Field(
        default=None,
        ge=1,
        description="PQ sub-vector count; None lets Lance derive it from dimension.",
    )
    num_bits: int = Field(default=8, ge=1, le=16, description="PQ quantization bits.")
    min_rows_to_build: int = Field(
        default=256,
        ge=0,
        description=(
            "Defer ANN training until the table has at least this many rows. "
            "Search remains correct meanwhile: unindexed fragments are scanned "
            "flat."
        ),
    )

    model_config = {"extra": "forbid"}


class LanceFTSConfig(BaseModel):
    """Full-text search configuration for the LanceDB backend."""

    columns: list[str] = Field(
        default_factory=lambda: ["content"],
        description="Columns to index for full-text search.",
    )
    language: str = Field(
        default="English",
        description="Stemming language for the tokenizer (e.g. 'English', 'Chinese').",
    )
    with_position: bool = Field(
        default=False,
        description="Store token positions; required for phrase queries.",
    )
    stem: bool = Field(default=True, description="Enable stemming.")
    remove_stop_words: bool = Field(default=True, description="Drop stop words.")
    ascii_folding: bool = Field(
        default=True,
        description="Fold accented characters to ASCII so diacritic-insensitive matches work.",
    )
    custom_stop_words: Optional[list[str]] = Field(
        default=None, description="Extra stop words beyond the language defaults."
    )

    model_config = {"extra": "forbid"}


class LanceHybridConfig(BaseModel):
    """Dense + lexical fusion configuration for the LanceDB backend.

    These are neutral controls: the generic ``sparse_weight`` setting is not
    reused for LanceDB lexical weighting.
    """

    method: Literal["rrf", "weighted"] = Field(
        default="rrf",
        description="Fusion method: reciprocal rank fusion or weighted scores.",
    )
    dense_weight: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Weight of the dense branch."
    )
    lexical_weight: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Weight of the lexical branch."
    )
    rrf_k: int = Field(default=60, ge=1, description="RRF smoothing constant.")

    model_config = {"extra": "forbid"}


class LanceDBConfig(BaseModel):
    """Configuration for the LanceDB backend.

    ``uri`` addresses the LanceDB database: a local directory path (for
    development) or an object-store prefix such as ``s3://bucket/parent``
    (durable deployments, e.g. SeaweedFS S3).  Storage credentials are passed
    via ``storage_options``; values of the form ``${ENV_VAR}`` are expanded
    from the environment so permanent keys never live in config files.
    """

    uri: Optional[str] = Field(
        default=None,
        description=(
            "LanceDB database URI: local path or object-store prefix "
            "(e.g. 's3://openviking/lancedb')"
        ),
    )
    namespace_uri: Optional[str] = Field(
        default=None,
        description=(
            "Lance Namespace REST endpoint (e.g. 'http://seaweedfs:9101'). "
            "When set, tables are managed through the namespace catalog "
            "instead of the plain URI."
        ),
    )
    namespace_path: list[str] = Field(
        default_factory=list,
        description=(
            "Namespace path segments below the catalog root. For SeaweedFS "
            "the first element is the Lance table bucket, e.g. "
            "['vectors', 'openviking']."
        ),
    )
    storage_options: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Storage options passed to LanceDB (e.g. aws_endpoint, aws_region, "
            "allow_http). '${ENV_VAR}' values are expanded from the environment."
        ),
    )
    read_consistency_interval_ms: Optional[float] = Field(
        default=None,
        ge=0,
        description=(
            "Interval for automatic table refresh, in milliseconds. None keeps the LanceDB default."
        ),
    )
    vector_index: Optional[LanceVectorIndexConfig] = Field(
        default=None,
        description=(
            "Native Lance ANN vector index; None keeps flat (unindexed) dense "
            "search, which stays correct at any table size."
        ),
    )
    scalar_indexes: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Native Lance scalar indexes per field, e.g. "
            "{'account_id': 'BTREE', 'search_tags': 'LABEL_LIST'}. "
            "Supported types: BTREE, BITMAP, LABEL_LIST."
        ),
    )
    optimize_after_bulk_ingest: bool = Field(
        default=True,
        description=(
            "Run one coalesced maintenance pass (compaction + index "
            "optimization) when a bulk-ingest scope ends, instead of per batch."
        ),
    )
    store_content: bool = Field(
        default=False,
        description=(
            "Persist the full 'content' field so LanceDB full-text search and "
            "grep integration can be enabled. AGFS remains the canonical "
            "content source."
        ),
    )
    fts: Optional[LanceFTSConfig] = Field(
        default=None,
        description="Opt-in full-text (BM25) index over persisted text columns.",
    )
    hybrid: Optional[LanceHybridConfig] = Field(
        default=None,
        description=(
            "Opt-in dense + lexical fusion used when a keyword query also carries a query vector."
        ),
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_scalar_indexes(self):
        allowed = {"BTREE", "BITMAP", "LABEL_LIST"}
        for field, index_type in self.scalar_indexes.items():
            if index_type not in allowed:
                raise ValueError(
                    f"LanceDB scalar index type {index_type!r} for field {field!r} "
                    f"must be one of {sorted(allowed)}"
                )
        return self

    @model_validator(mode="after")
    def validate_connection_mode(self):
        if self.namespace_uri and not self.namespace_path:
            raise ValueError(
                "LanceDB namespace_uri requires namespace_path "
                "(e.g. ['<bucket>', 'openviking']); the first segment is the "
                "Lance table bucket"
            )
        if not self.namespace_uri and not self.uri:
            raise ValueError(
                "LanceDB requires either 'uri' (local path or s3:// prefix) "
                "or 'namespace_uri' + 'namespace_path' (Lance catalog)"
            )
        return self

    @model_validator(mode="after")
    def validate_search_features(self):
        if self.fts is not None and not self.store_content:
            if "content" in self.fts.columns:
                raise ValueError(
                    "LanceDB full-text search over 'content' requires "
                    "'store_content: true'; refusing to index a field that is "
                    "never persisted"
                )
        if self.hybrid is not None and self.fts is None:
            raise ValueError(
                "LanceDB hybrid search requires 'fts' to be configured for the lexical branch"
            )
        if self.hybrid is not None:
            total = self.hybrid.dense_weight + self.hybrid.lexical_weight
            if total <= 0:
                raise ValueError("LanceDB hybrid weights must sum to a positive value")
        return self


class VectorDBBackendConfig(BaseModel):
    """
    Configuration for VectorDB backend.

    This configuration class consolidates all settings related to the VectorDB backend,
    including type, connection details, and backend-specific parameters.
    """

    backend: str = Field(
        default="local",
        description=(
            "VectorDB backend type: 'local', 'cuvs', 'http', "
            "'volcengine' (AK/SK signed or API key data-plane only), "
            "'vikingdb' (private deployment), or 'lancedb' (Lance format, "
            "local or object storage)"
        ),
    )

    name: Optional[str] = Field(default=COLLECTION_NAME, description="Collection name for VectorDB")

    path: Optional[str] = Field(
        default=None,
        description="[Deprecated in favor of `storage.workspace`] Local storage path for 'local' type. This will be ignored if `storage.workspace` is set.",
    )

    url: Optional[str] = Field(
        default=None,
        description="Remote service URL for 'http' type (e.g., 'http://localhost:5000')",
    )

    project_name: Optional[str] = Field(
        default=DEFAULT_PROJECT_NAME, description="project name", alias="project"
    )

    index_name: Optional[str] = Field(
        default=DEFAULT_INDEX_NAME,
        description="Default index name for VectorDB operations",
    )

    distance_metric: str = Field(
        default="cosine",
        description="Distance metric for vector similarity search (e.g., 'cosine', 'l2', 'ip')",
    )

    dimension: int = Field(
        default=0,
        description="Dimension of vector embeddings",
    )

    sparse_weight: float = Field(
        default=0.0,
        description=(
            "Sparse weight for hybrid vector search. "
            "When > 0, sparse vectors are used for index build and search."
        ),
    )

    volcengine: Optional[VolcengineConfig] = Field(
        default_factory=VolcengineConfig,
        description="Volcengine VikingDB configuration for 'volcengine' type",
    )

    # VikingDB private deployment mode
    vikingdb: Optional[VikingDBConfig] = Field(
        default_factory=VikingDBConfig,
        description="VikingDB private deployment configuration for 'vikingdb' type",
    )

    cuvs: Optional[CuVSConfig] = Field(
        default_factory=CuVSConfig,
        description="NVIDIA cuVS dense-vector search configuration for the 'cuvs' backend",
    )

    lancedb: Optional[LanceDBConfig] = Field(
        default_factory=LanceDBConfig,
        description="LanceDB configuration for the 'lancedb' backend",
    )

    custom_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom parameters for custom backend adapters",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_config(self):
        """Validate configuration completeness and consistency"""
        standard_backends = [
            "local",
            "cuvs",
            "http",
            "volcengine",
            "vikingdb",
            "lancedb",
        ]

        # Allow custom backend classes (containing dot) without standard validation
        if "." in self.backend:
            logger.info("Using custom VectorDB backend: %s", self.backend)
            return self

        if self.backend not in standard_backends:
            raise ValueError(
                f"Invalid VectorDB backend: '{self.backend}'. Must be one of: {standard_backends} "
                "or a valid Python class path."
            )

        if self.backend in {"local", "cuvs"}:
            pass

        elif self.backend == "http":
            if not self.url:
                raise ValueError("VectorDB http backend requires 'url' to be set")

        elif self.backend == "volcengine":
            if self.volcengine and self.volcengine.host:
                self.volcengine.host = self.volcengine.host.strip().rstrip("/")

            uses_api_key = bool(self.volcengine and self.volcengine.api_key)
            if uses_api_key:
                if not self.volcengine or not (self.volcengine.host or self.volcengine.region):
                    raise ValueError(
                        "VectorDB volcengine backend with 'api_key' requires 'host' or 'region' to be set"
                    )
            else:
                if not self.volcengine or not self.volcengine.ak or not self.volcengine.sk:
                    raise ValueError(
                        "VectorDB volcengine backend requires 'ak' and 'sk' to be set "
                        "when 'api_key' is not configured"
                    )
                if not self.volcengine.region:
                    raise ValueError("VectorDB volcengine backend requires 'region' to be set")
            if self.volcengine and self.volcengine.host and not uses_api_key:
                logger.warning(
                    "VectorDB volcengine backend: 'volcengine.host' is ignored in AK/SK mode. "
                    "Using region-based console/data hosts for region='%s'.",
                    self.volcengine.region or "",
                )

        elif self.backend == "vikingdb":
            if not self.vikingdb or not self.vikingdb.host:
                raise ValueError("VectorDB vikingdb backend requires 'host' to be set")

        elif self.backend == "lancedb":
            if not self.lancedb or not (self.lancedb.uri or self.lancedb.namespace_uri):
                raise ValueError(
                    "VectorDB lancedb backend requires 'lancedb.uri' "
                    "(or 'lancedb.namespace_uri' + 'namespace_path') to be set"
                )
            if self.sparse_weight > 0.0:
                raise ValueError(
                    "VectorDB lancedb backend is dense-only; 'sparse_weight' must be 0"
                )
            if self.distance_metric not in ("cosine", "l2", "ip"):
                raise ValueError(
                    "VectorDB lancedb backend supports distance_metric 'cosine', 'l2', or 'ip'"
                )

        return self
