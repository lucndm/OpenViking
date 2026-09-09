# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Adapter registry and factory entrypoints."""

from __future__ import annotations

import importlib

from .base import CollectionAdapter
from .http_adapter import HttpCollectionAdapter
from .local_adapter import CuVSCollectionAdapter, LocalCollectionAdapter
from .vikingdb_private_adapter import VikingDBPrivateCollectionAdapter
from .volcengine_adapter import VolcengineCollectionAdapter

_ADAPTER_REGISTRY: dict[str, type[CollectionAdapter] | str] = {
    "local": LocalCollectionAdapter,
    "cuvs": CuVSCollectionAdapter,
    "http": HttpCollectionAdapter,
    "volcengine": VolcengineCollectionAdapter,
    "vikingdb": VikingDBPrivateCollectionAdapter,
    # Lazily imported so environments without the optional 'lancedb' package
    # can still use every other backend.
    "lancedb": "openviking.storage.vectordb_adapters.lancedb_adapter.LanceDBCollectionAdapter",
}


def create_collection_adapter(config) -> CollectionAdapter:
    """Unified factory entrypoint for backend-specific collection adapters."""
    backend = config.backend
    entry = _ADAPTER_REGISTRY.get(backend)
    adapter_cls: type[CollectionAdapter] | None = entry if isinstance(entry, type) else None

    # Resolve string entries (registry or config-provided class paths).
    class_path = entry if isinstance(entry, str) else None
    if adapter_cls is None and backend not in _ADAPTER_REGISTRY and "." in backend:
        class_path = backend

    if class_path is not None:
        try:
            module_name, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            potential_cls = getattr(module, class_name)
            if issubclass(potential_cls, CollectionAdapter):
                adapter_cls = potential_cls
        except (ImportError, AttributeError, TypeError):
            # Fall through to raising error if dynamic loading fails
            pass

    if adapter_cls is None:
        raise ValueError(
            f"Vector backend {backend} is not supported. "
            f"Available backends: {sorted(_ADAPTER_REGISTRY)}"
        )
    return adapter_cls.from_config(config)
