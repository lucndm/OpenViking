# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Unit tests for the OpenViking filter DSL -> SQL translator (no lancedb needed)."""

import pytest

from openviking.storage.vectordb.collection.lancedb_collection import (
    sql_string_literal,
    translate_filter_to_sql,
)

FIELD_TYPES = {
    "account_id": "string",
    "owner_user_id": "string",
    "context_type": "string",
    "level": "int64",
    "created_at": "date_time",
    "uri": "path",
    "name": "string",
    "tags": "string",
    "search_tags": "list<string>",
    "acl_mode": "string",
    "acl_direct_grants": "list<string>",
    "acl_inherited_grants": "list<string>",
}


def test_empty_filter_returns_none():
    assert translate_filter_to_sql(None, FIELD_TYPES) is None
    assert translate_filter_to_sql({}, FIELD_TYPES) is None


def test_nested_filter_wrapper_is_unwrapped():
    assert translate_filter_to_sql({"filter": None}, FIELD_TYPES) is None
    assert (
        translate_filter_to_sql(
            {"filter": {"op": "must", "field": "level", "conds": [1]}}, FIELD_TYPES
        )
        == '("level" = 1)'
    )


def test_must_single_and_multi_values():
    assert (
        translate_filter_to_sql(
            {"op": "must", "field": "account_id", "conds": ["acct-1"]}, FIELD_TYPES
        )
        == "(\"account_id\" = 'acct-1')"
    )
    sql = translate_filter_to_sql(
        {"op": "must", "field": "context_type", "conds": ["resource", "memory"]}, FIELD_TYPES
    )
    assert sql == "(\"context_type\" IN ('resource', 'memory'))"


def test_must_on_list_field_uses_array_has_any():
    sql = translate_filter_to_sql(
        {"op": "must", "field": "acl_direct_grants", "conds": ["acct-1:r", "acct-2:r"]},
        FIELD_TYPES,
    )
    assert sql == "(array_has_any(\"acl_direct_grants\", ['acct-1:r', 'acct-2:r']))"


def test_must_not_keeps_null_rows_visible():
    sql = translate_filter_to_sql(
        {"op": "must_not", "field": "acl_mode", "conds": ["inherit", "restricted"]},
        FIELD_TYPES,
    )
    assert sql == ("(\"acl_mode\" IS NULL OR NOT (\"acl_mode\" IN ('inherit', 'restricted')))")


def test_and_or_combination():
    sql = translate_filter_to_sql(
        {
            "op": "and",
            "conds": [
                {"op": "must", "field": "account_id", "conds": ["acct-1"]},
                {
                    "op": "or",
                    "conds": [
                        {"op": "must", "field": "level", "conds": [0]},
                        {"op": "must", "field": "level", "conds": [1]},
                    ],
                },
            ],
        },
        FIELD_TYPES,
    )
    assert sql == '(("account_id" = \'acct-1\') AND (("level" = 0) OR ("level" = 1)))'


def test_and_with_empty_children_returns_none():
    assert translate_filter_to_sql({"op": "and", "conds": [{}, None]}, FIELD_TYPES) is None


def test_range_numeric():
    sql = translate_filter_to_sql({"op": "range", "field": "level", "gte": 0, "lt": 2}, FIELD_TYPES)
    assert sql == '("level" >= 0 AND "level" < 2)'


def test_range_out_negates():
    sql = translate_filter_to_sql({"op": "range_out", "field": "level", "gt": 5}, FIELD_TYPES)
    assert sql == '(NOT ("level" > 5))'


def test_time_range_uses_timestamp_literal():
    sql = translate_filter_to_sql(
        {
            "op": "range",
            "field": "created_at",
            "gte": "2026-01-01T00:00:00+00:00",
            "lt": "2026-02-01",
        },
        FIELD_TYPES,
    )
    assert sql is not None
    assert "TIMESTAMP '2026-01-01T00:00:00'" in sql
    assert "TIMESTAMP '2026-02-01T00:00:00'" in sql


def test_path_scope_any_depth():
    sql = translate_filter_to_sql(
        {"op": "must", "field": "uri", "conds": ["/user/u1/resources"], "para": "-d=-1"},
        FIELD_TYPES,
    )
    assert sql == (
        "(((\"uri\" = '/user/u1/resources' OR starts_with(\"uri\", '/user/u1/resources/'))))"
    )


def test_path_scope_exact_match():
    sql = translate_filter_to_sql(
        {"op": "must", "field": "uri", "conds": ["/user/u1/resources"], "para": "-d=0"},
        FIELD_TYPES,
    )
    assert sql == "(((\"uri\" = '/user/u1/resources')))"


def test_path_scope_limited_depth_uses_regex():
    sql = translate_filter_to_sql(
        {"op": "must", "field": "uri", "conds": ["/res/docs"], "para": "-d=2"},
        FIELD_TYPES,
    )
    assert sql is not None
    assert 'array_length(regexp_match("uri"' in sql
    assert "|/res/docs/[^/]+" in sql


def test_path_scope_relative_path_is_normalized():
    # Bare relative paths get a leading slash; ``viking://`` encoding is
    # handled by ``CollectionAdapter._compile_filter`` before this layer,
    # matching the reference native-engine semantics.
    sql = translate_filter_to_sql(
        {"op": "must", "field": "uri", "conds": ["user/u1/resources"], "para": "-d=-1"},
        FIELD_TYPES,
    )
    assert sql is not None
    assert "(\"uri\" = '/user/u1/resources'" in sql
    assert "starts_with(\"uri\", '/user/u1/resources/')" in sql


def test_contains_string_field():
    sql = translate_filter_to_sql(
        {"op": "contains", "field": "name", "substring": "50%_off"}, FIELD_TYPES
    )
    assert sql == "(\"name\" LIKE '%50\\%\\_off%')"


def test_contains_list_field():
    sql = translate_filter_to_sql(
        {"op": "contains", "field": "search_tags", "substring": "python"}, FIELD_TYPES
    )
    assert sql is not None
    assert 'array_to_string("search_tags"' in sql
    assert "LIKE" in sql


def test_unsupported_op_raises():
    with pytest.raises(NotImplementedError):
        translate_filter_to_sql({"op": "fuzzy", "field": "name", "conds": ["x"]}, FIELD_TYPES)


def test_malformed_nodes_raise():
    with pytest.raises(ValueError):
        translate_filter_to_sql({"op": "must", "field": "level", "conds": []}, FIELD_TYPES)
    with pytest.raises(ValueError):
        translate_filter_to_sql({"op": "must", "conds": ["x"]}, FIELD_TYPES)
    with pytest.raises(ValueError):
        translate_filter_to_sql("not-a-dict", FIELD_TYPES)  # type: ignore[arg-type]


def test_para_only_supported_for_path_fields():
    with pytest.raises(ValueError):
        translate_filter_to_sql(
            {"op": "must", "field": "name", "conds": ["x"], "para": "-d=1"}, FIELD_TYPES
        )


def test_string_literal_escapes_quotes():
    assert sql_string_literal("it's") == "'it''s'"
