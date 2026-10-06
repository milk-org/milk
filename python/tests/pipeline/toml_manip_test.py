from __future__ import annotations

import pytest

from milk.infra.toml_manipulation import (
    denest_toml_dicts,
    renest_toml_dicts,
    substitute_toml_variables_in_nested,
    substitute_toml_variables_in_denested,
)

NESTED = {
    "a": {"b": 1, "c": 2},
    "d": 3,
    "e": {"f": {"g": "x", "h": True}},
    "i": 1.5,
}

FLAT = {
    "a.b": 1,
    "a.c": 2,
    "d": 3,
    "e.f.g": "x",
    "e.f.h": True,
    "i": 1.5,
}


def test_denest_toml_dicts():
    assert denest_toml_dicts(NESTED) == FLAT


def test_renest_toml_dicts():
    assert renest_toml_dicts(FLAT) == NESTED


def test_denest_no_nesting():
    flat = {"a": 1, "b": "x", "c": False}
    assert denest_toml_dicts(flat) == flat


def test_renest_no_dots():
    flat = {"a": 1, "b": "x", "c": False}
    assert renest_toml_dicts(flat) == flat


def test_renest_non_contiguous_same_head():
    # Keys sharing the same head do not need to be adjacent.
    flat = {"a.b": 1, "c.d": 2, "a.e": 3}
    assert renest_toml_dicts(flat) == {"a": {"b": 1, "e": 3}, "c": {"d": 2}}  # type: ignore


def test_denest_then_renest_is_identity():
    assert renest_toml_dicts(denest_toml_dicts(NESTED)) == NESTED


def test_renest_then_denest_is_identity():
    assert denest_toml_dicts(renest_toml_dicts(FLAT)) == FLAT


def test_sub():
    assert substitute_toml_variables_in_nested({"a": "b", "b": "{a}"}) == {
        "a": "b",
        "b": "b",
    }
    assert substitute_toml_variables_in_nested({"a": "{c}", "b": "{a}", "c": 1}) == {
        "a": 1,
        "b": 1,
        "c": 1,
    }


def test_sub_missing_variable():
    with pytest.raises(ValueError):
        substitute_toml_variables_in_nested({"a": "{c}"})


def test_sub_missing_variable_transitive():
    # 'b' references 'a', which itself references an undefined 'c'.
    with pytest.raises(ValueError):
        substitute_toml_variables_in_nested({"a": "{c}", "b": "{a}"})


def test_sub_circular_self_reference():
    with pytest.raises(ValueError):
        substitute_toml_variables_in_nested({"a": "{a}"})


def test_sub_circular_two_way():
    with pytest.raises(ValueError):
        substitute_toml_variables_in_nested({"a": "{b}", "b": "{a}"})


def test_sub_circular_longer_cycle():
    with pytest.raises(ValueError):
        substitute_toml_variables_in_nested({"a": "{b}", "b": "{c}", "c": "{a}"})


def test_sub_circular_longer_cycle_nested():
    with pytest.raises(ValueError):
        substitute_toml_variables_in_nested({"a": {"b": "{c}"}, "c": "{a.b}"})


def test_sub_nested_sibling_reference():
    nested = {"a": {"b": 1}, "c": "{a.b}"}
    assert substitute_toml_variables_in_nested(nested) == {"a": {"b": 1}, "c": 1}


def test_sub_nested_within_subtree():
    nested = {"a": {"b": 1, "c": "{a.b}"}}
    assert substitute_toml_variables_in_nested(nested) == {"a": {"b": 1, "c": 1}}  # type: ignore


def test_sub_nested_with_format():
    nested = {"a": {"b": 3.14159}, "c": {"d": "value={a.b:.2f}"}}
    assert substitute_toml_variables_in_nested(nested) == {
        "a": {"b": 3.14159},
        "c": {"d": "value=3.14"},
    }


def test_sub_nested_missing_variable():
    with pytest.raises(ValueError):
        substitute_toml_variables_in_nested({"a": {"b": "{x.y}"}})


def test_sub_nested_circular():
    nested = {"a": {"b": "{c.d}"}, "c": {"d": "{a.b}"}}
    with pytest.raises(ValueError):
        substitute_toml_variables_in_nested(nested)  # type: ignore
