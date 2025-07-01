import numpy as np
import pandas as pd
import pytest

from tabularepimdl.arrayops import (
    grouped_sum,
    grouped_count,
    masked_sum,
)

# --- Fixtures ---

@pytest.fixture
def toy_dataframe():
    return pd.DataFrame({
        "age": ["0-4", "0-4", "5-9", "5-9", "0-4"],
        "region": ["North", "North", "South", "South", "North"],
        "sex": ["M", "F", "M", "F", "M"],
        "N": [10, 20, 30, 40, 50],
    })

@pytest.fixture
def toy_arrays(toy_dataframe):
    df = toy_dataframe
    return {
        "age": df["age"].values,
        "region": df["region"].values,
        "sex": df["sex"].values,
        "N": df["N"].values.astype(np.float32),
    }

# --- Core Utility Tests ---

def test_grouped_sum(toy_dataframe, toy_arrays):
    group_vals = toy_dataframe[["age", "region", "sex"]].agg("_".join, axis=1).values
    group_ids, mapping = encode_categories(group_vals)
    group_ids = group_ids.astype(np.int32)  # Ensure integer dtype for indexing
    result = grouped_sum(toy_arrays["N"], group_ids, len(mapping))
    expected = toy_dataframe.groupby(["age", "region", "sex"])["N"].sum().values.astype(np.float32)
    np.testing.assert_allclose(np.sort(result), np.sort(expected), rtol=1e-5)

def test_grouped_count(toy_dataframe, toy_arrays):
    group_vals = toy_dataframe[["age", "region", "sex"]].agg("_".join, axis=1).values
    group_ids, mapping = encode_categories(group_vals)
    group_ids = group_ids.astype(np.int32)  # ensure int dtype
    dummy_values = np.ones_like(group_ids, dtype=np.float32)
    result = grouped_count(dummy_values, group_ids, len(mapping))
    expected = toy_dataframe.groupby(["age", "region", "sex"]).size().values
    np.testing.assert_array_equal(np.sort(result), np.sort(expected))


@pytest.mark.parametrize("column", ["age", "region", "sex"])
def test_masked_sum(toy_dataframe, toy_arrays, column):
    val = toy_arrays[column][0]
    mask = toy_arrays[column] == val
    result = masked_sum(toy_arrays["N"], mask, T=1)
    expected = toy_dataframe.loc[toy_dataframe[column] == val, "N"].sum()
    np.testing.assert_allclose(result, expected, rtol=1e-5)

# --- Edge Case Tests ---

def test_grouped_sum_empty():
    result = grouped_sum(np.array([], dtype=np.float32), np.array([], dtype=np.int32), 0)
    assert result.size == 0

def test_grouped_count_all_same_group():
    values = np.ones(10, dtype=np.float32)
    group_ids = np.zeros(10, dtype=np.int32)
    result = grouped_count(values, group_ids, 1)
    assert result[0] == 10

@pytest.mark.parametrize("mask_fn, expected", [
    (lambda: np.zeros(10, dtype=bool), 0.0),
    (lambda: np.ones(10, dtype=bool), np.sum(np.arange(10, dtype=np.float32))),
])
def test_masked_sum_all_true_false(mask_fn, expected):
    values = np.arange(10, dtype=np.float32)
    result = masked_sum(values, mask_fn(), T=1)
    np.testing.assert_allclose(result, expected, rtol=1e-6)

# --- Consistency Test ---

def test_sum_matches_count_and_N(toy_dataframe, toy_arrays):
    group_ids, mapping = encode_categories(toy_arrays["sex"])
    group_ids = group_ids.astype(np.int32)  # Ensure integer dtype for indexing
    result = grouped_sum(toy_arrays["N"], group_ids, len(mapping))
    np.testing.assert_allclose(result.sum(), toy_arrays["N"].sum(), rtol=1e-5)
