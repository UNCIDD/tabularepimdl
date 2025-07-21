import numpy as np
import pytest
from numpy.testing import assert_allclose
import numba as nb
from tabularepimdl.operations import encode_categories


def setup_data(N):
    age = np.random.choice(["0-4", "5-9", "10-14"], size=N)
    region = np.random.choice(["North", "South", "East", "West"], size=N)
    sex = np.random.choice(["M", "F"], size=N)
    N_vals = np.random.randint(1, 10, size=N).astype(np.float32)
    probs = np.random.uniform(0, 1, size=N).astype(np.float32)
    mask = np.random.choice([True, False], size=N)

    group_keys = np.array([f"{a}_{r}_{s}" for a, r, s in zip(age, region, sex)])
    group_ids, mapping = encode_categories(group_keys)
    n_groups = len(mapping)
    return N_vals, probs, mask, group_ids, n_groups


@pytest.mark.parametrize("N", [10_000])
def test_grouped_sum_fastmath(N):
    values, _, _, group_ids, n_groups = setup_data(N)

    @nb.njit(fastmath=False)
    def grouped_sum_no(values, group_ids, n_groups):
        result = np.zeros(n_groups, dtype=values.dtype)
        for i in range(values.shape[0]):
            result[group_ids[i]] += values[i]
        return result

    @nb.njit(fastmath=True)
    def grouped_sum_yes(values, group_ids, n_groups):
        result = np.zeros(n_groups, dtype=values.dtype)
        for i in range(values.shape[0]):
            result[group_ids[i]] += values[i]
        return result

    assert_allclose(grouped_sum_yes(values, group_ids, n_groups),
                    grouped_sum_no(values, group_ids, n_groups),
                    rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("N", [10_000])
def test_grouped_sum_parallel_fastmath(N):
    values, _, _, group_ids, n_groups = setup_data(N)

    def grouped_sum_par(values, group_ids, n_groups):
        tmp = np.zeros((nb.get_num_threads(), n_groups), dtype=values.dtype)
        for i in nb.prange(values.shape[0]):
            tid = nb.get_thread_id()
            tmp[tid, group_ids[i]] += values[i]
        result = np.zeros(n_groups, dtype=values.dtype)
        for t in range(tmp.shape[0]):
            for g in range(n_groups):
                result[g] += tmp[t, g]
        return result

    fn_no = nb.njit(parallel=True, fastmath=False)(grouped_sum_par)
    fn_yes = nb.njit(parallel=True, fastmath=True)(grouped_sum_par)

    out_no = fn_no(values, group_ids, n_groups)
    out_yes = fn_yes(values, group_ids, n_groups)
    assert_allclose(out_yes, out_no, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("N", [10_000])
def test_grouped_count_fastmath(N):
    _, _, _, group_ids, n_groups = setup_data(N)

    @nb.njit(fastmath=False)
    def grouped_count_no(group_ids, n_groups):
        result = np.zeros(n_groups, dtype=np.int64)
        for i in range(group_ids.shape[0]):
            result[group_ids[i]] += 1
        return result

    @nb.njit(fastmath=True)
    def grouped_count_yes(group_ids, n_groups):
        result = np.zeros(n_groups, dtype=np.int64)
        for i in range(group_ids.shape[0]):
            result[group_ids[i]] += 1
        return result

    assert_allclose(grouped_count_yes(group_ids, n_groups),
                    grouped_count_no(group_ids, n_groups))


@pytest.mark.parametrize("N", [10_000])
def test_grouped_count_parallel_fastmath(N):
    _, _, _, group_ids, n_groups = setup_data(N)

    def grouped_count_par(group_ids, n_groups):
        tmp = np.zeros((nb.get_num_threads(), n_groups), dtype=np.int64)
        for i in nb.prange(group_ids.shape[0]):
            tid = nb.get_thread_id()
            tmp[tid, group_ids[i]] += 1
        result = np.zeros(n_groups, dtype=np.int64)
        for t in range(tmp.shape[0]):
            for g in range(n_groups):
                result[g] += tmp[t, g]
        return result

    fn_no = nb.njit(parallel=True, fastmath=False)(grouped_count_par)
    fn_yes = nb.njit(parallel=True, fastmath=True)(grouped_count_par)

    out_no = fn_no(group_ids, n_groups)
    out_yes = fn_yes(group_ids, n_groups)
    assert_allclose(out_yes, out_no)


@pytest.mark.parametrize("N", [10_000])
def test_masked_sum_fastmath(N):
    values, _, mask, _, _ = setup_data(N)

    @nb.njit(fastmath=False)
    def masked_sum_no(values, mask):
        total = 0.0
        for i in range(values.shape[0]):
            if mask[i]:
                total += values[i]
        return total

    @nb.njit(fastmath=True)
    def masked_sum_yes(values, mask):
        total = 0.0
        for i in range(values.shape[0]):
            if mask[i]:
                total += values[i]
        return total

    assert_allclose(masked_sum_yes(values, mask),
                    masked_sum_no(values, mask),
                    rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("N", [10_000])
def test_masked_sum_parallel_fastmath(N):
    values, _, mask, _, _ = setup_data(N)

    def masked_sum_par(values, mask):
        tmp = np.zeros(nb.get_num_threads(), dtype=values.dtype)
        for i in nb.prange(values.shape[0]):
            if mask[i]:
                tmp[nb.get_thread_id()] += values[i]
        return tmp.sum()

    fn_no = nb.njit(parallel=True, fastmath=False)(masked_sum_par)
    fn_yes = nb.njit(parallel=True, fastmath=True)(masked_sum_par)

    out_no = fn_no(values, mask)
    out_yes = fn_yes(values, mask)
    assert_allclose(out_yes, out_no, rtol=1e-6, atol=1e-6)
