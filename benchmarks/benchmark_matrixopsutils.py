import numpy as np
import time
from tabularepimdl.matrixops_utils import (
    matrix_grouped_sum_sparse, matrix_grouped_sum_dense,
    matrix_grouped_count_sparse, matrix_grouped_count_dense,
    matrix_masked_sum_sparse, matrix_masked_sum_dense,
    encode_sparse_groups, encode_dense_groups
)
from benchmarks.benchmark_utils import (
    generate_results_file, save_benchmark_result, hash_fn_source
)

def benchmark_op(fn_sparse, fn_dense, args_sparse, args_dense, fn_name, N, G, cap_time=60.0):
    # Benchmark dense
    t0 = time.perf_counter()
    dense_out = fn_dense(*args_dense)
    dense_elapsed = time.perf_counter() - t0

    # Benchmark sparse
    t0 = time.perf_counter()
    sparse_out = fn_sparse(*args_sparse)
    sparse_elapsed = time.perf_counter() - t0

    # Determine memory usage
    mem_dense = dense_out.nbytes
    mem_sparse = sparse_out.data.nbytes if hasattr(sparse_out, 'data') else sparse_out.nbytes

    save_benchmark_result(
        results_file, tier="dense", N=N, T=1, G=G,
        elapsed=dense_elapsed,
        peak_memory=mem_dense,
        result_shape=dense_out.shape,
        cap_time=cap_time,
        fn_hash=hash_fn_source(fn_dense),
        function_name=fn_name
    )

    save_benchmark_result(
        results_file, tier="sparse", N=N, T=1, G=G,
        elapsed=sparse_elapsed,
        peak_memory=mem_sparse,
        result_shape=sparse_out.shape,
        cap_time=cap_time,
        fn_hash=hash_fn_source(fn_sparse),
        function_name=fn_name
    )


def run_matrixops_benchmarks():
    global results_file
    results_file, _ = generate_results_file("matrixops")

    Ns = [250, 500, 750, 1000]
    for N in Ns:
        G = N // 5
        data = np.random.rand(N).astype(np.float32)
        group_ids = np.random.randint(0, G, size=N)

        G_sparse = encode_sparse_groups(group_ids, G)
        G_dense = encode_dense_groups(group_ids, G)

        # Grouped sum
        benchmark_op(
            matrix_grouped_sum_sparse,
            matrix_grouped_sum_dense,
            args_sparse=(G_sparse, data),
            args_dense=(G_dense, data),
            fn_name="matrix_grouped_sum",
            N=N, G=G
        )

        # Grouped count
        benchmark_op(
            matrix_grouped_count_sparse,
            matrix_grouped_count_dense,
            args_sparse=(G_sparse,),
            args_dense=(G_dense,),
            fn_name="matrix_grouped_count",
            N=N, G=G
        )

        # Masked sum
        benchmark_op(
            matrix_masked_sum_sparse,
            matrix_masked_sum_dense,
            args_sparse=(G_sparse, data),
            args_dense=(G_dense, data),
            fn_name="matrix_masked_sum",
            N=N, G=G
        )

        print(f"N={N:<5} | matrixops benchmarks complete")


if __name__ == "__main__":
    run_matrixops_benchmarks()


