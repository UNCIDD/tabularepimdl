import os
import sys
import time
import psutil
import signal
import pickle
import threading
import tempfile
import subprocess
import multiprocessing
import numpy as np
from tabularepimdl.arrayops import masked_sum as masked_sum_array
from tabularepimdl.matrixops import matrix_masked_sum
from benchmark_utils import (
    save_benchmark_result,
    generate_results_file,
    hash_fn_source,
)

# === CONFIG ===
MAX_TIME = 60.0
MAX_MEMORY_MB = 28000
MIN_PEAK_BYTES = 4096
NUM_CORES = multiprocessing.cpu_count()
CORE_CAP = max(1, NUM_CORES - 2)
print(f"Running with {CORE_CAP} threads (machine has {NUM_CORES})")

def setup_data(N, G, T):
    data = np.random.rand(N * T).astype(np.float32)  # Flattened for 1D input
    group_ids = np.repeat(np.random.randint(0, G, size=N).astype(np.int32), T)

    mask_matrix = np.zeros((G, N * T), dtype=np.float32)
    mask_matrix[group_ids, np.arange(N * T)] = 1.0

    return data, group_ids, mask_matrix

def run_benchmark_case(dispatch_type, data, group_ids, mask_matrix):
    if dispatch_type == "array":
        result = masked_sum_array(data, group_ids)
        return result, np.dtype(type(result)).itemsize


    elif dispatch_type == "matrix":
        result = matrix_masked_sum(mask_matrix, data)
        return result, np.dtype(type(result)).itemsize

    else:
        raise ValueError(f"Unknown dispatch type: {dispatch_type}")

def worker_main(args_file, result_file):
    with open(args_file, "rb") as f:
        args = pickle.load(f)
    result = run_benchmark_case(*args)
    with open(result_file, "wb") as f:
        pickle.dump(result, f)

def run_benchmarks():
    Ns = [10_000, 100_000, 1_000_000, 10_000_000]
    Gs = [1, 5, 10, 25, 50, 100, 250, 500, 1000]
    T = 300
    dispatch_types = ["array", "matrix"]

    results_file, _ = generate_results_file("array_vs_matrix_masked_sum")

    for N in Ns:
        for G in Gs:
            data, group_ids, mask_matrix = setup_data(N, G, T)

            for dispatch_type in dispatch_types:
                args = (dispatch_type, data, group_ids, mask_matrix)
                fn_hash = hash_fn_source(run_benchmark_case)

                with tempfile.NamedTemporaryFile(delete=False) as f_args, tempfile.NamedTemporaryFile(delete=False) as f_result:
                    pickle.dump(args, f_args)
                    f_args.flush()

                    cmd = [sys.executable, __file__, f_args.name, f_result.name]
                    proc = subprocess.Popen(cmd, start_new_session=True)
                    ps_proc = psutil.Process(proc.pid)
                    peak_rss = 0

                    def monitor():
                        nonlocal peak_rss
                        try:
                            while proc.poll() is None:
                                rss = ps_proc.memory_info().rss
                                peak_rss = max(peak_rss, rss)
                                if rss > (MAX_MEMORY_MB * 1024 ** 2):
                                    print(f"[{dispatch_type}] N={N}, G={G} | MEMORY CAP EXCEEDED")
                                    os.killpg(proc.pid, signal.SIGKILL)
                                    proc.wait()
                                    time.sleep(0.05)
                                    break
                                time.sleep(0.01)
                        except psutil.NoSuchProcess:
                            pass

                    monitor_thread = threading.Thread(target=monitor)
                    monitor_thread.start()

                    try:
                        t0 = time.perf_counter()
                        proc.wait(timeout=MAX_TIME)
                        elapsed = time.perf_counter() - t0

                        if proc.returncode != 0:
                            result = None
                            dynamic_mem = max(peak_rss, MIN_PEAK_BYTES)
                            status = "FAILED" if peak_rss < (MAX_MEMORY_MB * 1024**2) else "MEMORY_CAP"
                        else:
                            with open(f_result.name, "rb") as f:
                                result, dynamic_mem = pickle.load(f)
                            dynamic_mem = max(dynamic_mem, peak_rss, MIN_PEAK_BYTES)
                            status = "OK"

                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                        time.sleep(0.05)
                        elapsed = MAX_TIME + 1
                        result = None
                        dynamic_mem = max(peak_rss, MIN_PEAK_BYTES)
                        status = "TIMEOUT"
                        print(f"[{dispatch_type}] N={N}, G={G} | TIMEOUT")

                    finally:
                        monitor_thread.join()
                        os.unlink(f_args.name)
                        os.unlink(f_result.name)

                    result_shape = getattr(result, "shape", None)
                    result_shape = str(result_shape) if result_shape is not None else "n/a"

                    save_benchmark_result(
                        results_file=results_file,
                        tier=dispatch_type,
                        N=N,
                        T=T,
                        G=G,
                        elapsed=elapsed,
                        peak_memory=dynamic_mem,
                        result_shape=result_shape,
                        cap_time=MAX_TIME,
                        fn_hash=fn_hash,
                        function_name="masked_sum"
                    )

                    print(f"[{dispatch_type}] N={N}, G={G} | Time: {elapsed:.2f}s | Mem: {dynamic_mem / 1024**2:.2f}MB | Status: {status}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        worker_main(sys.argv[1], sys.argv[2])
    else:
        run_benchmarks()
