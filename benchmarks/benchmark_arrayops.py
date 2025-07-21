import numpy as np
import pandas as pd
import time
import os
import sys
import tempfile
import pickle
import subprocess
import multiprocessing
import numba
import psutil
import threading
import signal

from tabularepimdl.arrayops_utils import (
    grouped_sum_serial, grouped_sum_parallel,
    grouped_count_serial, grouped_count_parallel,
    masked_sum_serial, masked_sum_parallel)

from tabularepimdl.operations import (apply_deterministic_transition,
    apply_stochastic_transition_serial,
    encode_categories)

from tabularepimdl.benchmark_utils import save_benchmark_result, hash_fn_source

# === CONFIG ===
MAX_TIME = 60.0  # seconds
MAX_MEMORY_MB = 28000  # 28 GB
MIN_PEAK_BYTES = 4096  # 4 KB minimum
NUM_CORES = multiprocessing.cpu_count()
CORE_CAP = max(1, NUM_CORES - 2)

numba.set_num_threads(CORE_CAP)
print(f"Running with {CORE_CAP} threads (machine has {NUM_CORES})")

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
    df_base = pd.DataFrame({"age": age, "region": region, "sex": sex, "N": N_vals})

    return N_vals, probs, mask, group_ids, n_groups, df_base


def worker_main(args_file, result_file):


    # Load args
    with open(args_file, "rb") as f:
        args = pickle.load(f)

    result = run_benchmark_case(*args)
    with open(result_file, "wb") as f:
        pickle.dump(result, f)

def run_benchmark_case(func_name, tier, N_vals, probs, mask, group_ids, n_groups, df_base, T):
    if func_name == "grouped_sum":
        if tier == "tier_0":
            df_all = pd.DataFrame()
            for _ in range(T):
                df_step = df_base.groupby(["age", "region", "sex"])["N"].sum().reset_index()
                df_all = pd.concat([df_all, df_step], ignore_index=True)
            mem = df_all.memory_usage(deep=True).sum()
            return df_all, mem

        elif tier == "tier_1":
            result_all = []
            for _ in range(T):
                output = np.zeros(n_groups, dtype=np.float32)
                for i in range(N_vals.shape[0]):
                    output[group_ids[i]] += N_vals[i]
                result_all.append(output)
            mem = sum(arr.nbytes for arr in result_all)
            return result_all, mem

        elif tier == "tier_2":
            result = np.zeros((T, n_groups), dtype=np.float32)
            for t in range(T):
                np.add.at(result[t], group_ids, N_vals)
            return result, result.nbytes

        elif tier == "tier_3":
            result = np.zeros((T, n_groups), dtype=np.float32)
            for t in range(T):
                result[t] = grouped_sum_serial(N_vals, group_ids, n_groups)
            return result, result.nbytes

        elif tier == "tier_3p":
            result = np.zeros((T, n_groups), dtype=np.float32)
            for t in range(T):
                result[t] = grouped_sum_parallel(N_vals, group_ids, n_groups)
            return result, result.nbytes

    elif func_name == "grouped_count":
        if tier == "tier_0":
            df_all = pd.DataFrame()
            for _ in range(T):
                df_step = df_base.groupby(["age", "region", "sex"]).size().reset_index(name='count')
                df_all = pd.concat([df_all, df_step], ignore_index=True)
            mem = df_all.memory_usage(deep=True).sum()
            return df_all, mem

        elif tier == "tier_1":
            result_all = []
            for _ in range(T):
                output = np.zeros(n_groups, dtype=np.int32)
                for i in range(group_ids.shape[0]):
                    output[group_ids[i]] += 1
                result_all.append(output)
            mem = sum(arr.nbytes for arr in result_all)
            return result_all, mem

        elif tier == "tier_2":
            result = np.zeros((T, n_groups), dtype=np.int32)
            for t in range(T):
                np.add.at(result[t], group_ids, 1)
            return result, result.nbytes

        elif tier == "tier_3":
            result = np.zeros((T, n_groups), dtype=np.int32)
            for t in range(T):
                result[t] = grouped_count_serial(group_ids, n_groups)
            return result, result.nbytes

        elif tier == "tier_3p":
            result = np.zeros((T, n_groups), dtype=np.int32)
            for t in range(T):
                result[t] = grouped_count_parallel(group_ids, n_groups)
            return result, result.nbytes

    elif func_name == "masked_sum":
        if tier == "tier_0":
            results = []
            for _ in range(T):
                results.append(df_base.loc[mask, "N"].sum())
            result_array = np.array(results, dtype=np.float32)
            return result_array, result_array.nbytes

        elif tier == "tier_1":
            results = []
            for _ in range(T):
                s = 0.0
                for i in range(N_vals.shape[0]):
                    if mask[i]:
                        s += N_vals[i]
                results.append(s)
            result_array = np.array(results, dtype=np.float32)
            return result_array, result_array.nbytes

        elif tier == "tier_2":
            results = np.zeros(T, dtype=np.float32)
            for t in range(T):
                results[t] = (N_vals * mask).sum()
            return results, results.nbytes

        elif tier == "tier_3":
            result = np.zeros(T, dtype=np.float32)
            for t in range(T):
                result[t] = masked_sum_serial(N_vals, mask)
            return result, result.nbytes

        elif tier == "tier_3p":
            result = np.zeros(T, dtype=np.float32)
            for t in range(T):
                result[t] = masked_sum_parallel(N_vals, mask)
            return result, result.nbytes

    elif func_name == "apply_deterministic_transition":
        if tier == "tier_0":
            df_all = pd.DataFrame()
            for _ in range(T):
                new_N = df_base["N"].values * probs
                df_step = df_base.copy()
                df_step["N"] = new_N
                df_all = pd.concat([df_all, df_step], ignore_index=True)
            mem = df_all.memory_usage(deep=True).sum()
            return df_all, mem

        elif tier == "tier_1":
            result = np.zeros((T, N_vals.shape[0]), dtype=np.float32)
            for t in range(T):
                for i in range(N_vals.shape[0]):
                    result[t, i] = N_vals[i] * probs[i]
            return result, result.nbytes

        elif tier == "tier_2":
            result = np.zeros((T, N_vals.shape[0]), dtype=np.float32)
            for t in range(T):
                result[t] = N_vals * probs
            return result, result.nbytes

        elif tier == "tier_3":
            result = np.zeros((T, N_vals.shape[0]), dtype=np.float32)
            for t in range(T):
                result[t] = apply_deterministic_transition(N_vals, probs)
            return result, result.nbytes

    elif func_name == "apply_stochastic_transition":
        if tier == "tier_0":
            df_all = pd.DataFrame()
            for _ in range(T):
                new_N = [np.random.binomial(int(N_vals[i]), probs[i]) for i in range(N_vals.shape[0])]
                df_step = df_base.copy()
                df_step["N"] = new_N
                df_all = pd.concat([df_all, df_step], ignore_index=True)
            mem = df_all.memory_usage(deep=True).sum()
            return df_all, mem

        elif tier == "tier_1":
            result = np.zeros((T, N_vals.shape[0]), dtype=np.int32)
            for t in range(T):
                for i in range(N_vals.shape[0]):
                    result[t, i] = np.random.binomial(int(N_vals[i]), probs[i])
            return result, result.nbytes

        elif tier == "tier_2":
            result = np.zeros((T, N_vals.shape[0]), dtype=np.int32)
            for t in range(T):
                result[t] = np.random.binomial(N_vals.astype(np.int32), probs)
            return result, result.nbytes

        elif tier == "tier_3":
            result = np.zeros((T, N_vals.shape[0]), dtype=np.int32)
            for t in range(T):
                result[t] = apply_stochastic_transition_serial(N_vals.astype(np.int32), probs)
            return result, result.nbytes

    else:
        raise ValueError(f"Unknown function or tier: {func_name}, {tier}")

def run_benchmarks():
    Ns = [10**4, 10**5, 10**6, 10**7]
    Ts = [100, 300, 500, 700]
    funcs_and_tiers = {
        "grouped_sum": ["tier_0", "tier_1", "tier_2", "tier_3", "tier_3p"],
        "grouped_count": ["tier_0", "tier_1", "tier_2", "tier_3", "tier_3p"],
        "masked_sum": ["tier_0", "tier_1", "tier_2", "tier_3", "tier_3p"],
        "apply_deterministic_transition": ["tier_0", "tier_1", "tier_2", "tier_3"],
        "apply_stochastic_transition": ["tier_0", "tier_1", "tier_2", "tier_3"]
    }

    for N in Ns:
        for T in Ts:
            N_vals, probs, mask, group_ids, n_groups, df_base = setup_data(N)
            for func_name, tiers in funcs_and_tiers.items():
                for tier in tiers:
                    args = (func_name, tier, N_vals, probs, mask, group_ids, n_groups, df_base, T)

                    fn_hash = hash_fn_source(lambda: run_benchmark_case(*args))

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
                                    if rss > peak_rss:
                                        peak_rss = rss
                                    if rss > (MAX_MEMORY_MB * 1024 ** 2):
                                        print(f"[{func_name} - {tier}] N={N}, T={T} | MEMORY CAP EXCEEDED: {rss / 1024 ** 2:.2f} MB > {MAX_MEMORY_MB} MB")
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
                                if peak_rss >= (MAX_MEMORY_MB * 1024 ** 2):
                                    status = "MEMORY_CAP"
                                else:
                                    status = "FAILED"
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
                            print(f"[{func_name} - {tier}] N={N}, T={T} | TIMEOUT after {MAX_TIME} seconds")

                        finally:
                            monitor_thread.join()
                            os.unlink(f_args.name)
                            os.unlink(f_result.name)

                        save_benchmark_result(
                            tier=tier,
                            N=N,
                            T=T,
                            elapsed=elapsed,
                            peak_memory=dynamic_mem,
                            result_shape=getattr(result, "shape", "n/a") if result is not None else "n/a",
                            cap_time=MAX_TIME,
                            fn_hash=fn_hash,
                            function_name=func_name
                        )

                        print(f"[{func_name} - {tier}] N={N}, T={T} | Time: {elapsed:.2f}s | Mem: {dynamic_mem / 1024 ** 2:.4f}MB | Status: {status}")

if __name__ == "__main__":
    if len(sys.argv) == 3:
        worker_main(sys.argv[1], sys.argv[2])
    else:
        run_benchmarks()
