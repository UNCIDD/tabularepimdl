import json
import pandas as pd
from tabularepimdl.benchmark_utils import plot_benchmark_heatmap
import glob
import os

def load_and_prepare_benchmark(filepath):
    """Load benchmark JSON lines and map to code styles."""
    with open(filepath, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise ValueError(f"Benchmark file {filepath} contains no valid records.")
    df = pd.DataFrame(records)

    tier_map = {
        "tier_0": "Legacy (Pandas)",
        "tier_1": "Legacy (Loop)",
        "tier_2": "Refactored (Vectorized)",
        "tier_3": "Refactored (Numba)",
        "tier_3p": "Refactored (Parallel Numba)"
    }
    df["Code Style"] = df["tier"].map(tier_map)

    if df["Code Style"].isnull().any():
        missing = df.loc[df["Code Style"].isnull(), "tier"].unique()
        raise ValueError(f"Unmapped tiers found in {filepath}: {missing}")

    if "function_name" not in df.columns:
        raise ValueError(f"'function_name' column missing in {filepath}.")

    return df

def get_latest_benchmark_file(directory="benchmarks/runs/"):
    files = glob.glob(os.path.join(directory, "benchmark_results_*.json"))
    if not files:
        raise FileNotFoundError(f"No benchmark result files found in {directory}")
    latest = max(files, key=os.path.getmtime)
    return latest

def generate_plots_for_function(df, func_name):
    df_fn = df[df["function_name"] == func_name]
    if df_fn.empty:
        print(f"No data found for function {func_name}, skipping.")
        return

    plot_benchmark_heatmap(df_fn, func_name, value_col="wall_time")
    plot_benchmark_heatmap(df_fn, func_name, value_col="peak_memory_MB")

if __name__ == "__main__":
    filepath = get_latest_benchmark_file()
    print(f"Using latest benchmark file: {filepath}")

    df = load_and_prepare_benchmark(filepath)

    if df.empty:
        raise ValueError(f"Loaded benchmark file {filepath} produced empty DataFrame.")

    unique_funcs = df["function_name"].unique()

    for func_name in unique_funcs:
        print(f"Generating plots for {func_name}...")
        generate_plots_for_function(df, func_name)

