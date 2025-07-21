import pandas as pd
from benchmarks.benchmark_utils import (
    get_latest_benchmark_file,
    load_benchmark_results,
    plot_benchmark_heatmap,
)

CATEGORY = "arrayops"

def load_and_prepare_benchmark(category):
    """Load benchmark results and map tier to descriptive code styles."""
    df = load_benchmark_results(category)

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
        raise ValueError(f"Unmapped tiers found in benchmark: {missing}")

    if "function_name" not in df.columns:
        raise ValueError("'function_name' column missing in benchmark data.")

    return df


def generate_plots_for_function(df, func_name, category):
    df_fn = df[df["function_name"] == func_name]
    if df_fn.empty:
        print(f"No data found for function {func_name}, skipping.")
        return

    print(f"Plotting: {func_name}")
    plot_benchmark_heatmap(df_fn, func_name, category, value_col="wall_time")
    plot_benchmark_heatmap(df_fn, func_name, category, value_col="peak_memory_MB")


if __name__ == "__main__":
    try:
        filepath = get_latest_benchmark_file(CATEGORY)
        print(f"Using latest benchmark file: {filepath}")
    except FileNotFoundError as e:
        print(str(e))
        exit(1)

    df = load_and_prepare_benchmark(CATEGORY)

    if df.empty:
        raise ValueError("Loaded benchmark produced empty DataFrame.")

    for func_name in df["function_name"].unique():
        generate_plots_for_function(df, func_name, CATEGORY)
