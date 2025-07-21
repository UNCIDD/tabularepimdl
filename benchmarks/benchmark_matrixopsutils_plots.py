import os
import pandas as pd
from benchmark_utils import (
    load_benchmark_results,
    plot_benchmark_heatmap,
    get_latest_benchmark_file,
)

def main():
    category = "matrixops"
    latest_file = get_latest_benchmark_file(category)
    df = pd.read_json(latest_file, lines=True)

    # Get all unique function names
    fnames = df["function_name"].unique()

    for fname in fnames:
        sub_df = df[df["function_name"] == fname].copy()
        sub_df["Code Style"] = sub_df["tier"].map(
            lambda x: "Dense" if x == "dense" else "Sparse"
        )
        # Plot wall time
        plot_benchmark_heatmap(
            df=sub_df,
            func_name=fname,
            category=category,
            value_col="wall_time"
        )

        # Plot memory
        plot_benchmark_heatmap(
            df=sub_df,
            func_name=fname,
            category=category,
            value_col="peak_memory_MB"
        )

if __name__ == "__main__":
    main()

