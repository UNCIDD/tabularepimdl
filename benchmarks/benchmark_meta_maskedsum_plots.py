import os
import pandas as pd
from benchmark_utils import (
    get_latest_benchmark_file,
    plot_benchmark_heatmap,
)

def main():
    category = "array_vs_matrix_masked_sum"
    latest_file = get_latest_benchmark_file(category)
    df = pd.read_json(latest_file, lines=True)

    # Get all unique function names
    fnames = df["function_name"].unique()

    for fname in fnames:
        sub_df = df[df["function_name"] == fname].copy()

        # Drop entries with missing G or N
        sub_df = sub_df.dropna(subset=["N", "G"])
        sub_df = sub_df[sub_df["N"].apply(lambda x: isinstance(x, (int, float)))]
        sub_df = sub_df[sub_df["G"].apply(lambda x: isinstance(x, (int, float)))]

        # Map tiers to display names
        sub_df["Code Style"] = sub_df["tier"].map(
            lambda x: "Array" if x == "array" else "Matrix"
        )

        # Plot wall time heatmap
        plot_benchmark_heatmap(
            df=sub_df,
            func_name=fname,
            category=category,
            value_col="wall_time",
            secondary_axis="G"
        )

        # Plot memory heatmap
        plot_benchmark_heatmap(
            df=sub_df,
            func_name=fname,
            category=category,
            value_col="peak_memory_MB",
            secondary_axis="G"
        )

if __name__ == "__main__":
    main()
