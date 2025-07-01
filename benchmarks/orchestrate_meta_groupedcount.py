# file: orchestrate_meta_groupedcount.py

import benchmark_meta_groupedcount
import benchmark_meta_groupedcount_plot

def main():
    # Run the grouped sum benchmarks (array vs matrix)
    benchmark_meta_groupedcount.run_benchmarks()

    # Generate plots from the benchmark results
    benchmark_meta_groupedcount_plot.main()

if __name__ == "__main__":
    main()
