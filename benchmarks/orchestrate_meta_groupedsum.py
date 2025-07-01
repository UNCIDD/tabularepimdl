# file: orchestrate_meta_groupedsum.py

import benchmark_meta_groupedsum
import benchmark_meta_groupedsum_plots

def main():
    # Run the grouped sum benchmarks (array vs matrix)
    benchmark_meta_groupedsum.run_benchmarks()

    # Generate plots from the benchmark results
    benchmark_meta_groupedsum_plots.main()

if __name__ == "__main__":
    main()
