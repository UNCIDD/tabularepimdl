# file: orchestrate_meta_maskedsum.py

import benchmark_meta_maskedsum
import benchmark_meta_maskedsum_plots

def main():
    # Run the grouped sum benchmarks (array vs matrix)
    benchmark_meta_maskedsum.run_benchmarks()

    # Generate plots from the benchmark results
    benchmark_meta_maskedsum_plots.main()

if __name__ == "__main__":
    main()
