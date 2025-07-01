# file: orchestrate_matrixopsutils.py

import benchmark_matrixopsutils
import benchmark_matrixopsutils_plots

def main():
    # Run all matrixops utility benchmarks
    benchmark_matrixopsutils.run_matrixops_benchmarks()

    # Generate plots from the benchmark results
    benchmark_matrixopsutils_plots.main()

if __name__ == "__main__":
    main()
