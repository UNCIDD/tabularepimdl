import subprocess

def run_benchmark_arrayops():
    print("=== Running benchmark_arrayops.py ===")
    subprocess.run(["python", "benchmarks/benchmark_arrayops.py"], check=True)
    print("=== benchmark_arrayops.py completed ===")

def run_benchmark_arrayops_plots():
    print("=== Running benchmark_arrayops_plots.py ===")
    subprocess.run(["python", "benchmarks/benchmark_arrayops_plots.py"], check=True)
    print("=== benchmark_arrayops_plots.py completed ===")

if __name__ == "__main__":
    run_benchmark_arrayops()
    run_benchmark_arrayops_plots()
    print("=== All benchmarking and plotting complete ===")
