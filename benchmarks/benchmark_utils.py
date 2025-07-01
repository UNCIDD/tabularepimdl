import os
import json
import platform
import subprocess
from datetime import datetime
import hashlib
import inspect
import numpy as np
import scipy as sp
import importlib
import seaborn as sns
import matplotlib.pyplot as plt
import cmasher as cmr
import pandas as pd
import glob

ENV_INFO = {
    "platform": platform.platform(),
    "python_version": platform.python_version(),
    "scipy_version": sp.__version__,
    "numpy_version": np.__version__,
    "numba_version": importlib.import_module("numba").__version__,
    "git_commit": None,
}

try:
    ENV_INFO["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).decode("utf-8").strip()
except Exception:
    pass

def generate_results_file(category):
    """
    Create a results file path inside benchmarks/runs/<category>/ with timestamp.
    """
    run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = os.path.join("benchmarks", "runs", category)
    os.makedirs(run_dir, exist_ok=True)
    results_file = os.path.join(run_dir, f"benchmark_results_{run_id}.json")
    return results_file, run_id

def hash_fn_source(fn):
    try:
        src = inspect.getsource(fn)
        return hashlib.sha256(src.encode()).hexdigest()
    except Exception:
        return None

def save_benchmark_result(
    results_file,
    tier, N, T,G, elapsed, peak_memory, result_shape,
    cap_time=None, fn_hash=None, max_memory_mb=28000, function_name=None
):
    if elapsed is None or elapsed <= 0:
        elapsed = (cap_time + 1) if cap_time else 1e-6
    if peak_memory is None or peak_memory <= 0:
        peak_memory = 4096  # 4 KB minimum

    peak_MB = peak_memory / 1024**2

    statuses = []
    if cap_time and elapsed > cap_time:
        statuses.append("fail_time")
    if peak_MB > max_memory_mb:
        statuses.append("fail_memory")
    if cap_time and elapsed > cap_time and result_shape == "n/a":
        statuses.append("exception")

    status = "ok" if not statuses else ",".join(statuses)

    record = {
        "timestamp": datetime.now().isoformat(),
        "function_name": function_name,
        "tier": tier,
        "N": int(N),
        "T": int(T) if T is not None else None,
        "G": int(G) if G is not None else None,
        "wall_time": round(min(elapsed, cap_time) if cap_time else elapsed, 8),
        "peak_memory_MB": round(peak_MB, 8),
        "result_shape": str(result_shape),
        "status": status,
        "fn_hash": fn_hash,
        **ENV_INFO
    }

    with open(results_file, "a") as f:
        f.write(json.dumps(record) + "\n")




def load_benchmark_results(category):
    """
    Load benchmark results from all JSON files in a given category folder.
    """
    run_dir = f"benchmarks/runs/{category}"
    files = glob.glob(os.path.join(run_dir, "benchmark_results_*.json"))
    records = []
    for fpath in files:
        with open(fpath) as f:
            for line in f:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def get_latest_benchmark_file(category):
    directory=f"benchmarks/runs/{category}"
    files = glob.glob(os.path.join(directory, "benchmark_results_*.json"))
    if not files:
        raise FileNotFoundError(f"No benchmark result files found in {directory}")
    latest = max(files, key=os.path.getmtime)
    return latest


def plot_benchmark_heatmap(
    df,
    func_name,
    category,
    value_col="wall_time",
    save_path=None,
    styles_expected=None,
    secondary_axis="T"  # allows overriding 'T' as the column used for pivot columns
):
    """
    Plot benchmark heatmap for a function, with cap indication and annotations.

    Parameters:
    - df: DataFrame with benchmark results
    - func_name: str, name of the function being benchmarked
    - category: str, benchmark category (e.g. 'arrayops', 'sir_network')
    - value_col: 'wall_time' or 'peak_memory_MB'
    - save_path: optional path to save (if None, auto path is generated)
    - styles_expected: list of code styles to plot (auto-detected if None)
    - secondary_axis: str, column to use as heatmap x-axis (default: 'T')
    """
    if value_col == "wall_time":
        cap_value = 60.0
        title_suffix = r"(seconds, $\geq$ 60 means time cap hit)"
    elif value_col == "peak_memory_MB":
        cap_value = 2800.0
        title_suffix = r"(MB, $\geq$ 2800 means memory cap hit)"
    else:
        raise ValueError("value_col must be 'wall_time' or 'peak_memory_MB'")

    if secondary_axis not in df.columns:
        raise ValueError(f"Secondary axis '{secondary_axis}' not found in DataFrame columns.")

    sns.set(style="white", font_scale=1.5)

    if styles_expected is None:
        styles_expected = list(df["Code Style"].unique())

    n_styles = len(styles_expected)
    n_cols = min(3, n_styles)
    n_rows = int(np.ceil(n_styles / n_cols))

    vmin = np.log10(df[value_col].clip(lower=1e-6)).min()
    vmax = np.log10(df[value_col].clip(lower=1e-6)).max()

    # Prepass to determine heatmap shape and dynamically scale figure size
    pivot_sample = df[df["Code Style"] == styles_expected[0]].pivot(index="N", columns=secondary_axis, values=value_col)
    n_x = pivot_sample.columns.size
    n_y = pivot_sample.index.size
    fig_width = max(5, n_x * 0.9)
    fig_height = max(3, n_y * 0.7)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_width * n_cols, fig_height * n_rows),
        constrained_layout=True
    )
    axes = np.array(axes).reshape(-1)

    for idx, style in enumerate(styles_expected):
        ax = axes[idx]
        sub = df[df["Code Style"] == style]
        if sub.empty:
            ax.axis("off")
            ax.set_title(f"{style}\n(no data)")
            continue

        pivot = sub.pivot(index="N", columns=secondary_axis, values=value_col)
        log_data = np.log10(pivot.clip(lower=1e-6))

        annot_data = pivot.applymap(
            lambda x: f"≥{int(cap_value)}" if x >= cap_value else (f"{x:.1E}" if x < 0.01 else f"{x:.2f}")
        )

        sns.heatmap(
            log_data,
            annot=annot_data,
            fmt="",
            cmap=cmr.iceburn,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            ax=ax,
            annot_kws={"size": 10}
        )

        for y, row in enumerate(pivot.index):
            for x, col in enumerate(pivot.columns):
                if pivot.loc[row, col] >= cap_value:
                    ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=False, edgecolor='red', lw=3))

        ax.set_title(style)

        row_idx = idx // n_cols
        col_idx = idx % n_cols

        if col_idx == 0:
            ax.set_ylabel("N (rows)")
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        if row_idx == n_rows - 1:
            ax.set_xlabel(f"{secondary_axis}")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

    for ax in axes[n_styles:]:
        ax.axis("off")

    fig.suptitle(f"{func_name} benchmark {title_suffix}", fontsize=16)

    if save_path is None:
        plot_dir = f"benchmarks/plots/{category}"
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, f"{func_name}_{value_col}_heatmap.png")

    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
