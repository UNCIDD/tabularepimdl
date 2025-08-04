import numpy as np
import pandas as pd
import pytest
import tracemalloc

from tabularepimdl.EpiModel_Vector import EpiModel as EpiModel
from tabularepimdl.SimpleInfection_Vector import SimpleInfection as SimpleInfection
from tabularepimdl.SimpleTransition_Vector import SimpleTransition as SimpleTransition

from tabularepimdl.EpiModel import EpiModel as EpiModel_Base
from tabularepimdl.SimpleInfection import SimpleInfection as SimpleInfection_Base
from tabularepimdl.SimpleTransition import SimpleTransition as SimpleTransition_Base

BETA = 0.25
GAMMA = 1/7

def get_init_df():
    return pd.DataFrame({
        "compartment": ["S", "I"],
        "N": [1_500_000, 10],
        "T": [0.0, 0.0]
    })

@pytest.fixture
def sir_vector_model():
    df_init = get_init_df()
    t0, tf, dt = 0.0, 500.0, 0.25

    rules = [[
        SimpleInfection(beta=BETA, column="compartment", s_st="S", i_st="I", inf_to="I"),
        SimpleTransition(rate=GAMMA, column="compartment", from_st="I", to_st="R")
    ]]

    model = EpiModel(
        init_state=df_init,
        rules=rules,
        t0=t0,
        tf=tf,
        dt=dt,
        compartment_col="compartment"
    )
    return model


@pytest.fixture
def sir_base_model():
    df_init = get_init_df()


    rules = [[
        SimpleInfection_Base(beta=BETA, column="compartment", s_st="S", i_st="I", inf_to="I"),
        SimpleTransition_Base(rate=GAMMA, column="compartment", from_st="I", to_st="R")
    ]]

    model = EpiModel_Base(
        init_state=df_init,
        rules=rules,
    )
    return model


def test_vector_model_benchmark(benchmark, sir_vector_model):
    """Benchmark wall time and memory for vectorized model."""
    tracemalloc.start()
    benchmark(lambda: sir_vector_model.run())
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"[BENCHMARK] Vector model peak memory: {peak / 1024 / 1024:.3f} MB")
    benchmark.extra_info["vector_peak_memory_MB"] = round(peak / 1024 / 1024, 3)


def test_base_model_benchmark(benchmark, sir_base_model):
    """Benchmark wall time and memory for base model."""
    dt = 0.25
    tf = 500.0
    n_steps = int(tf / dt)

    def run():
        for _ in range(n_steps):
            sir_base_model.do_timestep(dt=dt)

    tracemalloc.start()
    benchmark(run)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"[BENCHMARK] Base model peak memory: {peak / 1024 / 1024:.3f} MB")
    benchmark.extra_info["base_peak_memory_MB"] = round(peak / 1024 / 1024, 3)

def test_compare_memory_and_runtime(benchmark_session):
    """Compare memory and runtime between vector and base models."""
    vector_bench = None
    base_bench = None

    for b in benchmark_session.stats:
        if b.name == "test_vector_model_benchmark":
            vector_bench = b
        elif b.name == "test_base_model_benchmark":
            base_bench = b

    assert vector_bench and base_bench, "Missing benchmark results"

    # --- Time comparison ---
    vector_time = vector_bench.stats["mean"]
    base_time = base_bench.stats["mean"]
    time_ratio = base_time / vector_time
    print(f"[COMPARE] Runtime speedup: {time_ratio:.2f}× faster (vector over base)")

    # --- Memory comparison ---
    vmem = vector_bench.extra_info.get("vector_peak_memory_MB")
    bmem = base_bench.extra_info.get("base_peak_memory_MB")

    if vmem and bmem:
        mem_ratio = bmem / vmem
        print(f"[COMPARE] Memory usage: {vmem:.2f} MB (vector) vs {bmem:.2f} MB (base)")
        print(f"[COMPARE] Memory improvement: {mem_ratio:.2f}× smaller (vector over base)")
    else:
        print("[WARN] No memory data available in benchmark.extra_info")
