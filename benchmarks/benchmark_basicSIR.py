import numpy as np
import pandas as pd
import pytest
import tracemalloc
import gc

from tabularepimdl.EpiModel_Vector import EpiModel as EpiModel
from tabularepimdl.SimpleInfection_Vector import SimpleInfection as SimpleInfection
from tabularepimdl.SimpleTransition_Vector import SimpleTransition as SimpleTransition

from tabularepimdl.EpiModel import EpiModel as EpiModel_Base
from tabularepimdl.SimpleInfection import SimpleInfection as SimpleInfection_Base
from tabularepimdl.SimpleTransition import SimpleTransition as SimpleTransition_Base

BETA = 0.25
GAMMA = 1 / 7

def get_init_df():
    return pd.DataFrame({
        "compartment": ["S", "I"],
        "N": [1_500_000, 10],
        "T": [0.0, 0.0]
    })

def build_vector_model(store_stride):
    df_init = get_init_df()
    t0, tf, dt = 0.0, 500.0, 0.25

    rules = [[
        SimpleInfection(beta=BETA, column="compartment", s_st="S", i_st="I", inf_to="I"),
        SimpleTransition(rate=GAMMA, column="compartment", from_st="I", to_st="R")
    ]]

    return EpiModel(
        init_state=df_init,
        rules=rules,
        t0=t0,
        tf=tf,
        dt=dt,
        store_stride=store_stride,
        compartment_col="compartment"
    )


@pytest.fixture
def sir_base_model():
    df_init = get_init_df()
    rules = [[
        SimpleInfection_Base(beta=BETA, column="compartment", s_st="S", i_st="I", inf_to="I"),
        SimpleTransition_Base(rate=GAMMA, column="compartment", from_st="I", to_st="R")
    ]]

    return EpiModel_Base(init_state=df_init, rules=rules)


@pytest.mark.parametrize("label, store_stride", [
    ("every_step", 1),
    ("daily", int(round(1.0 / 0.25))),      # dt = 0.25, so every 4 steps
    ("weekly", int(round(7.0 / 0.25)))      # dt = 0.25, so every 28 steps
])
def test_vector_model_benchmark(benchmark, label, store_stride):
    """Benchmark wall time and memory for vectorized model at different store frequencies."""
    model = build_vector_model(store_stride)

    gc.collect()
    tracemalloc.start()

    # Measure baseline (initialization) memory
    start_current, start_peak = tracemalloc.get_traced_memory()

    benchmark(lambda: model.run())

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    init_MB = start_peak / 1024 / 1024
    total_MB = peak / 1024 / 1024
    delta_MB = total_MB - init_MB

    print(f"[VECTOR-{label.upper()}] Init memory: {init_MB:.3f} MB")
    print(f"[VECTOR-{label.upper()}] Peak total memory: {total_MB:.3f} MB")
    print(f"[VECTOR-{label.upper()}] Post-init growth: {delta_MB:.3f} MB")

    benchmark.extra_info[f"vector_{label}_memory_MB_init"] = round(init_MB, 3)
    benchmark.extra_info[f"vector_{label}_memory_MB_peak"] = round(total_MB, 3)
    benchmark.extra_info[f"vector_{label}_memory_MB_growth"] = round(delta_MB, 3)


def test_base_model_benchmark(benchmark, sir_base_model):
    """Benchmark wall time and memory for base model."""
    dt = 0.25
    tf = 500.0
    n_steps = int(tf / dt)

    def run():
        for _ in range(n_steps):
            sir_base_model.do_timestep(dt=dt)

    gc.collect()
    tracemalloc.start()

    start_current, start_peak = tracemalloc.get_traced_memory()
    benchmark(run)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    init_MB = start_peak / 1024 / 1024
    total_MB = peak / 1024 / 1024
    delta_MB = total_MB - init_MB

    print(f"[BASE MODEL] Init memory: {init_MB:.3f} MB")
    print(f"[BASE MODEL] Peak total memory: {total_MB:.3f} MB")
    print(f"[BASE MODEL] Post-init growth: {delta_MB:.3f} MB")

    benchmark.extra_info["base_memory_MB_init"] = round(init_MB, 3)
    benchmark.extra_info["base_memory_MB_peak"] = round(total_MB, 3)
    benchmark.extra_info["base_memory_MB_growth"] = round(delta_MB, 3)


