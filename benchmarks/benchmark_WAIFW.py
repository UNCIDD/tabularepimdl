import numpy as np
import pandas as pd
import pytest
import tracemalloc
import gc

from tabularepimdl.EpiModel import EpiModel as EpiModel_Base
from tabularepimdl.EpiModel_Vector import EpiModel as EpiModel_Vector
from tabularepimdl.SimpleTransition import SimpleTransition as SimpleTransition_Base
from tabularepimdl.SimpleTransition_Vector import SimpleTransition as SimpleTransition_Vector
from tabularepimdl.WAIFWTransmission import WAIFWTransmission as WAIFW_Base
from tabularepimdl.WAIFWTransmission_Vector import WAIFWTransmission as WAIFW_Vector

CALIFORNIA_CSV = "~/Documents/California_df.csv"
BETA0, BETA1, BETA2 = 0.01, 0.02, 0.005
RECOVERY_DAYS = 7.0
TF = 50.0
DT = 0.25

@pytest.fixture(scope="module")
def california_df():
    return pd.read_csv(CALIFORNIA_CSV)

@pytest.fixture(scope="module")
def waifw_matrix(california_df):
    df = california_df
    groups = df['Group number'].unique()
    n_groups = len(groups)

    cattle_pops = [
        df[(df['Host'] == 'cattle') & (df['Group number'] == num)]['N'].sum()
        for num in df[df['Host'] == 'cattle']['Group number'].unique()
    ]
    cattle_pops = np.array(cattle_pops)

    waifw = np.zeros((n_groups, n_groups))
    count = 0
    for i in range(1, n_groups):
        for j in range(2, n_groups):
            if i == j:
                waifw[i, j] = BETA0 / cattle_pops[count]
                count += 1
            elif i < 2 and i != j:
                waifw[i, j] = BETA1 / df[df['Host'] == 'dairy_worker']['N'].values[0]
            elif i >= 2 and i != j:
                waifw[i, j] = BETA2 / cattle_pops.sum()
    return waifw

def build_waifw_model(rule_cls, transition_cls, df, waifw_matrix, is_vector, store_stride=None):
    df_copy = df.copy()

    if is_vector:
        df_copy = df_copy.rename(columns={"InfState": "compartment"})
        inf_col = "compartment"
        compartment_col = "compartment"
    else:
        inf_col = "InfState"
        compartment_col = None  # unused

    recovery = transition_cls(
        column=inf_col,
        from_st="I", to_st="R",
        rate=1.0 / RECOVERY_DAYS,
        stochastic=False
    )

    transmission = rule_cls(
        waifw_matrix=waifw_matrix,
        inf_col=inf_col,
        group_col="Group number",
        s_st="S", i_st="I", inf_to="I",
        stochastic=False
    )

    if is_vector:
        return EpiModel_Vector(
            init_state=df_copy,
            rules=[[transmission], [recovery]],
            t0=0.0,
            tf=TF,
            dt=DT,
            store_stride=store_stride,
            compartment_col=compartment_col
        )
    else:
        return EpiModel_Base(
            init_state=df_copy,
            rules=[[transmission], [recovery]]
        )

@pytest.mark.parametrize("label, is_vector, rule_cls, transition_cls", [
    ("vector", True, WAIFW_Vector, SimpleTransition_Vector),
    ("base", False, WAIFW_Base, SimpleTransition_Base),
])
def test_waifw_model_benchmark(benchmark, california_df, waifw_matrix, label, is_vector, rule_cls, transition_cls):
    model = build_waifw_model(rule_cls, transition_cls, california_df, waifw_matrix, is_vector, store_stride=4)

    gc.collect()
    tracemalloc.start()
    start_current, start_peak = tracemalloc.get_traced_memory()

    if is_vector:
        benchmark(lambda: model.run())
    else:
        def run_base():
            steps = int(TF / DT)
            for _ in range(steps):
                model.do_timestep(dt=DT)
        benchmark(run_base)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    init_MB = start_peak / 1024 / 1024
    total_MB = peak / 1024 / 1024
    delta_MB = total_MB - init_MB

    print(f"[{label.upper()}] Init memory: {init_MB:.3f} MB")
    print(f"[{label.upper()}] Peak total memory: {total_MB:.3f} MB")
    print(f"[{label.upper()}] Post-init growth: {delta_MB:.3f} MB")

    benchmark.extra_info[f"{label}_memory_MB_init"] = round(init_MB, 3)
    benchmark.extra_info[f"{label}_memory_MB_peak"] = round(total_MB, 3)
    benchmark.extra_info[f"{label}_memory_MB_growth"] = round(delta_MB, 3)
