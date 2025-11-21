import numpy as np
import pandas as pd
import pytest
import tracemalloc
import time
import gc
import json
import matplotlib.pyplot as plt
import seaborn as sns

from tabularepimdl.EpiModel import EpiModel as EpiModel_pd
from tabularepimdl.SimpleInfection import SimpleInfection as SimpleInfection_pd
from tabularepimdl.SimpleTransition import SimpleTransition as SimpleTransition_pd

from tabularepimdl.SimpleInfection_Vec_Encode import SimpleInfection_Vec_Encode
from tabularepimdl.SimpleTransition_Vec_Encode import SimpleTransition_Vec_Encode

from tabularepimdl.EpiModel_Vec_Encode1 import EpiModel_Vec_Encode_1 #model vec engine 1
from tabularepimdl.EpiModel_Vec_Encode2 import EpiModel_Vec_Encode_2 #model vec engine 2

#Global Setup
iters = 200

infection_rate = 0.2
transition_rate = 0.25
infstate_compartments = ['S', 'I', 'R']

# Store expected output globally for reuse
#pandas_result = None
benchmark_results = []


@pytest.fixture
def build_pandas_model(init_df):
    """EpiModel Pandas version"""
    init_state, _ = init_df
    infect_rule_pd = SimpleInfection_pd(beta=infection_rate, column='InfState', s_st='S', i_st='I', inf_to='I')
    recover_rule_pd = SimpleTransition_pd(column='InfState', from_st='I', to_st='R', rate=transition_rate)
    determ_epi_mdl_pd = EpiModel_pd(init_state = init_state, rules=[[infect_rule_pd, recover_rule_pd]])

    return determ_epi_mdl_pd

@pytest.fixture
def build_vec1_model(init_df):
    """EpiModel Vec version 1"""
    init_state, _ = init_df
    infect_rule_vec = SimpleInfection_Vec_Encode(beta=infection_rate, column='InfState', s_st='S', i_st='I', inf_to='I', infstate_compartments=infstate_compartments)
    recover_rule_vec = SimpleTransition_Vec_Encode(column='InfState', from_st='I', to_st='R', rate=transition_rate, infstate_compartments=infstate_compartments)
    determ_epi_mdl_vec1 = EpiModel_Vec_Encode_1(init_state = init_state ,rules=[[infect_rule_vec, recover_rule_vec]], compartment_col = 'InfState')

    return determ_epi_mdl_vec1

@pytest.fixture
def build_vec2_model(init_df):
    """EpiModel Vec version 2"""
    init_state, _ = init_df
    infect_rule_vec = SimpleInfection_Vec_Encode(beta=infection_rate, column='InfState', s_st='S', i_st='I', inf_to='I', infstate_compartments=infstate_compartments)
    recover_rule_vec = SimpleTransition_Vec_Encode(column='InfState', from_st='I', to_st='R', rate=transition_rate, infstate_compartments=infstate_compartments)
    determ_epi_mdl_vec2 = EpiModel_Vec_Encode_2(init_state = init_state ,rules=[[infect_rule_vec, recover_rule_vec]], compartment_col = 'InfState')

    return determ_epi_mdl_vec2

# -------------------
# Benchmark collector fixture
# -------------------
@pytest.fixture(scope="session", autouse=True)
def benchmark_results():
    """Collect results across tests into a list."""
    results = []
    yield results
    # After all tests, plot summary

    if results:
        df = pd.DataFrame(results)
        df['label'] = df.apply(lambda row: f"{int(row['n']):,} rows \n{row['iters']} iters", axis=1)

    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ---- Time plot ----
    sns.barplot(
            data=df,
            x="label",
            y="time_sec",
            hue="model",
            ax=axes[0]
        )
    axes[0].set_title("Runtime (seconds) by Model")
    axes[0].set_xlabel("Data Size & Iterations")
    axes[0].set_ylabel("Time (seconds)")
    axes[0].tick_params(axis='x', rotation=30)

    # ---- Memory plot ----
    sns.barplot(
            data=df,
            x="label",
            y="peak_memory_mb",
            hue="model",
            ax=axes[1]
        )
    axes[1].set_title("Peak Memory (MB) by Model")
    axes[1].set_xlabel("Data Size & Iterations")
    axes[1].set_ylabel("Memory (MB)")
    axes[1].tick_params(axis='x', rotation=30)

        # Add legends
    axes[0].legend(title="Backend")
    axes[1].legend(title="Backend")

    plt.tight_layout()
    plt.show()
    #plt.savefig("benchmark_EpiModel_Pandas_vs_Vec_summary.png") #for picture saving purpose
    #print("\nBenchmark summary figure saved as benchmark_EpiModel_Pandas_vs_Vec_summary.png") #for picture saving purpose

# -------------------
#Parameterized test
# -------------------
@pytest.mark.parametrize("model_label, model_fixture_name", [
    ("pandas", "build_pandas_model"),
    ("vec1", "build_vec1_model"),
    ("vec2", "build_vec2_model")
])
def test_model_performance_and_output(request, model_label, model_fixture_name, benchmark_results, init_df):
    global pandas_result
    _, n = init_df  #retrieve n value
    
    print("\n=== Running test for model:", model_label, "===")

    # Load the model using the name of the fixture
    model = request.getfixturevalue(model_fixture_name)
    #print('model is\n', model.__class__.__name__)
    #print('model population', model.init_state)

    # Time and memory tracking
    tracemalloc.start()
    t0 = time.perf_counter()

    for _ in range(iters):
        model.do_timestep(dt=1.0)

    t1 = time.perf_counter()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    runtime = round(t1 - t0, 3)
    peak_mb = round(peak / 1024**2, 2)

    print(f"# Model: {model_fixture_name}")
    print(f"# Time (s): {runtime}")
    print(f"# Peak Memory (MB): {peak_mb}")

    # Save result for charting
    benchmark_results.append({
        "model": model_label,
        "time_sec": runtime,
        "peak_memory_mb": peak_mb,
        "n" : n,
        "iters": iters
    })

    # Output comparison
    if model_label == "pandas":
        #print("\n=== Pandas Model full_epi ===")
        #print(model.full_epi)
        pandas_sorted = model.full_epi.sort_values(by=['T', 'InfState'], ascending=[True, True]) #sort values before comparing pandas and array result to make sure the order of N lines up
        #pandas_result = model.full_epi['N'].round(3).values
        pandas_result = pandas_sorted['N'].round(3).values
    else:
        #arr = model._covnert_list_of_arrays_to_df(model._full_epi_list)['N'].round(3).values
        arr_sorted = model._covnert_list_of_arrays_to_df(model._full_epi_list).sort_values(by=['T', 'InfState'], ascending=[True, True]) #sort values before comparing pandas and array result to make sure the order of N lines up
        #print(f"\n=== {model_label} Model full_epi ===")
        #print(model._covnert_list_of_arrays_to_df(model._full_epi_list))
        #assert np.allclose(arr, pandas_result, rtol=1e-3), f"{model_label} does not match pandas model"
        arr_result = arr_sorted['N'].round(3).values
        assert np.array_equal(arr_result, pandas_result), "Values do not match after rounding"

#def pytest_sessionfinish(session, exitstatus):
#    """Called after the whole test run finishes."""
#    if benchmark_results:
#        with open("benchmark_results.json", "w") as f:
#            json.dump(benchmark_results, f, indent=2)
#        print("\nBenchmark results saved to benchmark_results.json")

#pytest -s benchmark_EpiModel_Pandas_Vec_test.py
#pytest -s benchmark_EpiModel_Pandas_Vec_test.py --switch=random