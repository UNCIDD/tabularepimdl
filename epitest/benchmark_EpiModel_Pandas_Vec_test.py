import numpy as np
import pandas as pd
import pytest
import tracemalloc
import time
import gc
import json
import matplotlib.pyplot as plt

from tabularepimdl.EpiModel import EpiModel as EpiModel_pd
from tabularepimdl.SimpleInfection import SimpleInfection as SimpleInfection_pd
from tabularepimdl.SimpleTransition import SimpleTransition as SimpleTransition_pd

from tabularepimdl.SimpleInfection_Vec_Encode import SimpleInfection_Vec_Encode
from tabularepimdl.SimpleTransition_Vec_Encode import SimpleTransition_Vec_Encode

from tabularepimdl.EpiModel_Vec_Encode1 import EpiModel_Vec_Encode_1 #model vec engine 1
from tabularepimdl.EpiModel_Vec_Encode2 import EpiModel_Vec_Encode_2 #model vec engine 2

#Global Setup
np.random.seed(3) #set seed to ensure a same epidemic each time of run
n = 10_000_000 #data size
infstate_values = np.random.choice(['S', 'I', 'R'], size=n)
n_values = np.random.randint(1, 10, size=n)
t_values = np.full(n, 0.0)
iters = 1000
switch = 'f'

infection_rate = 0.2
transition_rate = 0.25
infstate_compartments = ['S', 'I', 'R']

# Store expected output globally for reuse
#pandas_result = None
benchmark_results = []

def get_init_df():
    """Initial dataframe preparation."""
    if switch == 'fixed':
        population = pd.DataFrame({
        'N' : [1_500_000, 10],
        'T': [0, 0],
        'InfState' : ['S','I']
        })
    else:
        population = pd.DataFrame({
        'N' : n_values, 
        'T': t_values,
        'InfState' : infstate_values
        })
    return population

@pytest.fixture
def build_pandas_model():
    """EpiModel Pandas version"""
    init_state = get_init_df()
    infect_rule_pd = SimpleInfection_pd(beta=infection_rate, column='InfState', s_st='S', i_st='I', inf_to='I')
    recover_rule_pd = SimpleTransition_pd(column='InfState', from_st='I', to_st='R', rate=transition_rate)
    determ_epi_mdl_pd = EpiModel_pd(init_state = init_state, rules=[[infect_rule_pd, recover_rule_pd]])

    return determ_epi_mdl_pd

@pytest.fixture
def build_vec1_model():
    """EpiModel Vec version 1"""
    init_state = get_init_df()
    infect_rule_vec = SimpleInfection_Vec_Encode(beta=infection_rate, column='InfState', s_st='S', i_st='I', inf_to='I', infstate_compartments=infstate_compartments)
    recover_rule_vec = SimpleTransition_Vec_Encode(column='InfState', from_st='I', to_st='R', rate=transition_rate, infstate_compartments=infstate_compartments)
    determ_epi_mdl_vec1 = EpiModel_Vec_Encode_1(init_state = init_state ,rules=[[infect_rule_vec, recover_rule_vec]], compartment_col = 'InfState')

    return determ_epi_mdl_vec1

@pytest.fixture
def build_vec2_model():
    """EpiModel Vec version 2"""
    init_state = get_init_df()
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
        labels = [f"{r['model']}\nN={r['n']}, iters={r['iters']}" for r in results]
        runtimes = [r['time_sec'] for r in results]
        peaks = [r['peak_memory_mb'] for r in results]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].bar(labels, runtimes, color='skyblue')
        axes[0].set_title("Runtime (s)")
        axes[0].set_ylabel("Seconds")

        axes[1].bar(labels, peaks, color='salmon')
        axes[1].set_title("Peak Memory (MB)")
        axes[1].set_ylabel("MB")

        plt.tight_layout()
        plt.savefig("benchmark_summary.png")
        print("\nBenchmark summary figure saved as benchmark_summary.png")

# -------------------
#Parameterized test
# -------------------
@pytest.mark.parametrize("model_label, model_fixture_name", [
    ("pandas", "build_pandas_model"),
    ("vec1", "build_vec1_model"),
    ("vec2", "build_vec2_model")
])
def test_model_performance_and_output(request, model_label, model_fixture_name, benchmark_results):
    global pandas_result
    
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
        pandas_result = model.full_epi['N'].round(3).values
    else:
        arr = model._covnert_list_of_arrays_to_df(model._full_epi_list)['N'].round(3).values
        #print(f"\n=== {model_label} Model full_epi ===")
        #print(model._covnert_list_of_arrays_to_df(model._full_epi_list))
        assert np.allclose(arr, pandas_result, rtol=1e-3), f"{model_label} does not match pandas model"

#def pytest_sessionfinish(session, exitstatus):
#    """Called after the whole test run finishes."""
#    if benchmark_results:
#        with open("benchmark_results.json", "w") as f:
#            json.dump(benchmark_results, f, indent=2)
#        print("\nBenchmark results saved to benchmark_results.json")

#pytest -s benchmark_EpiModel_Pandas_Vec_test.py