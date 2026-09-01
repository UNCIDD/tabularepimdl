import time
import tracemalloc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from legacy.pandas_reference.BirthProcess import BirthProcess
from legacy.pandas_reference.EpiModel import EpiModel as EpiModel_pd

# from tabularepimdl.SimpleInfection import SimpleInfection as SimpleInfection_pd
from legacy.pandas_reference.SimpleTransition import SimpleTransition as SimpleTransition_pd
from legacy.pandas_reference.StateBasedDeathProcess import StateBasedDeathProcess
from legacy.pandas_reference.WAIFWTransmission import WAIFWTransmission

from tabularepimdl.BirthProcess_Vec_Encode import BirthProcess_Vec_Encode

# from tabularepimdl.EpiModel_Vec_Encode1_2 import EpiModel_Vec_Encode_1_2 #model vec engine 1 with pre-allocated buffer moved to do_timestep
from tabularepimdl.EpiModel_Vec_Encode1_5 import EpiModel_Vec_Encode_1_5

# from tabularepimdl.SimpleInfection_Vec_Encode import SimpleInfection_Vec_Encode
from tabularepimdl.SimpleTransition_Vec_Encode import SimpleTransition_Vec_Encode
from tabularepimdl.StateBasedDeathProcess_Vec_Encode import StateBasedDeathProcess_Vec_Encode
from tabularepimdl.WAIFWTransmission_Vec_Encode_Bincount import WAIFWTransmission_Vec_Encode_Bincount

# Global Setup
start = 0
end = 1  # used 25 to run a long simulaiton
iters = int((end - start) / (1 / 52))

infstate_compartments = ["S", "I", "R"]
trans_infect_compartments = ["S", "I", "R"]
# age_compartments_by_5 = ['0 to 4', '5 to 9', '10 to 14', '15 to 19', '20 to 24', '25 to 29', '30 to 34', '35 to 39', '40 to 44', '45 to 49', '50 to 54', '55 to 59', '60 to 64', '65 to 69', '70+']
age_compartments_by_10 = ["0 to 9", "10 to 19", "20 to 29", "30 to 39", "40 to 49", "50 to 59", "60 to 69", "70+"]
start_age = 0
end_age = 70
age_step = 10
column_to_sort = "AgeCat"
waifw = np.array(
    [
        [1, 1, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25],
        [1, 1, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25],
        [0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5],
        [0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5],
        [0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5],
        [0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5],
        [0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 1, 1],
        [0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 1, 1],
    ]
)

# data preparation
nc_like_pop_pd = pd.DataFrame(
    {
        "InfState": pd.Categorical(["S"] * 8, ["S", "I", "R"]),
        "AgeCat": [f"{i} to {i + 9}" for i in range(start_age, end_age, age_step)] + [f"{end_age}+"],
        "N": [1140000, 1320000, 1320000, 1290000, 1280000, 1280000, 1185000, 1175000],
        "T": 2023,  # question: do we want to assign T as an integer or float at the beginning?
    }
)
infect_init = pd.DataFrame({"InfState": "I", "AgeCat": "10 to 19", "N": [1], "T": 2023})
add_infect = pd.concat([nc_like_pop_pd, infect_init]).reset_index(drop=True)
add_infect = add_infect.sample(frac=1, random_state=37).reset_index(drop=True)
waifw = waifw * 18 * 26 / nc_like_pop_pd["N"].sum()

pandas_result: np.ndarray | None = None

# Rule setup  - Pandas based
nc_aging_rules = []
for i in range(start_age, end_age, age_step):
    if i < end_age - age_step:
        tmp = SimpleTransition_pd(column=column_to_sort, from_st=f"{i} to {i + 9}", to_st=f"{i + 10} to {i + 19}", rate=1 / 10)  # from 0 to 10, takes 10 years, so rate is 1/10
    else:
        tmp = SimpleTransition_pd(column=column_to_sort, from_st=f"{i} to {i + 9}", to_st=f"{end_age}+", rate=1 / 10)
    nc_aging_rules.append(tmp)

nc_birth = BirthProcess(rate=7 / 1000, start_state_sig=nc_like_pop_pd.iloc[[0]], stochastic=False)

nc_deaths = [
    StateBasedDeathProcess(columns=["AgeCat"], states=["0 to 9"], rate=1.5 / 1000, stochastic=False),
    StateBasedDeathProcess(columns=["AgeCat"], states=["10 to 19"], rate=0.1 / 1000, stochastic=False),
    StateBasedDeathProcess(columns=["AgeCat"], states=["20 to 29"], rate=1 / 1000, stochastic=False),
    StateBasedDeathProcess(columns=["AgeCat"], states=["30 to 39"], rate=1.9 / 1000, stochastic=False),
    StateBasedDeathProcess(columns=["AgeCat"], states=["40 to 49"], rate=2.7 / 1000, stochastic=False),
    StateBasedDeathProcess(columns=["AgeCat"], states=["50 to 59"], rate=4.9 / 1000, stochastic=False),
    StateBasedDeathProcess(columns=["AgeCat"], states=["60 to 69"], rate=10.7 / 1000, stochastic=False),
    StateBasedDeathProcess(columns=["AgeCat"], states=["70+"], rate=43.9 / 1000, stochastic=False),
]

age_mix_pd = WAIFWTransmission(waifw_matrix=waifw, inf_col="InfState", group_col="AgeCat")

recover_rule_pd = SimpleTransition_pd(column="InfState", from_st="I", to_st="R", rate=26)

# Rule setup  - Vec based
nc_aging_rules_vec = []
for i in range(start_age, end_age, age_step):
    if i < end_age - age_step:
        nc_aging_rule_vec = SimpleTransition_Vec_Encode(
            column="AgeCat", from_st=f"{i} to {i + 9}", to_st=f"{i + 10} to {i + 19}", rate=1 / 10, infstate_compartments=infstate_compartments, column_categories=age_compartments_by_10
        )
    else:
        nc_aging_rule_vec = SimpleTransition_Vec_Encode(
            column="AgeCat", from_st=f"{i} to {i + 9}", to_st=f"{end_age}+", rate=1 / 10, infstate_compartments=infstate_compartments, column_categories=age_compartments_by_10
        )
    nc_aging_rules_vec.append(nc_aging_rule_vec)

nc_birth_vec = BirthProcess_Vec_Encode(rate=7 / 1000, stochastic=False, column_to_sort=column_to_sort, infstate_compartments=infstate_compartments)

nc_deaths_vec = [
    StateBasedDeathProcess_Vec_Encode(column="AgeCat", column_states=age_compartments_by_10, target_states=["0 to 9"], rate=1.5 / 1000, stochastic=False, infstate_compartments=infstate_compartments),
    StateBasedDeathProcess_Vec_Encode(
        column="AgeCat", column_states=age_compartments_by_10, target_states=["10 to 19"], rate=0.1 / 1000, stochastic=False, infstate_compartments=infstate_compartments
    ),
    StateBasedDeathProcess_Vec_Encode(column="AgeCat", column_states=age_compartments_by_10, target_states=["20 to 29"], rate=1 / 1000, stochastic=False, infstate_compartments=infstate_compartments),
    StateBasedDeathProcess_Vec_Encode(
        column="AgeCat", column_states=age_compartments_by_10, target_states=["30 to 39"], rate=1.9 / 1000, stochastic=False, infstate_compartments=infstate_compartments
    ),
    StateBasedDeathProcess_Vec_Encode(
        column="AgeCat", column_states=age_compartments_by_10, target_states=["40 to 49"], rate=2.7 / 1000, stochastic=False, infstate_compartments=infstate_compartments
    ),
    StateBasedDeathProcess_Vec_Encode(
        column="AgeCat", column_states=age_compartments_by_10, target_states=["50 to 59"], rate=4.9 / 1000, stochastic=False, infstate_compartments=infstate_compartments
    ),
    StateBasedDeathProcess_Vec_Encode(
        column="AgeCat", column_states=age_compartments_by_10, target_states=["60 to 69"], rate=10.7 / 1000, stochastic=False, infstate_compartments=infstate_compartments
    ),
    StateBasedDeathProcess_Vec_Encode(column="AgeCat", column_states=age_compartments_by_10, target_states=["70+"], rate=43.9 / 1000, stochastic=False, infstate_compartments=infstate_compartments),
]

age_mix_vec = WAIFWTransmission_Vec_Encode_Bincount(
    waifw_matrix=waifw,
    inf_col="InfState",
    group_col="AgeCat",
    s_st="S",
    i_st="I",
    inf_to="I",
    stochastic=False,
    infstate_compartments=infstate_compartments,
    group_col_all_categories=age_compartments_by_10,
)
recover_rule_vec = SimpleTransition_Vec_Encode(column="InfState", from_st="I", to_st="R", rate=26, infstate_compartments=infstate_compartments, column_categories=trans_infect_compartments)


# Infect fixture
@pytest.fixture
def infect():
    return pd.DataFrame({"InfState": ["I"], "AgeCat": ["10 to 19"], "N": [1], "T": [2023]})


# ---Deterministic model---#
@pytest.fixture
def aging_determ_pandas_model():
    """EpiModel Pandas version"""
    determ_epi_mdl_pd = EpiModel_pd(init_state=add_infect, rules=[nc_aging_rules, [nc_birth], nc_deaths, [age_mix_pd, recover_rule_pd]])
    return determ_epi_mdl_pd


@pytest.fixture
def aging_determ_vec1_5_model():
    """EpiModel Vec version 1_5"""
    determ_epi_mdl_vec1_5 = EpiModel_Vec_Encode_1_5(init_state=add_infect, rules=[nc_aging_rules_vec, [nc_birth_vec], nc_deaths_vec, [age_mix_vec, recover_rule_vec]])
    return determ_epi_mdl_vec1_5


# define step function for each structure
def run_pandas_step(model, infect, t):
    infect.T = 2023 + t
    if t != 0:
        model.cur_state = pd.concat([model.cur_state, infect])
    model.do_timestep(dt=1 / 52)


def run_vec_step(model, infect, t):
    infect.T = 2023 + t
    if t != 0:
        model.add_new_data_to_current_state(new_data=infect)
    model.do_timestep(dt=1 / 52)


# -------------------
# Parameterized test
# -------------------
@pytest.mark.parametrize("model_label, model_fixture_name, step_fn", [("pandas", "aging_determ_pandas_model", run_pandas_step), ("vec1_5", "aging_determ_vec1_5_model", run_vec_step)])
def test_model_performance_and_output(request, model_label, model_fixture_name, step_fn, infect, benchmark_results):
    global pandas_result

    # print('\ndata:\n', _, '\nn_value:', n)
    print("\n=== Running test for model:", model_label, "===")

    # Load the model using the name of the fixture
    model = request.getfixturevalue(model_fixture_name)
    # print('model is\n', model.__class__.__name__)
    # print('model population', model.init_state)

    # Time and memory tracking
    tracemalloc.start()
    t0 = time.perf_counter()

    for t in np.arange(start, end, 1 / 52):
        step_fn(model, infect, t)

    t1 = time.perf_counter()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    runtime = round(t1 - t0, 3)
    peak_mb = round(peak / 1024**2, 2)

    print(f"# Model: {model_fixture_name}")
    print(f"# Time (s): {runtime}")
    print(f"# Peak Memory (MB): {peak_mb}")

    # Save result for charting
    benchmark_results.append(
        {
            "model": model_label,
            "model_fixture_name": model_fixture_name,
            "time_sec": runtime,
            "peak_memory_mb": peak_mb,
            "n": len(add_infect),  # n,
            "iters": iters,
        }
    )

    # Output comparison
    if model_label == "pandas":
        # print("\n=== Pandas Model full_epi ===")
        # print('pandas epi\n', model.full_epi)
        pandas_sorted = model.full_epi.sort_values(
            by=["T", "InfState", "AgeCat", "N"], ascending=[True, True, True, True]
        )  # sort values before comparing pandas and array result to make sure the order of N lines up
        # pandas_result = model.full_epi['N'].round(3).values
        pandas_result = pandas_sorted["N"].round(3).values
    elif model_label == "vec1_5":
        # arr = model._covnert_list_of_arrays_to_df(model._full_epi_list)['N'].round(3).values
        vec_sorted = model.full_epi().sort_values(
            by=["T", "InfState", "AgeCat", "N"], ascending=[True, True, True, True]
        )  # sort values before comparing pandas and array result to make sure the order of N lines up
        # print(f"\n=== {model_label} Model full_epi ===")
        # print('arr epi\n', model._covnert_list_of_arrays_to_df(model._full_epi_list))
        # assert np.allclose(arr, pandas_result, rtol=1e-3), f"{model_label} does not match pandas model"
        vec_result = vec_sorted["N"].round(3).values
        assert np.array_equal(vec_result, pandas_result), "Values do not match after rounding"


# -------------------
# Benchmark collector fixture
# -------------------
@pytest.fixture(scope="session", autouse=True)
def benchmark_results():
    """Collect results across tests into a list."""
    results: list[dict] = []
    yield results
    # After all tests, plot summary

    if results:
        df = pd.DataFrame(results)
        df["label"] = df.apply(lambda row: f"{int(row['n']):,} rows \n{row['iters']} iters", axis=1)
        # print('\ndf_plot:\n', df)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ---- Time plot ----
    sns.barplot(data=df, x="label", y="time_sec", hue="model", ax=axes[0])
    axes[0].set_title("Runtime (seconds) by Model")
    axes[0].set_xlabel("Data Size & Iterations")
    axes[0].set_ylabel("Time (seconds)")
    axes[0].tick_params(axis="x", rotation=30)

    # ---- Memory plot ----
    sns.barplot(data=df, x="label", y="peak_memory_mb", hue="model", ax=axes[1])
    axes[1].set_title("Peak Memory (MB) by Model")
    axes[1].set_xlabel("Data Size & Iterations")
    axes[1].set_ylabel("Memory (MB)")
    axes[1].tick_params(axis="x", rotation=30)

    # Add legends
    axes[0].legend(title="Backend")
    axes[1].legend(title="Backend")

    plt.tight_layout()
    plt.show()


# pytest test_Aging_Population_Pandas_Vec_performance.py
# pytest -k vec1_5 -vv #Run only the vec test:
