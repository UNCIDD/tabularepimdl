import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from tabularepimdl.EpiModel_Vector import EpiModel
from tabularepimdl.SimpleInfection_Vector import SimpleInfection
from tabularepimdl.SimpleTransition_Vector import SimpleTransition


@pytest.fixture
def sir_vector_model():
    """Fixture for a simple SIR model with vectorized rules."""
    df_init = pd.DataFrame({
        "group": [0, 0],
        "compartment": ["S", "I"],
        "N": [990, 10],
        "T": [0.0, 0.0]
    })

    beta = 3.0
    gamma = 0.1
    t0, tf, dt = 0.0, 10.0, 0.25

    rules = [[
        SimpleInfection(beta=beta, column="compartment", s_st="S", i_st="I", inf_to="I"),
        SimpleTransition(rate=gamma, column="compartment", from_st="I", to_st="R")
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


def test_vector_sir_conservation(sir_vector_model):
    sir_vector_model.run()
    epi = sir_vector_model.get_full_epi()
    pop_by_time = epi.groupby("T", sort=True)["N"].sum()
    assert_allclose(pop_by_time.values, pop_by_time.iloc[0], atol=1e-4,
                    err_msg="Total population not conserved over time")


def test_vector_sir_nonnegative(sir_vector_model):
    sir_vector_model.run()
    epi = sir_vector_model.get_full_epi()

    assert np.all(epi["N"].values >= 0), "Negative population counts detected"


def test_vector_sir_compartment_labels(sir_vector_model):
    sir_vector_model.run()
    epi = sir_vector_model.get_full_epi()

    allowed = {"S", "I", "R"}
    observed = set(epi["compartment"].unique())


    assert observed.issubset(allowed), f"Unexpected compartments: {observed - allowed}"


def test_vector_sir_time_consistency(sir_vector_model):
    sir_vector_model.run()
    epi = sir_vector_model.get_full_epi()
    t_values = np.unique(epi["T"].values)
    dt_diffs = np.diff(t_values)
    expected_dt = sir_vector_model.dt

    assert_allclose(dt_diffs, expected_dt, rtol=1e-4,
                    err_msg="Time steps are not uniform across simulation")


def test_vector_sir_has_dynamics(sir_vector_model):
    sir_vector_model.run()
    epi = sir_vector_model.get_full_epi()

    initial = epi[epi["T"] == 0.0].sort_values("compartment")["N"].values
    final = epi[epi["T"] == sir_vector_model.tf].sort_values("compartment")["N"].values


    assert not np.allclose(initial, final), "No dynamics occurred over time"


def test_vector_sir_expected_trend(sir_vector_model):
    sir_vector_model.run()
    epi = sir_vector_model.get_full_epi()


    S_series = epi[epi["compartment"] == "S"].groupby("T")["N"].sum()
    R_series = epi[epi["compartment"] == "R"].groupby("T")["N"].sum()



    assert S_series.iloc[-1] < S_series.iloc[0], "Susceptible population did not decrease"
    assert R_series.iloc[-1] > R_series.iloc[0], "Recovered population did not increase"


def test_vector_sir_reset_behavior(sir_vector_model):
    sir_vector_model.run()
    post_df = sir_vector_model.get_cur_state()

    reset_df = sir_vector_model.reset()
    init_df = sir_vector_model.init_state.copy()

    # Encode both DataFrames for consistent comparison
    init_df["compartment"] = init_df["compartment"].astype(str).map(sir_vector_model.comp_map).astype(np.int32)
    reset_df["compartment"] = reset_df["compartment"].map(sir_vector_model.comp_map).astype(np.int32)



    merged = pd.merge(
        reset_df.sort_values("compartment"),
        init_df.sort_values("compartment"),
        on="compartment", suffixes=("_reset", "_init")
    )
    assert_allclose(merged["N_reset"], merged["N_init"], rtol=1e-6)


def test_vector_sir_num_steps(sir_vector_model):
    sir_vector_model.run()
    epi = sir_vector_model.get_full_epi()
    unique_times = epi["T"].unique()
    expected_steps = int((sir_vector_model.tf - sir_vector_model.t0) / sir_vector_model.dt) + 1

    assert len(unique_times) == expected_steps, "Incorrect number of time steps"
