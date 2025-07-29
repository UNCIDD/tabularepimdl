import numpy as np
import pandas as pd
import pytest

from tabularepimdl.EpiModel_Vector import EpiModel as EpiModel
from tabularepimdl.SimpleInfection_Vector import SimpleInfection as SimpleInfection
from tabularepimdl.SimpleTransition_Vector import SimpleTransition as SimpleTransition

from tabularepimdl.EpiModel import EpiModel as EpiModel_Base
from tabularepimdl.SimpleInfection import SimpleInfection as SimpleInfection_Base
from tabularepimdl.SimpleTransition import SimpleTransition as SimpleTransition_Base


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
    t0, tf, dt = 0.0, 100.0, 0.25

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


@pytest.fixture
def sir_base_model():
    df_init = pd.DataFrame({
        "group": [0, 0, 0],
        "compartment": ["S", "I", "R"],
        "N": [990, 10, 0],
        "T": [0.0, 0.0, 0.0]
    })

    beta = 3.0
    gamma = 0.1

    rules = [[
        SimpleInfection_Base(beta=beta, column="compartment", s_st="S", i_st="I", inf_to="I"),
        SimpleTransition_Base(rate=gamma, column="compartment", from_st="I", to_st="R")
    ]]

    model = EpiModel_Base(
        init_state=df_init,
        rules=rules,
    )
    return model


def test_vector_base_equivalence(sir_vector_model, sir_base_model):
    # Run both models
    sir_vector_model.run()

    # Match number of steps and dt
    dt = 0.25
    tf = 100.0
    n_steps = int(tf / dt)
    for _ in range(n_steps):
        sir_base_model.do_timestep(dt=dt)

    # Extract and sort outputs
    vec_epi = sir_vector_model.get_full_epi()
    base_epi = sir_base_model.full_epi

    vec_epi_sorted = vec_epi.sort_values(["T", "group", "compartment"]).reset_index(drop=True)
    base_epi_sorted = base_epi.sort_values(["T", "group", "compartment"]).reset_index(drop=True)

    # Restrict to shared time steps
    common_times = np.intersect1d(vec_epi_sorted["T"].unique(), base_epi_sorted["T"].unique())
    vec_epi_sorted = vec_epi_sorted[vec_epi_sorted["T"].isin(common_times)]
    base_epi_sorted = base_epi_sorted[base_epi_sorted["T"].isin(common_times)]

    # Pivot for comparison
    vec_pivot = vec_epi_sorted.pivot_table(index=["T", "group"], columns="compartment", values="N")
    base_pivot = base_epi_sorted.pivot_table(index=["T", "group"], columns="compartment", values="N")

    diff = np.abs(vec_pivot - base_pivot)
    max_diff = diff.max().max()

    print("[DEBUG] max diff:", max_diff)
    assert max_diff < 1e-3, f"Discrepancy too large: {max_diff}"
