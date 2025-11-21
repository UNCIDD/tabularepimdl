import pytest
import pandas as pd
import numpy as np

"""
confest.py provides fixtures for the entire directory.
"""
#Define the custom CLI option
def pytest_addoption(parser):
    #separate CLI options into a named group
    TabularEpi_group = parser.getgroup("epimodel", "TabularEpi Model Test Options")

    TabularEpi_group.addoption("--switch", action="store", default="fixed", help="Set data type: fixed or random")

    TabularEpi_group.addoption("--iters", action="store", type=int, default=200,
                    help="Number of iterations to run each model"
    ) #could add iters as a runtime option, not currently being used.

@pytest.fixture(scope="session")
def switch(request):
    """Get the command-line option for switch."""
    return request.config.getoption("--switch")

@pytest.fixture(scope="session") #could add iters as a runtime option, not currently being used.
def iters(request):
    """Get the command-line option for number of iterations."""
    return request.config.getoption("--iters")

@pytest.fixture(scope="session")
def init_df(switch):
    """Create the initial population DataFrame and return (df, n)."""
    np.random.seed(3)
    
    if switch == 'fixed':
        n = 2 #two rows of fixed data
        df = pd.DataFrame({
            'InfState': ['S', 'I'],
            'N': [1_500_000, 10],
            'T': [0, 0]
        })
    else:
        n = 1_000_000
        infstate_values = np.random.choice(['S', 'I', 'R'], size=n)
        n_values = np.random.randint(1, 10, size=n)
        t_values = np.full(n, 0.0)
        df = pd.DataFrame({
            'InfState': infstate_values,
            'N': n_values,
            'T': t_values
        })
    
    return df, n