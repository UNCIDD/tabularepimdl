from benchmark.SimpleObservationProcessRunner import SimpleObservationProcessRunner
from benchmark.Visualizer import Visualizer
import numpy as np
import pandas as pd
import random

compartments = ['S', 'I', 'R']
observations = ['U', 'P', 'I']
size = 100 #data size

###===fixed data===###
"""
n_values = [10, 30, 50, 20, 5]
t_values = [2023]*5

current_state = pd.DataFrame({
    'InfState': ['S', 'S', 'S', 'I', 'R'], #links to source_col
    'Hosp': ['U', 'P', 'I', 'U', 'P'], #links to obs_col 
    'N': n_values,
    'T': t_values
})

"""
###===random data===###

#n_samples = size
#random_indices = np.random.randint(0, size, size=n_samples)
#random_cats = [f'catg_{i}' for i in random_indices]
#print('random cats:', random_cats) #debug

inf_state = [random.choice(compartments) for _ in range(size)] #inf states
hosp_state = [random.choice(observations) for _ in range(size)] #inf states
n_values = [random.randint(10, 1_000_000) for _ in range(size)]
t_values = [2023]*size

current_state = pd.DataFrame({
                'InfState' : inf_state,
                'Hosp': hosp_state,
                'N' : n_values,
                'T' : t_values
                })

#current_state = current_state.groupby(['InfState', 'Hosp'], observed=True).agg({'N': 'sum', 'T': 'max'}).reset_index()
#print('current_state\n', current_state)

def SimpleObservation_Run_n_Visual():
    runner = SimpleObservationProcessRunner(
        data_sizes= [size],
        data_input = current_state,
        structures= ["Pandas", "Numpy_Vec_Encode", "Numpy_Vec_Encode_nobuffer"],
        iterations= [500],#[100, 300, 500, 700],
        
        source_col = "InfState",
        source_state = "I",
        obs_col = 'Hosp',
        rate = 0.05,
        unobs_state = 'U',
        incobs_state = 'I',
        prevobs_state = 'P',
        
        stochastic=False,
        col_idx_map = {'InfState': 0, 'Hosp': 1, 'N': 2, 'T': 3},
        infstate_compartments = compartments,
        obs_col_all_categories = observations
    )
    results = runner.run()

    viz = Visualizer(runner_results = results)
    viz.plot()

if __name__ == "__main__":
    SimpleObservation_Run_n_Visual()
#python -m benchmark.SimpleObservationProcess_Run_n_Visual