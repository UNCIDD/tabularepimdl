from benchmark.SharedTraitInfectionRunner import SharedTraitInfectionRunner
from benchmark.Visualizer import Visualizer
import numpy as np
import pandas as pd
import random

compartments = ['S', 'I', 'R']
size = 1000 #data size

###===fixed data===###
"""
random_cats = [7, 8, 9, 10, 11, 12]
c0 = 7
c1 = 8
c2 = 9
c_rest = [10, 11, 12]
            
n_values = [10, 20, 30, 22, 12, 40, 50, 7]
t_values = [0]*8

current_state = pd.DataFrame({
    'InfState':['S', 'S', 'S', 'I', 'I', 'I', 'I', 'R'],
    'HH_Number': [c0, c1, c2, c1, c0, 10, 11, 12], 
    'N': n_values,
    'T': t_values
})
"""

###===random data===###

n_samples = size
random_indices = np.random.choice(size, size=n_samples, replace=False)
random_cats = [f'catg_{i}' for i in random_indices]
#print('random cats:', random_cats) #debug

inf_state = [random.choice(compartments) for _ in range(size)] #inf states
n_values = [random.randint(10, 1_000_000) for _ in range(size)]
t_values = [2023]*size

current_state = pd.DataFrame({
                'InfState' : inf_state,
                'HH_Number': random_cats,
                'N' : n_values,
                'T' : t_values
                })

current_state = current_state.groupby(['InfState', 'HH_Number'], observed=True).agg({'N': 'sum', 'T': 'max'}).reset_index()

def SharedTrait_Run_n_Visual():
    runner = SharedTraitInfectionRunner(
        data_sizes= [size],
        data_input = current_state,
        structures= ["Pandas", "Numpy_Vec_Encode"],
        iterations= [500],#[100, 300, 500, 700],
        
        inf_col = "InfState",
        in_beta = 0.2/5,
        out_beta = 0.002/5,
        trait_col = "HH_Number",
        trait_col_all_categories=random_cats,
        
        s_st="S",
        i_st="I",
        inf_to="I",
        stochastic=False,
        col_idx_map = {'InfState': 0, 'HH_Number': 1, 'N': 2, 'T': 3},
        infstate_compartments = compartments
    )
    results = runner.run()

    viz = Visualizer(runner_results = results)
    viz.plot()

if __name__ == "__main__":
    SharedTrait_Run_n_Visual()
#python -m benchmark.SharedTraitInfection_Run_n_Visual