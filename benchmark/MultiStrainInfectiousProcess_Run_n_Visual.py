from benchmark.MultiStrainInfectiousProcessRunner import MultiStrainInfectiousProcessRunner
from benchmark.Visualizer import Visualizer
import numpy as np
import pandas as pd
import random

compartments = ['S', 'I', 'R']
columns_all_categories = ['S', 'I', 'R']
size = 5 #data size

###===fixed data===###
"""
n_values = [100, 200, 150]
t_values = [2023]*3
betas_orig = np.array([0.1, 0.05]) #two strains
columns = ["strain1", "strain2"] #two strains
cross_protect = np.array([[1.0, 0.5], [0.5, 1.0]]) #two strains
col_idx_map = {'strain1': 0, 'strain2': 1, 'N': 2, 'T': 3} #two strains

current_state = pd.DataFrame({
    'strain1': ['S', 'I', 'R'],
    'strain2': ['S', 'S', 'I'],
    'N': [100, 200, 150],
    'T': [2023]*3
})
"""



###===random data===###
#n_samples = size
#random_indices = np.random.randint(0, size, size=n_samples)
#random_cats = [f'catg_{i}' for i in random_indices]
#print('random cats:', random_cats) #debug

strain1 = [random.choice(columns_all_categories) for _ in range(size)]
strain2 = [random.choice(columns_all_categories) for _ in range(size)]
strain3 = [random.choice(columns_all_categories) for _ in range(size)]
strain4 = [random.choice(columns_all_categories) for _ in range(size)]
n_values = [random.randint(10, 1_000_000) for _ in range(size)]
t_values = [2023]*size
betas_orig = np.array([0.1, 0.05, 0.2, 0.08]) #four strains
columns = ["strain1", "strain2", "strain3", "strain4"] #four strains
cross_protect = np.array([[1.0, 0.5, 0.2, 0.3], [0.5, 1.0, 0.3, 0.2], [0.3, 0.4, 0.5, 0.6], [0.2, 0.7, 0.9, 0.5]]) #four strains
col_idx_map = {'strain1': 0, 'strain2': 1, 'strain3': 2,  'strain4': 3, 'N': 4, 'T': 5} #four strains

current_state = pd.DataFrame({
                'N' : n_values,
                'strain1': strain1,
                'T' : t_values,
                'strain2': strain2,
                'strain3': strain3,
                'strain4': strain4,
                })

current_state = current_state.groupby(columns, observed=True).agg({'N': 'sum', 'T': 'max'}).reset_index()

#print('current_state\n', current_state)

def MultiStrainInfectious_Run_n_Visual():
    runner = MultiStrainInfectiousProcessRunner(
        data_sizes= [size],
        data_input = current_state,
        structures= ["Pandas", "Numpy_Vec_Encode_1", "Numpy_Vec_Encode_2"],
        iterations= [1],#[100, 300, 500, 700],
        
        betas = betas_orig,
        columns = columns,
        cross_protect = cross_protect,
        s_st = 'S',
        i_st = 'I',
        r_st = 'R',
        inf_to = 'I',
        stochastic=False,
        freq_dep = True,
        col_idx_map = col_idx_map,
        infstate_compartments = compartments,
        columns_all_categories = columns_all_categories
    )
    results = runner.run()

    viz = Visualizer(runner_results = results)
    viz.plot()

if __name__ == "__main__":
    MultiStrainInfectious_Run_n_Visual()
#python -m benchmark.MultiStrainInfectiousProcess_Run_n_Visual