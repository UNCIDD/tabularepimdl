from benchmark.WAIFWTransmissionRunner import WAIFWTransmissionRunner
from benchmark.Visualizer import Visualizer
import numpy as np
import pandas as pd
import random

compartments = ['S', 'I', 'R']
size = 700 #data size
"""
###===fixed data===###
waiwf = np.array([[1,1,0.5,0.5,0.5,0.5,0.25,0.25],
                  [1,1,0.5,0.5,0.5,0.5,0.25,0.25],
                  [0.5,0.5,0.25,0.25,0.25,0.25,0.5,0.5],
                  [0.5,0.5,0.25,0.25,0.25,0.25,0.5,0.5],
                  [0.5,0.5,0.25,0.25,0.25,0.25,0.5,0.5],
                  [0.5,0.5,0.25,0.25,0.25,0.25,0.5,0.5],
                  [0.25,0.25,0.5,0.5,0.5,0.5,1,1],
                  [0.25,0.25,0.5,0.5,0.5,0.5,1,1]]) 
unique_cats = ['0 to 9', '10 to 19', '20 to 29', '30 to 39', '40 to 49', '50 to 59', '60 to 69', '70+']
age_c0 = '0 to 9'
age_c1 = '10 to 19'
age_c_rest = ['20 to 29', '30 to 39', '40 to 49', '50 to 59', '60 to 69', '70+']

v0 = 1140000
v1 = 1320000
v_rest = [1320000,1290000,1280000, 1280000,1185000,1175000, 3, 4, 99, 1]
            
n_values = [v1, v0] + v_rest
t_values = [2023]*12
            
pop_catg = pd.DataFrame({
                'InfState' : ["S"]*8 + ['I', 'I', 'I', 'R'],
                'AgeCat': pd.Categorical([age_c1, age_c0] + age_c_rest + ["10 to 19", "50 to 59", "50 to 59", '70+'], categories=unique_cats),
                'N' : n_values,
                'T' : t_values
                })

"""
###===random data===###
unique_cats = [f'catg_{i}' for i in range(size)] #unique categories
inf_state = [random.choice(compartments) for _ in range(size)] #inf states
n_values = [random.randint(10, 1_000_000) for _ in range(size)]
t_values = [2023]*size
waiwf = np.random.rand(size, size) #creates a n×n array of floats between 0.0 and 1.0.

pop_catg = pd.DataFrame({
                'InfState' : inf_state,
                'AgeCat': pd.Categorical(unique_cats, categories=unique_cats),
                'N' : n_values,
                'T' : t_values
                })
#print('pop catg Age\n', pop_catg['AgeCat'].cat.categories) #debug
#print('cat codes\n', pop_catg['AgeCat'].cat.codes)

def WTrans_Run_n_Visual():
    runner = WAIFWTransmissionRunner(
        data_sizes= [size],
        data_input = pop_catg,
        structures= ["Pandas", "Pandas_Numba", "Numpy_Vec_Encode_Numba", "Numpy_Vec_Encode_Bincount"],
        iterations= [500],#[100, 300, 500, 700],
        waifw_matrix=waiwf,
        inf_col = "InfState",
        group_col = "AgeCat",
        group_col_all_categories=unique_cats,
        s_st="S",
        i_st="I",
        inf_to="I",
        stochastic=False,
        col_idx_map = {'InfState': 0, 'AgeCat': 1, 'N': 2, 'T': 3},
        infstate_compartments = compartments
    )
    results = runner.run()

    viz = Visualizer(runner_results = results)
    viz.plot()

if __name__ == "__main__":
    WTrans_Run_n_Visual()
#python -m benchmark.WAIFWTransmission_Run_n_Visual