from benchmark.WAIFWTransmissionRunner import WAIFWTransmissionRunner
from benchmark.Visualizer import Visualizer
import numpy as np

waiwf = np.array([[1,1,0.5,0.5,0.5,0.5,0.25,0.25],
                  [1,1,0.5,0.5,0.5,0.5,0.25,0.25],
                  [0.5,0.5,0.25,0.25,0.25,0.25,0.5,0.5],
                  [0.5,0.5,0.25,0.25,0.25,0.25,0.5,0.5],
                  [0.5,0.5,0.25,0.25,0.25,0.25,0.5,0.5],
                  [0.5,0.5,0.25,0.25,0.25,0.25,0.5,0.5],
                  [0.25,0.25,0.5,0.5,0.5,0.5,1,1],
                  [0.25,0.25,0.5,0.5,0.5,0.5,1,1]]) 


def WTrans_Run_n_Visual():
    runner = WAIFWTransmissionRunner(
        data_sizes= [1],
        structures= ["Pandas", "Pandas_Numba", "Numpy_Vec_Encode_Numba", "Numpy_Vec_Encode_Bincount"],
        iterations= [1],#[100, 300, 500, 700],
        waifw_matrix=waiwf,
        inf_col = "InfState",
        group_col = "AgeCat",
        group_col_all_categories=['0 to 9', '10 to 19', '20 to 29', '30 to 39', '40 to 49', '50 to 59', '60 to 69', '70+'],
        s_st="S",
        i_st="I",
        inf_to="I",
        stochastic=False,
        col_idx_map = {'InfState': 0, 'AgeCat': 1, 'N': 2, 'T': 3},
        infstate_compartments = ['S', 'I', 'R']
    )
    results = runner.run()

    viz = Visualizer(runner_results = results)
    viz.plot()

if __name__ == "__main__":
    WTrans_Run_n_Visual()
#python -m benchmark.WAIFWTransmission_Run_n_Visual