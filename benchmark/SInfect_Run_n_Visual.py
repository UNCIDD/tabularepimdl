from benchmark.SimpleInfectionRunner import SimpleInfectionRunner
from benchmark.Visualizer import Visualizer

def SInfect_Run_n_Visual():
    runner = SimpleInfectionRunner(
        data_sizes= [15],#[10**6, 10**7], #15, 15000, 15000000 works for Josh's code
        structures= ['Pandas', 'Numpy_Encode', 'Josh_Encode_Vec'],#['Josh_Encode_Vec'],
        iterations= [1],#[100, 300, 500, 700],
        beta=0.1,
        column='InfState',
        s_st='S',
        i_st='I',
        inf_to='I',
        freq_dep=True,
        stochastic=False,
        col_idx_map = {'InfState' : 0, 'N': 1},
        state_map = {'S': 0, 'I': 1, 'R': 2},
        infstate_compartments = ['S', 'I', 'R']
    )
    results = runner.run()

    viz = Visualizer(runner_results = results)
    viz.plot()

if __name__ == "__main__":
    SInfect_Run_n_Visual()
#python -m benchmark.SInfect_Run_n_Visual