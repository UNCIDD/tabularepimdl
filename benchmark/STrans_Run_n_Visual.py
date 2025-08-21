from benchmark.SimpleTransitionRunner import SimpleTransitionRunner
from benchmark.Visualizer import Visualizer

def STrans_Run_n_Visual():
    runner = SimpleTransitionRunner(
        data_sizes= [15],#[10**6, 10**7], #15, 15000, 15000000 works for Josh's code
        structures= ['Pandas', 'Numpy', 'Numpy_Encode', 'Josh_Encode_Vec'],#['Josh_Encode_Vec'],
        iterations= [1],#[100, 300, 500, 700],
        column='InfState',
        from_st='S',
        to_st='I',
        rate=0.1,
        stochastic=False,
        col_idx_map = {'InfState' : 0, 'N': 1},
        state_map = {'I': 0, 'R': 1, 'S': 2},
        infstate_compartments = ['S', 'I', 'R']
    )
    results = runner.run()

    viz = Visualizer(runner_results = results)
    viz.plot()

if __name__ == "__main__":
    STrans_Run_n_Visual()
#python -m benchmark.STrans_Run_n_Visual