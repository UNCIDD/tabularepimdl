from benchmark.SimpleTransitionRunner import SimpleTransitionRunner
from benchmark.Visualizer import Visualizer

def Run_n_Visual():
    runner = SimpleTransitionRunner(
        data_sizes= [10**6, 10**7],
        structures=['Pandas', 'Numpy', 'Numpy_Encode'],
        iterations= [1],#[100, 300, 500, 700],
        column='InfState',
        from_st='S',
        to_st='I',
        rate=0.05,
        stochastic=False,
        data_col = {'InfState' : 0, 'N': 1},
        state_map = {'S': 0, 'I': 1, 'R': 2},
        infstate_compartments = ['S', 'I', 'R']
    )
    results = runner.run()

    viz = Visualizer(runner_results = results)
    viz.plot()

if __name__ == "__main__":
    Run_n_Visual()
#python -m benchmark.Run_n_Visual