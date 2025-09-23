from benchmark.BirthProcessRunner import BirthProcessRunner
from benchmark.Visualizer import Visualizer

def BirthProcess_Run_n_Visual():
    runner = BirthProcessRunner(
        data_sizes= [15],
        structures= ['Pandas', 'Numpy_Encode'],
        iterations= [700],#[100, 300, 500, 700],
        rate=0.01,
        stochastic=False,
        col_idx_map = {'InfState': 0, 'AgeCat': 1, 'N': 2, 'T': 3},
        infstate_compartments = ['S', 'I', 'R']
    )
    results = runner.run()

    viz = Visualizer(runner_results = results)
    viz.plot()

if __name__ == "__main__":
    BirthProcess_Run_n_Visual()
#python -m benchmark.BirthProcess_Run_n_Visual