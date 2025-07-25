from benchmark.SimpleTransitionRunner import SimpleTransitionRunner
from benchmark.Visualizer import Visualizer

def Run_n_Visual():
    runner = SimpleTransitionRunner(
        data_sizes= [10],#[10**4, 10**5, 10**6, 10**7],
        structures=['Pandas', 'Numpy'],
        iterations= [10],#[100, 300, 500, 700],
        column='InfState',
        from_st='S',
        to_st='I',
        rate=0.05,
        stochastic=False
    )
    results = runner.run()

    viz = Visualizer(runner_results = results)
    viz.plot()

if __name__ == "__main__":
    Run_n_Visual()
