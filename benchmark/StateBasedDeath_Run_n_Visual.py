from benchmark.StateBasedDeathRunner import StateBasedDeathProcessRunner
from benchmark.Visualizer import Visualizer
import numpy as np

def All_States_List():
    start_age=0
    end_age=70
    size=15
    age_step= (end_age-start_age)/(size-1)
    all_states = ["{} to {}".format(i, i+ (age_step-1)) for i in np.arange(start_age, end_age, age_step)]+["{}+".format(end_age)]
    return all_states

def StateBasedDeath_Run_n_Visual():
    runner = StateBasedDeathProcessRunner(
        data_sizes= [15],
        structures= ['Pandas', 'Numpy_Encode'],
        iterations= [700],#[100, 300, 500, 700],
        column='AgeCat',
        columns=['AgeCat', 'AgeCat'],
        all_states=All_States_List(),
        target_states=['0.0 to 4.0', '70+'],
        rate=0.2,
        stochastic=False,
        col_idx_map = {'InfState': 0, 'AgeCat': 1, 'N': 2, 'T': 3},
        infstate_compartments = ['S', 'I', 'R']
    )
    results = runner.run()

    viz = Visualizer(runner_results = results)
    viz.plot()

if __name__ == "__main__":
    StateBasedDeath_Run_n_Visual()
#python -m benchmark.StateBasedDeath_Run_n_Visual