from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
from benchmark.StateBasedDeathDispatcher import StateBasedDeathProcessDispatcher

import numpy as np
import pandas as pd
import time
import tracemalloc
import gc

class StateBasedDeathProcessRunner(BaseModel):
    """
    Benchmarks time and memory usage of the StateBasedDeathProcess Pandas and NumPy versions.
    @param data_sizes: the number of records/rows in a input data.
    @param structure: data structure used for rules. E.g. Pandas, Numpy, Numpy_Encode
    @param iterations: the number of iterations to run the rule.
    @param column: single column used for StateBasedDeathProcess_Vec_Encode.
    @param columns: list of string used for StateBasedDeathProcess.
    @param rate: the rate at whihc people will die from.
    @param stochastic: whether the process is stochastic or deterministic.
    @param col_idx_map: mapping of input data columns and their column index. E.g. col_idx_map = {'InfState': 0, 'N': 1}.
    @param _infstate_comp_map: mapping between infectin states values and their categorical values. E.g. state_map = {'I': 0, 'R': 1, 'S': 2}.
    @param infstate_compartments: the infection compartments used in epidemics. E.g.infstate_compartments = ['S', 'I', 'R'].
    return: the time and memory usage of the rule with different data sizes, structures and iterations.
    """

    data_sizes: list[int]
    structures: list[str]
    iterations: list[int]
    column: str
    columns: list[str]
    all_states: list[str]
    target_states: list[str]
    rate: float
    stochastic: bool = False
    col_idx_map: dict[str, int] = Field(default_factory=dict)
    infstate_compartments: list[str] = Field(default_factory=list)

    time_mem_results: list[dict] = []

    _infstate_comp_map: dict[str, int] = PrivateAttr(default_factory=dict)

    def model_post_init(self, _):
        self._infstate_comp_map = {comp: i for i, comp in enumerate(sorted(self.infstate_compartments))}

    
    def run(self) -> list[dict]:
        """
        Creates input data with different sizes and runs deltas calculation with different data structures in different iterations.
        Tracks each combination's time and memory usage.
        """
        for size in self.data_sizes:
            start_age=0
            end_age=70
            age_step= (end_age-start_age)/(size-1)

            age_struct_pop = pd.DataFrame({
                'InfState' : pd.Categorical(["S"]*size, categories=["I","R","S"]),
                'AgeCat': ["{} to {}".format(i, i+ (age_step-1)) for i in np.arange(start_age, end_age, age_step)]+["{}+".format(end_age)],
                'N' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
                'T': 0
            })

            
            #print('age structure\n', age_struct_pop)
            for struct in self.structures:
                for iters in self.iterations:
                    print(f"\nRunning {struct} | size={size} | iterations={iters}")
                    if struct == 'Pandas': #provide dataframe to Pandas
                        data = age_struct_pop
                    elif struct == 'Numpy_Encode': #provide true Numpy array to Numpy_Encode
                        agecat_map = {label: idx for idx, label in enumerate((age_struct_pop['AgeCat'].unique()))}

                        age_struct_pop['InfState'] = age_struct_pop['InfState'].map(self._infstate_comp_map)
                        age_struct_pop['AgeCat'] = age_struct_pop['AgeCat'].map(agecat_map)
                        age_array = age_struct_pop.to_numpy(dtype=np.float64)
                        #print('age array\n', age_array) #debug
                        n_rows = age_array.shape[0] #detect the number of rows and columns in input array
                        n_cols = age_array.shape[1]
                        result_preallocation = np.empty((n_rows * 2, n_cols), dtype=np.float64) #preallocate a result array
                    dispatcher = StateBasedDeathProcessDispatcher(
                        structure=struct,
                        column=self.column,
                        columns=self.columns,
                        all_states=self.all_states,
                        target_states=self.target_states,
                        rate=self.rate,
                        stochastic=self.stochastic,
                        infstate_compartments = self.infstate_compartments
                    )
                    #print('dispatcher created ok\n') #debug
                    if struct  == 'Pandas':
                        gc.collect()
                        tracemalloc.start() #track memory
                        t0 = time.perf_counter() #track time
                        for _ in range(iters):
                            deltas = dispatcher.get_deltas(current_state=data)
                        t1 = time.perf_counter()
                        peak = tracemalloc.get_traced_memory()[1]
                        tracemalloc.stop()
                    elif struct == 'Numpy_Encode':
                        gc.collect()
                        tracemalloc.start() #track memory
                        t0 = time.perf_counter() #track time
                        for _ in range(iters):
                            deltas = dispatcher.get_deltas(current_state=age_array, col_idx_map=self.col_idx_map, result_buffer=result_preallocation)
                        t1 = time.perf_counter()
                        peak = tracemalloc.get_traced_memory()[1]
                        tracemalloc.stop()
                    
                    print(f"Sample deltas for {struct}:\n{deltas}\n, data length: {len(deltas)}\n, non-zero counts: {np.count_nonzero(deltas[:, self.col_idx_map['N']] if isinstance(deltas, np.ndarray) else deltas.loc[:, 'N'])}") #debug

                    #concatenate each iteration's result
                    self.time_mem_results.append({
                        'structure': struct,
                        'size': size,
                        'iterations': iters,
                        'time_sec': round(t1 - t0, 3),
                        'peak_memory_MB': round(peak / 1024**2, 2)
                    })

                    #print('time_mem_result\n', self.time_mem_results) #debug
        return self.time_mem_results