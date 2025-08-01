from pydantic import BaseModel
from benchmark.SimpleTransitionDispatcher import SimpleTransitionDispatcher
from typing import Dict

import numpy as np
import pandas as pd
import time
import tracemalloc

class SimpleTransitionRunner(BaseModel):
    """
    Benchmarks time and memory usage of the SimpleTransition Pandas and NumPy versions.
    @param data_sizes: the number of records/rows in a input data.
    @param structure: data structure used for rules. E.g. Pandas, Numpy, Numpy_Encode
    @param iterations: the number of iterations to run the rule.
    @param column: Name of the column this rule applies to.
    @param from_st: the state that column transitions from.
    @param to_st: the state that column transitions to.
    @param rate: transition rate per unit time.
    @param stochastic: whether the process is stochastic or deterministic.
    @param data_col: mapping of input data columns and their column index. E.g. data_col = {'InfState' : 0, 'N': 1}.
    @param state_map: mapping between infectin states values and their categorical values. E.g. state_map = {'S': 0, 'I': 1, 'R': 2}.
    @param infstate_compartments:the infection compartments used in epidemics. E.g.infstate_compartments = ['S', 'I', 'R'].
    return: the time and memory usage of the rule with different data sizes, structures and iterations.
    """
    data_sizes: list[int]
    structures: list[str]
    iterations: list[int]
    column: str
    from_st: str
    to_st: str
    rate: float
    stochastic: bool = False
    data_col: Dict[str, int] = None
    state_map: Dict[str, int] = None
    infstate_compartments: list[str] = None

    time_mem_results: list[dict] = []

    def run(self) -> list[Dict]:
        """
        Creates input data with different sizes and runs deltas calculation with different data structures in different iterations.
        Tracks each combination's time and memory usage.
        """
        for size in self.data_sizes:
            np.random.seed(3)
            infstate_values = np.random.choice(self.infstate_compartments, size=size) #column InfState values setup
            n_values = np.random.randint(1, 10, size=size) #column N values setup
            for struct in self.structures:
                for iters in self.iterations:
                    print(f"\nRunning {struct} | size={size} | iterations={iters}")

                    if struct == 'Pandas': #provide dataframe to Pandas
                        df = pd.DataFrame({
                            self.column: infstate_values,
                            'N': n_values
                        })
                        data = df
                    elif struct == 'Numpy': #provide dataframe to Numpy to keep top layer data as dataframe
                        data = pd.DataFrame({
                            self.column: infstate_values,
                            'N': n_values
                        })
                    elif struct == 'Numpy_Encode': #provide true Numpy array to Numpy_Encode
                        arr = np.column_stack((infstate_values, n_values))
                        arr_numba = arr.copy()
                        infstate_idx = self.data_col[self.column]
                        arr_numba[:, infstate_idx] = [self.state_map[val] for val in arr[:, infstate_idx]]
                        arr_numba = arr_numba.astype(np.float64)
                        n_rows = arr_numba.shape[0] #detect the number of rows and columns in input array
                        n_cols = arr_numba.shape[1]
                        result_preallocation = np.empty((n_rows * 2, n_cols), dtype=np.float64) #preallocate a result array
                   
                    dispatcher = SimpleTransitionDispatcher(
                        structure=struct,
                        column=self.column,
                        from_st=self.from_st,
                        to_st=self.to_st,
                        rate=self.rate,
                        stochastic=self.stochastic,
                        infstate_compartments = self.infstate_compartments
                    )
                    
                    if struct  == 'Pandas' or struct == 'Numpy':
                        tracemalloc.start() #track memory
                        t0 = time.perf_counter() #track time
                        for _ in range(iters):
                            deltas = dispatcher.get_deltas(current_state=data)
                        t1 = time.perf_counter()
                        peak = tracemalloc.get_traced_memory()[1]
                        tracemalloc.stop()
                    elif struct == 'Numpy_Encode':
                        tracemalloc.start() #track memory
                        t0 = time.perf_counter() #track time
                        for _ in range(iters):
                            deltas = dispatcher.get_deltas(current_state=arr_numba, data_col= self.data_col, result_buffer=result_preallocation)
                        t1 = time.perf_counter()
                        peak = tracemalloc.get_traced_memory()[1]
                        tracemalloc.stop()

                    #print(f"Sample deltas for {struct}:\n{deltas}\n") #debug

                    #concatenate each iteration's result
                    self.time_mem_results.append({
                        'structure': struct,
                        'size': size,
                        'iterations': iters,
                        'time_sec': round(t1 - t0, 3),
                        'peak_memory_MB': round(peak / 1024**2, 2)
                    })

                    print('time_mem_result\n', self.time_mem_results) #debug
        return self.time_mem_results