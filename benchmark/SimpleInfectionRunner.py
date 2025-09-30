from pydantic import BaseModel, Field, PrivateAttr
from benchmark.SimpleInfectionDispatcher import SimpleInfectionDispatcher
from typing import Annotated

import numpy as np
import pandas as pd
import time
import tracemalloc
import gc

class SimpleInfectionRunner(BaseModel):
    """
    Benchmarks time and memory usage of the SimpleInfection Pandas and NumPy versions.
    @param data_sizes: the number of records/rows in a input data.
    @param structure: data structure used for rules. E.g. Pandas, Numpy, Numpy_Encode
    @param iterations: the number of iterations to run the rule.
    @param beta: the transmission parameter. 
    @param column: Name of the column this rule applies to.
    @param s_st: the state for susceptibles, assumed to be S.
    @param i_st: the state for infectious, assumed to be I.
    @param inf_to: the state infectious folks go to, assumed to be I.
    @param stochastic: whether the process is stochastic or deterministic.
    @param col_idx_map: mapping of input data columns and their column index. E.g. col_idx_map = {'InfState': 0, 'N': 1}.
    @param _infstate_comp_map: mapping between infectin states values and their categorical values. E.g. state_map = {'I': 0, 'R': 1, 'S': 2}.
    @param infstate_compartments: the infection compartments used in epidemics. E.g.infstate_compartments = ['S', 'I', 'R'].
    return: the time and memory usage of the rule with different data sizes, structures and iterations.
    """
    data_sizes: list[int]
    structures: list[str]
    iterations: list[int]
    beta: Annotated[int | float, Field(ge=0)]
    column: str
    s_st: str
    i_st: str
    inf_to: str
    freq_dep: bool = True
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
            np.random.seed(3)
            #infstate_values = ['S', 'I', 'R', 'S', 'I', 'R'] #column InfState values setup
            #n_values = [10, 5, 0, 20, 0, 0] #column N values setup
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
                        infstate_idx = self.col_idx_map[self.column]
                        #print('infstate_idx:', infstate_idx)
                        arr_numba[:, infstate_idx] = [self._infstate_comp_map[val] for val in arr[:, infstate_idx]]
                        arr_numba = arr_numba.astype(np.float64)
                        #print('arr_numba\n', arr_numba)
                        n_rows = arr_numba.shape[0] #detect the number of rows and columns in input array
                        n_cols = arr_numba.shape[1]
                        result_preallocation = np.empty((n_rows * 2, n_cols), dtype=np.float64) #preallocate a result array
                    elif struct == 'Josh_Encode_Vec': #speical value creation for Josh's class
                        arr = np.column_stack((infstate_values, n_values))
                        arr_numba = arr.copy()
                        infstate_idx = self.col_idx_map[self.column]
                        comp_map = {label: i for i, label in enumerate(sorted(self.infstate_compartments))}
                        #print('comp_map:', comp_map)
                        arr_numba[:, infstate_idx] = [comp_map[val] for val in arr[:, infstate_idx]]
                        arr_numba = arr_numba.astype(np.float64)
                        #print('arr_numba for J\n', arr_numba)
                    dispatcher = SimpleInfectionDispatcher(
                        structure=struct,
                        beta=self.beta,
                        column=self.column,
                        s_st=self.s_st,
                        i_st=self.i_st,
                        inf_to=self.inf_to,
                        freq_dep=self.freq_dep,
                        stochastic=self.stochastic,
                        infstate_compartments = self.infstate_compartments
                    )
                    
                    if struct  == 'Pandas':
                        gc.collect()
                        tracemalloc.start() #track memory
                        t0 = time.perf_counter() #track time
                        for _ in range(iters):
                            deltas = dispatcher.get_deltas(current_state=data, stochastic=self.stochastic)
                        t1 = time.perf_counter()
                        peak = tracemalloc.get_traced_memory()[1]
                        tracemalloc.stop()
                    elif struct == 'Numpy':
                        pass
                    elif struct == 'Numpy_Encode':
                        gc.collect()
                        tracemalloc.start() #track memory
                        t0 = time.perf_counter() #track time
                        for _ in range(iters):
                            deltas = dispatcher.get_deltas(current_state=arr_numba, col_idx_map=self.col_idx_map, result_buffer=result_preallocation, stochastic=self.stochastic)
                        t1 = time.perf_counter()
                        peak = tracemalloc.get_traced_memory()[1]
                        tracemalloc.stop()
                    elif struct == 'Josh_Encode_Vec':
                        gc.collect()
                        #converting compartments to code first
                        #comp_map = {label: i for i, label in enumerate (sorted(self.infstate_compartments))}
                        #print('comp_map:', comp_map)
                        dispatcher.compile(comp_map)
                        tracemalloc.start() #track memory
                        t0 = time.perf_counter() #track time
                        for _ in range(iters):
                            deltas = dispatcher.apply(state=arr_numba, col_idx=self.col_idx_map, dt=1.0)
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