from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
from benchmark.SimpleObservationProcessDispatcher import SimpleObservationProcessDispatcher

import numpy as np
import pandas as pd
import time
import tracemalloc
import gc

class SimpleObservationProcessRunner(BaseModel):
    """
    Benchmarks time and memory usage of the SharedTraitInfection Pandas and NumPy versions.
    @param data_sizes: the number of records/rows in a input data.
    @param data_input: the test data for rules.
    @param structure: data structure used for rules. E.g. Pandas, Numpy, Numpy_Encode
    @param iterations: the number of iterations to run the rule.
    @param source_col: the column containing source_state for the observation process.
    @param source_state: the state individuals start, listed in source_col.
    @param obs_col: the column that contains each group of individuals' observed state.
    @param rate: the number of people move from a particular state into another state per unit time.
    @param unobs_state: un-observed state, listed in obs_col.
    @param incobs_state: incident-observed state, listed in obs_col.
    @param prevobs_state: previously-observed state, listed in obs_col.
    @param stochastic: whether the process is stochastic or deterministic.
    @param col_idx_map: mapping of input data columns and their column index. E.g. col_idx_map = {'InfState': 0, 'N': 1}.
    @param _infstate_comp_map: mapping between infectin states values and their categorical values. E.g. state_map = {'I': 0, 'R': 1, 'S': 2}.
    @param infstate_compartments: the infection compartments used in epidemics. E.g.infstate_compartments = ['S', 'I', 'R'].
    @param observation_compartments: the observation compartments used in epidemics. e.g. ['U', 'P', 'I'], U=unobserved, P=previously-observed, I=incident-observed
    return: the time and memory usage of the rule with different data sizes, structures and iterations.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data_sizes: list[int]
    data_input: pd.DataFrame
    structures: list[str]
    iterations: list[int]
    source_col: str
    source_state: str
    obs_col: str
    rate: float
    unobs_state: str
    incobs_state: str
    prevobs_state: str
    stochastic: bool = False
    col_idx_map: dict[str, int] = Field(default_factory=dict)
    infstate_compartments: list[str] = Field(default_factory=list)
    observation_compartments: list[str] = Field(default_factory=list)

    time_mem_results: list[dict] = Field(default_factory=list)

    _infstate_comp_map: dict[str, int] = PrivateAttr(default_factory=dict)

    def model_post_init(self, _):
        self._infstate_comp_map = {comp: i for i, comp in enumerate(sorted(self.infstate_compartments))}


    def encode_column_sorted(self, compartment_list, df_col):
        print('comp list:', compartment_list)
        unique_vals = list(sorted(set(compartment_list)))  # Sorted to ensure consistent index, set needs to be inside sorted because the order in a set is not guranteed
        print('unique values:', unique_vals)
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        print('mapping\n', mapping) #debug
        return df_col.map(mapping)
    
    def run(self) -> list[dict]:
        """
        Creates input data with different sizes and runs deltas calculation with different data structures in different iterations.
        Tracks each combination's time and memory usage.
        """
        for size in self.data_sizes:
            #print('populaton\n', pop_catg)

            for struct in self.structures:
                for iters in self.iterations:
                    print(f"\nRunning {struct} | size={len(self.data_input)} | iterations={iters}") #replace size with pop_catg

                    if struct == 'Pandas': #provide dataframe to Pandas
                        data = self.data_input
                    elif struct == 'Numpy_Vec_Encode': #provide true Numpy array to Numpy_Encode
                        #print('input data\n', self.data_input)
                        InfState_encode = self.encode_column_sorted(self.infstate_compartments, self.data_input[self.source_col]) #infstate column is sorted
                        Obs_encode = self.encode_column_sorted(self.observation_compartments, self.data_input[self.obs_col]) #categorical column is sorted
                        #print('Trait sorted code\n', Trait_encode)
                        arr_numba = np.column_stack((InfState_encode, Obs_encode, self.data_input['N'], self.data_input['T']))
                        arr_numba = arr_numba.astype(np.float64) #this makes all columns a float number, it will later cause float indexing error for Numba, but WAIFW rule will convert group_col category back to integers.
                        print('arr_numba\n', arr_numba)
                        n_rows = arr_numba.shape[0] #detect the number of rows and columns in input array
                        n_cols = arr_numba.shape[1]
                        result_preallocation = np.empty((n_rows * 2, n_cols), dtype=np.float64) #preallocate a result array
                    
                    dispatcher = SimpleObservationProcessDispatcher(
                        structure=struct,
                        source_col=self.source_col,
                        source_state=self.source_state,
                        obs_col=self.obs_col,
                        rate=self.rate,
                        unobs_state=self.unobs_state,
                        incobs_state=self.incobs_state,
                        prevobs_state=self.prevobs_state,
                        stochastic=self.stochastic,
                        infstate_compartments=self.infstate_compartments,
                        observation_compartments=self.observation_compartments
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
                    elif struct == 'Numpy_Vec_Encode':
                        gc.collect()
                        tracemalloc.start() #track memory
                        t0 = time.perf_counter() #track time
                        for _ in range(iters):
                            deltas = dispatcher.get_deltas(current_state=arr_numba, col_idx_map=self.col_idx_map, result_buffer=result_preallocation, stochastic=self.stochastic)
                        t1 = time.perf_counter()
                        peak = tracemalloc.get_traced_memory()[1]
                        tracemalloc.stop()
                    
                    print(f"Sample deltas for {struct}:\n{deltas}\n, data length: {len(deltas)}\n, non-zero counts: {np.count_nonzero(deltas[:, self.col_idx_map['N']] if isinstance(deltas, np.ndarray) else deltas.loc[:, 'N'])}") #debug

                    #concatenate each iteration's result
                    self.time_mem_results.append({
                        'structure': struct,
                        'size': len(self.data_input), #replace size with pop_catg for WAIFW benchmark
                        'iterations': iters,
                        'time_sec': round(t1 - t0, 3),
                        'peak_memory_MB': round(peak / 1024**2, 2)
                    })

                    #print('time_mem_result\n', self.time_mem_results) #debug
        return self.time_mem_results