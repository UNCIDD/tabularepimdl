from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
from benchmark.MultiStrainInfectiousProcessDispatcher import MultiStrainInfectiousProcessDispatcher

import numpy as np
import pandas as pd
import time
import tracemalloc
import gc

class MultiStrainInfectiousProcessRunner(BaseModel):
    """
    Benchmarks time and memory usage of the SharedTraitInfection Pandas and NumPy versions.
    @param data_sizes: the number of records/rows in a input data.
    @param data_input: the test data for rules.
    @param structure: data structure used for rules. E.g. Pandas, Numpy, Numpy_Encode
    @param iterations: the number of iterations to run the rule.
    @param betas: a beta for each strain.
    @param columns: the strain columns for infection state. The number of strains should be the same length and order as betas.
    @param columns_all_categories: all the infection state categories the strain columns should have.
    @param cross_protect: a N(strain)*N(strain) matrix of cross protections.
    @param s_st: the state for susceptibles, assumed to be S.
    @param i_st: the state for infectious, assumed to be I.
    @param r_st: the state for immune/recovered, assumed to be R.
    @param inf_to: the state infectious folks go to, assumed to be I.
    @param stochastic: whether the process is stochastic or deterministic.
    @param freq_dep: whether this model is a frequency dependent model.
    @param infstate_compartments: the infection compartments used in epidemics. e.g. ['I', 'R', 'S']
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data_sizes: list[int]
    data_input: pd.DataFrame
    structures: list[str]
    iterations: list[int]
    betas: np.ndarray
    columns: list[str]
    cross_protect: np.ndarray
    s_st: str
    i_st: str
    r_st: str
    inf_to: str
    stochastic: bool
    freq_dep: bool
    col_idx_map: dict[str, int] = Field(default_factory=dict)
    infstate_compartments: list[str] = Field(default_factory=list)
    columns_all_categories: list[str] = Field(default_factory=list)

    time_mem_results: list[dict] = Field(default_factory=list)

    _infstate_comp_map: dict[str, int] = PrivateAttr(default_factory=dict)

    def model_post_init(self, _):
        self._infstate_comp_map = {comp: i for i, comp in enumerate(sorted(self.infstate_compartments))}


    def encode_column_sorted(self, compartment_list, df_col):
        #print('comp list:', compartment_list)
        unique_vals = list(sorted(set(compartment_list)))  # Sorted to ensure consistent index, set needs to be inside sorted because the order in a set is not guranteed
        #print('unique values:', unique_vals)
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        #print('mapping\n', mapping) #debug
        return df_col.map(mapping)
    
    def run(self) -> list[dict]:
        """
        Creates input data with different sizes and runs deltas calculation with different data structures in different iterations.
        Tracks each combination's time and memory usage.
        """
        for size in self.data_sizes:
            
            for struct in self.structures:
                for iters in self.iterations:
                    print(f"\nRunning {struct} | size={len(self.data_input)} | iterations={iters}") #replace size with pop_catg

                    if struct == 'Pandas': #provide dataframe to Pandas
                        data = self.data_input
                    elif struct == 'Numpy_Vec_Encode_1' or struct == 'Numpy_Vec_Encode_2': #provide true Numpy array to Numpy_Encode
                        print('input data\n', self.data_input)
                        encoded_cols = [self.encode_column_sorted(self.columns_all_categories, self.data_input[col]).to_numpy() for col in self.columns] #put each encoded col in a list
                        print('encoded\n', encoded_cols)
                        arr_numba = np.column_stack((*encoded_cols, self.data_input['N'], self.data_input['T']))
                        arr_numba = arr_numba.astype(np.float64) #this makes all columns a float number, it will later cause float indexing error for Numba, but WAIFW rule will convert group_col category back to integers.
                        print('arr_numba\n', arr_numba)
                        n_rows = arr_numba.shape[0] #detect the number of rows and columns in input array
                        n_cols = arr_numba.shape[1]
                        result_preallocation = np.empty((n_rows * 2, n_cols), dtype=np.float64) #preallocate a result array
                    
                    dispatcher = MultiStrainInfectiousProcessDispatcher(
                        structure=struct,
                        betas=self.betas,
                        columns=self.columns,
                        columns_all_categories=self.columns_all_categories,
                        cross_protect=self.cross_protect,
                        s_st=self.s_st,
                        i_st=self.i_st,
                        r_st=self.r_st,
                        inf_to=self.inf_to,
                        stochastic=self.stochastic,
                        freq_dep=self.freq_dep,
                        infstate_compartments=self.infstate_compartments
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
                    elif struct == 'Numpy_Vec_Encode_1':
                        gc.collect()
                        tracemalloc.start() #track memory
                        t0 = time.perf_counter() #track time
                        for _ in range(iters):
                            deltas = dispatcher.get_deltas(current_state=arr_numba, col_idx_map=self.col_idx_map, result_buffer=result_preallocation, stochastic=self.stochastic)
                        t1 = time.perf_counter()
                        peak = tracemalloc.get_traced_memory()[1]
                        tracemalloc.stop()
                    elif struct == 'Numpy_Vec_Encode_2':
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