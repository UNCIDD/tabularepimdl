from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
from benchmark.SharedTraitInfectionDispatcher import SharedTraitInfectionDispatcher

import numpy as np
import pandas as pd
import time
import tracemalloc
import gc

class SharedTraitInfectionRunner(BaseModel):
    """
    Benchmarks time and memory usage of the SharedTraitInfection Pandas and NumPy versions.
    @param data_sizes: the number of records/rows in a input data.
    @param data_input: the test data for rules.
    @param structure: data structure used for rules. E.g. Pandas, Numpy, Numpy_Encode
    @param iterations: the number of iterations to run the rule.
    @param inf_col: the infection state column for this infectious process.
    @param in_beta: transmission rate if trait shared.
    @param out_beta: transmission rrate if trait not shared.
    @param trait_col: the trait column shared by different populations.
    @param trait_col_all_categories: all the categories the trait column should have.
    @param s_st: the state for susceptibles, assumed to be S.
    @param i_st: the state for infectious, assumed to be I.
    @param inf_to: the state infectious folks go to, assumed to be I.
    @param stochastic: whether the process is stochastic or deterministic.
    @param col_idx_map: mapping of input data columns and their column index. E.g. col_idx_map = {'InfState': 0, 'N': 1}.
    @param _infstate_comp_map: mapping between infectin states values and their categorical values. E.g. state_map = {'I': 0, 'R': 1, 'S': 2}.
    @param infstate_compartments: the infection compartments used in epidemics. E.g.infstate_compartments = ['S', 'I', 'R'].
    return: the time and memory usage of the rule with different data sizes, structures and iterations.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data_sizes: list[int]
    data_input: pd.DataFrame
    structures: list[str]
    iterations: list[int]
    inf_col: str
    in_beta: float
    out_beta: float
    trait_col: str
    trait_col_all_categories: list[str | int]
    s_st: str = "S"
    i_st: str = "I"
    inf_to: str = "I"
    stochastic: bool = False
    col_idx_map: dict[str, int] = Field(default_factory=dict)
    infstate_compartments: list[str] = Field(default_factory=list)

    time_mem_results: list[dict] = Field(default_factory=list)

    _infstate_comp_map: dict[str, int] = PrivateAttr(default_factory=dict)

    def model_post_init(self, _):
        self._infstate_comp_map = {comp: i for i, comp in enumerate(sorted(self.infstate_compartments))}


    def encode_column_sorted(self, df_col):
        unique_vals = sorted(df_col.unique())  # Sorted to ensure consistent index
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        #print('mapping\n', mapping) #debug
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
                        InfState_encode = self.encode_column_sorted(self.data_input[self.inf_col]) #infstate column is sorted
                        #print('input trait val\n', self.data_input[self.trait_col])
                        Trait_encode = self.encode_column_sorted(self.data_input[self.trait_col]) #categorical column is sorted
                        #print('Trait sorted code\n', Trait_encode)
                        arr_numba = np.column_stack((InfState_encode, Trait_encode, self.data_input['N'], self.data_input['T']))
                        arr_numba = arr_numba.astype(np.float64) #this makes all columns a float number, it will later cause float indexing error for Numba, but WAIFW rule will convert group_col category back to integers.
                        #print('arr_numba\n', arr_numba)
                        n_rows = arr_numba.shape[0] #detect the number of rows and columns in input array
                        n_cols = arr_numba.shape[1]
                        result_preallocation = np.empty((n_rows * 2, n_cols), dtype=np.float64) #preallocate a result array
                    
                    dispatcher = SharedTraitInfectionDispatcher(
                        structure=struct,
                        inf_col=self.inf_col,
                        in_beta=self.in_beta,
                        out_beta=self.out_beta,
                        trait_col=self.trait_col,
                        trait_col_all_categories=self.trait_col_all_categories,
                        s_st=self.s_st,
                        i_st=self.i_st,
                        inf_to=self.inf_to,
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