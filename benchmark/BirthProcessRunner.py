from pydantic import BaseModel, Field, PrivateAttr, ConfigDict
from benchmark.BirthProcessDispatcher import BirthProcessDispatcher

import numpy as np
import pandas as pd
import time
import tracemalloc
import gc

class BirthProcessRunner(BaseModel):
    """
    Benchmarks time and memory usage of the BirthProcess Pandas and NumPy versions.
    @param data_sizes: the number of records/rows in a input data.
    @param structure: data structure used for rules. E.g. Pandas, Numpy, Numpy_Encode
    @param iterations: the number of iterations to run the rule.
    @param rate: Birth rate per timestep.
    @param stochastic: whether the process is stochastic or deterministic.
    @param col_idx_map: mapping of input data columns and their column index. E.g. col_idx_map = {'InfState': 0, 'N': 1}.
    @param _infstate_comp_map: mapping between infectin states values and their categorical values. E.g. state_map = {'I': 0, 'R': 1, 'S': 2}.
    @param infstate_compartments: the infection compartments used in epidemics. E.g.infstate_compartments = ['S', 'I', 'R'].
    return: the time and memory usage of the rule with different data sizes, structures and iterations.
    """
    data_sizes: list[int] #inital value can be 15
    structures: list[str]
    iterations: list[int]
    rate: float
    column_to_sort: str
    #start_state_sig: dict | pd.DataFrame #not needed in runner since a dict is given below
    stochastic: bool = False
    col_idx_map: dict[str, int] = Field(default_factory=dict)
    infstate_compartments: list[str] = Field(default_factory=list)
    
    
    time_mem_results: list[dict] = []
    _age_all_categories: list[str] = ["0 to 4", "5 to 9", "10 to 14", "15 to 19", "20 to 24", "25 to 29", "30 to 34", \
                                       "35 to 39", "40 to 44", "45 to 49", "50 to 54", "55 to 59", "60 to 64", "65 to 69", "70+"]
    #_infstate_comp_map: dict[str, int] = PrivateAttr(default_factory=dict)

    #def model_post_init(self, _):
    #    self._infstate_comp_map = {comp: i for i, comp in enumerate(sorted(self.infstate_compartments))}

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
            start_age=0
            end_age=70
            age_step= (end_age-start_age)/(size-1)

            age_struct_pop = pd.DataFrame({
                'InfState' : pd.Categorical(["S"]*size, categories=["I","R","S"]),
                'AgeCat': ["{} to {}".format(int(i), int(i+ (age_step-1))) for i in np.arange(start_age, end_age, age_step)]+["{}+".format(end_age)],
                'N' : [10, 20, 30, 40, 50, 60,  70, 80, 90, 100, 101, 102, 103, 104, 105],
                'T': 0
            })

            age_struct_pop = age_struct_pop.sample(frac=1, random_state=3).reset_index(drop=True) #row-level random shuffle
            #print('shuffled age_struct_pop\n', age_struct_pop)
            
            start_state_sig_dict = {
                'InfState': 'S',
                'AgeCat': '0 to 4',
                'N': 10,
                'T': 0
            }
            #print('age structure\n', age_struct_pop)
            for struct in self.structures:
                for iters in self.iterations:
                    print(f"\nRunning {struct} | size={size} | iterations={iters}")
                    if struct == 'Pandas': #provide dataframe to Pandas
                        data = age_struct_pop
                    elif struct == 'Numpy_Encode': #provide true Numpy array to Numpy_Encode
                        InfState_encode = self.encode_column_sorted(self.infstate_compartments, age_struct_pop['InfState'])
                        Age_encode = self.encode_column_sorted(self._age_all_categories, age_struct_pop['AgeCat'])
                        age_array = np.column_stack((InfState_encode, Age_encode, age_struct_pop['N'], age_struct_pop['T']))
                        age_array = age_array.astype(np.float64) #this makes all columns a float number, it will later cause float indexing error for Numba, but WAIFW rule will convert group_col category back to integers.
                        #print('current_state array\n', age_array)
                        n_rows = age_array.shape[0] #detect the number of rows and columns in input array
                        n_cols = age_array.shape[1]
                        result_preallocation = np.empty((n_rows * 2, n_cols), dtype=np.float64) #preallocate a result array
                    dispatcher = BirthProcessDispatcher(
                        structure=struct,
                        rate=self.rate,
                        column_to_sort=self.column_to_sort,
                        start_state_sig = start_state_sig_dict,
                        stochastic=self.stochastic,
                        infstate_compartments=self.infstate_compartments
                    )
                    #print('dispatcher created ok\n')
                    if struct  == 'Pandas':
                        gc.collect()
                        tracemalloc.start() #track memory
                        t0 = time.perf_counter() #track time
                        for _ in range(iters):
                            deltas = dispatcher.get_deltas(current_state=data, stochastic=self.stochastic)
                        t1 = time.perf_counter()
                        peak = tracemalloc.get_traced_memory()[1]
                        tracemalloc.stop()
                    elif struct == 'Numpy_Encode':
                        gc.collect()
                        tracemalloc.start() #track memory
                        t0 = time.perf_counter() #track time
                        for _ in range(iters):
                            deltas = dispatcher.get_deltas(current_state=age_array, col_idx_map=self.col_idx_map, result_buffer=result_preallocation, stochastic=self.stochastic)
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