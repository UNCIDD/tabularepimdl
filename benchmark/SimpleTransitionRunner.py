from pydantic import BaseModel
from benchmark.SimpleTransitionDispatcher import SimpleTransitionDispatcher

import numpy as np
import pandas as pd
import time
import tracemalloc

class SimpleTransitionRunner(BaseModel):
    """
    Benchmarks time and memory usage of the SimpleTransition Pandas and NumPy versions.
    @param data_sizes: the number of records in a input data.
    @param structure: data structure used for rules.
    @param iterations: the number of iterations to run the rule.
    @param column: Name of the column this rule applies to.
    @param from_st: the state that column transitions from.
    @param to_st: the state that column transitions to.
    @param rate: transition rate per unit time.
    @param stochastic: whether the process is stochastic or deterministic.
    @return: the time and memory usage of the rule with different data sizes, structures and iterations.
    """
    data_sizes: list[int]
    structures: list[str]
    iterations: list[int]
    column: str
    from_st: str
    to_st: str
    rate: float
    stochastic: bool = False

    results: list[dict] = []

    def run(self):
        """
        Creates input data with different sizes and runs deltas calculation with different data structures in different iterations.
        Tracks each combination's time and memory usage.
        """
        for size in self.data_sizes:
            np.random.seed(3)
            infstate_values = np.random.choice(['S', 'I', 'R'], size=size)
            n_values = np.random.randint(1, 10, size=size)
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
                        
                        #{ #spares array input for now
                        #    self.column: np.array([self.from_st] * size),
                        #    'N': np.random.poisson(10, size)
                        #}

                    dispatcher = SimpleTransitionDispatcher(
                        structure=struct,
                        column=self.column,
                        from_st=self.from_st,
                        to_st=self.to_st,
                        rate=self.rate,
                        stochastic=self.stochastic
                    )
                    
                    tracemalloc.start() #track memory
                    t0 = time.perf_counter() #track time
                    for _ in range(iters):
                        deltas = dispatcher.get_deltas(data)
                    t1 = time.perf_counter()
                    peak = tracemalloc.get_traced_memory()[1]
                    tracemalloc.stop()

                    #rint(f"Sample deltas:\n{deltas}\n") #debug

                    self.results.append({
                        'structure': struct,
                        'size': size,
                        'iterations': iters,
                        'time_sec': round(t1 - t0, 3),
                        'peak_memory_MB': round(peak / 1024**2, 2)
                    })
        return self.results