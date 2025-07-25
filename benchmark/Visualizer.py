import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from pydantic import BaseModel, field_validator, ConfigDict

class Visualizer(BaseModel):
    """
    Visualizes the time and memory usage of rule simulations. Purposely keep the visual class a separate module.
    @param runner_results: a list of dictionaries containing time, memory, structure, data size, iteration information.
    @param df: dataframe converted from the above dictionaries.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    runner_results: List[Dict[str, Any]]
    df: pd.DataFrame = None  #will be set after validation

    @field_validator("runner_results")
    @classmethod
    def validate_results(cls, result):
        if not isinstance(result, list):
            raise ValueError("results must be a list of dictionaries.")
        if not all(isinstance(item, dict) for item in result):
            raise ValueError("each item in results must be a dict.")
        return result

    def model_post_init(self, _):
        #converts dict to a DataFrame once the model is created
        self.df = pd.DataFrame(self.runner_results)
        self.df['label'] = self.df.apply(
            lambda row: f"{int(row['size']):,} rows \n{row['iterations']} iters", axis=1
        )

    def plot(self):
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # ---- Time plot ----
        sns.barplot(
            data=self.df,
            x="label",
            y="time_sec",
            hue="structure",
            ax=axes[0]
        )
        axes[0].set_title("Runtime (seconds) by Structure & Condition")
        axes[0].set_xlabel("Data Size & Iterations")
        axes[0].set_ylabel("Time (seconds)")
        axes[0].tick_params(axis='x', rotation=30)

        # ---- Memory plot ----
        sns.barplot(
            data=self.df,
            x="label",
            y="peak_memory_MB",
            hue="structure",
            ax=axes[1]
        )
        axes[1].set_title("Peak Memory (MB) by Structure & Condition")
        axes[1].set_xlabel("Data Size & Iterations")
        axes[1].set_ylabel("Memory (MB)")
        axes[1].tick_params(axis='x', rotation=30)

        # Add legends
        axes[0].legend(title="Backend")
        axes[1].legend(title="Backend")

        plt.tight_layout()
        plt.show()
