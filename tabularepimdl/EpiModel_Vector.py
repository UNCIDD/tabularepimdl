import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from pydantic import BaseModel, field_validator, ConfigDict
from tabularepimdl.Rule import Rule


class EpiModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    init_state: pd.DataFrame
    rules: List[List[Rule]]
    t0: float
    tf: float
    dt: float = 0.25
    stoch_policy: str = "rule_based"
    compartment_col: str = "compartment"

    # Internal state
    cur_state_arr: Optional[np.ndarray] = None
    full_epi_arr: Optional[np.ndarray] = None
    col_idx: Optional[Dict[str, int]] = None
    comp_map: Optional[Dict[str, int]] = None
    inv_comp_map: Optional[Dict[int, str]] = None
    column_order: Optional[List[str]] = None
    step_idx: int = 0
    t_current: float = 0.0
    max_steps: int = 0

    @field_validator("init_state", mode="before")
    @classmethod
    def validate_init_state(cls, df):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("init_state must be a DataFrame")
        if not {"N", "T"}.issubset(df.columns):
            raise ValueError("init_state must contain 'N' and 'T'")
        return df

    def model_post_init(self, _):
        self.max_steps = int(np.ceil((self.tf - self.t0) / self.dt))
        self.t_current = self.t0
        self._setup_internal_arrays()

    def _setup_internal_arrays(self):
        df = self.init_state.copy()

        # === Collect compartments from init state ===
        init_comps = set(df[self.compartment_col].astype(str).unique())

        # === Collect compartments from rules ===
        rule_comps = set()
        for ruleset in self.rules:
            for rule in ruleset:
                rule_comps.update(str(x) for x in rule.source_states)
                rule_comps.update(str(x) for x in rule.target_states)

        all_comps = sorted(init_comps.union(rule_comps))

        # === Inject zero-row entries for missing compartments ===
        missing_comps = rule_comps - init_comps
        if missing_comps:
            missing_rows = []
            for group in df["group"].unique():
                for comp in missing_comps:
                    missing_rows.append({
                        "group": group,
                        self.compartment_col: comp,
                        "N": 0.0,
                        "T": self.t0,
                    })
            df = pd.concat([df, pd.DataFrame(missing_rows)], ignore_index=True)

        df = df.sort_values(["group", self.compartment_col]).reset_index(drop=True)

        # === Build compartment encodings ===
        self.comp_map = {label: i for i, label in enumerate(all_comps)}
        self.inv_comp_map = {v: k for k, v in self.comp_map.items()}
        num_comps = len(self.comp_map)

        # === Compile rules using compartment map ===
        for ruleset in self.rules:
            for rule in ruleset:
                if hasattr(rule, "compile"):
                    try:
                        rule.compile(self.comp_map, num_comps=num_comps)
                    except TypeError:
                        rule.compile(self.comp_map)

        # === Encode initial compartment column ===
        df[self.compartment_col] = df[self.compartment_col].astype(str).map(self.comp_map).astype(np.int32)

        # === Initialize column order and index map ===
        self.column_order = list(df.columns)
        self.col_idx = {col: i for i, col in enumerate(self.column_order)}
        n_rows = df.shape[0]
        n_cols = len(self.column_order)

        self.cur_state_arr = df[self.column_order].to_numpy(dtype=np.float32)
        self.cur_state_arr[:, self.col_idx["T"]] = self.t0  # Set initial time

        self.full_epi_arr = np.zeros((self.max_steps + 1, n_rows, n_cols), dtype=np.float32)
        self.full_epi_arr[0] = self.cur_state_arr.copy()
        self.step_idx = 1

    def do_timestep(self):
        delta_arr = np.zeros_like(self.cur_state_arr)

        for ruleset in self.rules:
            for rule in ruleset:
                delta = rule.apply(self.cur_state_arr, self.col_idx, self.dt)
                delta_arr += delta

        self.cur_state_arr += delta_arr
        self.t_current += self.dt
        self.cur_state_arr[:, self.col_idx["T"]] = self.t_current
        self.full_epi_arr[self.step_idx] = self.cur_state_arr
        self.step_idx += 1

    def run(self):
        while self.t_current < self.tf and self.step_idx <= self.max_steps:
            self.do_timestep()

    def get_cur_state(self) -> pd.DataFrame:
        df = pd.DataFrame(self.cur_state_arr, columns=self.column_order)
        df[self.compartment_col] = df[self.compartment_col].astype(int).map(self.inv_comp_map)
        if df[self.compartment_col].isna().any():
            raise RuntimeError("Unrecognized compartment code in current state array.")
        return df

    def get_full_epi(self) -> pd.DataFrame:
        arr = self.full_epi_arr[:self.step_idx].reshape(-1, self.cur_state_arr.shape[1])
        df = pd.DataFrame(arr, columns=self.column_order)
        df[self.compartment_col] = df[self.compartment_col].astype(int).map(self.inv_comp_map)
        if df[self.compartment_col].isna().any():
            raise RuntimeError("Unrecognized compartment code in full epidemic trajectory.")
        return df

    def reset(self):
        self.t_current = self.t0
        self._setup_internal_arrays()
        return self.get_cur_state()
