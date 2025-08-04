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
    store_stride: Optional[int] = None

    # Internal state
    cur_state_arr: Optional[np.ndarray] = None
    full_epi_arr: Optional[np.ndarray] = None
    col_idx: Optional[Dict[str, int]] = None
    comp_map: Optional[Dict[str, int]] = None
    inv_comp_map: Optional[Dict[int, str]] = None
    column_order: Optional[List[str]] = None
    step_idx: int = 1
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
        if self.store_stride is None:
            self.store_stride = max(1, round(1.0 / self.dt))
        self.t_current = self.t0
        self._setup_internal_arrays()

    def _setup_internal_arrays(self):
        df = self.init_state.copy()

        # === Collect compartments from init state and rules ===
        init_comps = set(df[self.compartment_col].astype(str).unique())
        rule_comps = {
            str(x)
            for ruleset in self.rules
            for rule in ruleset
            for x in (rule.source_states + rule.target_states)
        }
        all_comps = sorted(init_comps.union(rule_comps))

        # === Ensure all group_col fields are present and inject missing compartments per rule ===
        for ruleset in self.rules:
            for rule in ruleset:
                group_col = getattr(rule, "group_col", "group")
                if group_col not in df.columns:
                    df[group_col] = 0

                rule_comps = set(rule.source_states + rule.target_states)
                present = df[self.compartment_col].astype(str).unique()
                missing = rule_comps - set(present)

                if missing:
                    unique_groups = df[group_col].unique()
                    missing_rows = [
                        {group_col: g, self.compartment_col: comp, "N": 0.0, "T": self.t0}
                        for g in unique_groups
                        for comp in missing
                    ]
                    df = pd.concat([df, pd.DataFrame(missing_rows)], ignore_index=True)

        # === Finalize compartment mapping ===
        self.comp_map = {label: i for i, label in enumerate(all_comps)}
        self.inv_comp_map = {v: k for k, v in self.comp_map.items()}
        num_comps = len(self.comp_map)

        # === Determine group column for dense reindexing ===
        group_col_candidates = list({getattr(rule, "group_col", None) for ruleset in self.rules for rule in ruleset})
        group_col_candidates = [col for col in group_col_candidates if col]
        group_col = group_col_candidates[0] if group_col_candidates else "group"
        if group_col not in df.columns:
            df[group_col] = 0

        # === Replace string compartment labels with integer codes ===
        df[self.compartment_col] = df[self.compartment_col].astype(str).map(self.comp_map).astype(np.int32)

        # === Ensure rectangular group-major layout ===
        unique_groups = np.sort(df[group_col].unique())
        rows = []
        for g in unique_groups:
            for comp_code in range(num_comps):
                mask = (df[group_col] == g) & (df[self.compartment_col] == comp_code)
                if mask.any():
                    row = df.loc[mask].iloc[0]
                else:
                    row = {group_col: g, self.compartment_col: comp_code, "N": 0.0, "T": self.t0}
                rows.append(row)
        df = pd.DataFrame(rows)

        # === Compile rules with integer-encoded compartment map ===
        for ruleset in self.rules:
            for rule in ruleset:
                if hasattr(rule, "compile"):
                    try:
                        rule.compile(self.comp_map, num_comps=num_comps)
                    except TypeError:
                        rule.compile(self.comp_map)

        # === Final column order: numeric dtype columns only ===
        self.column_order = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        self.col_idx = {col: i for i, col in enumerate(self.column_order)}
        n_rows, n_cols = df.shape[0], len(self.column_order)

        self.cur_state_arr = df[self.column_order].to_numpy(dtype=np.float32)
        self.cur_state_arr[:, self.col_idx["T"]] = self.t0

        n_store_steps = (self.max_steps // self.store_stride) + 1
        self.full_epi_arr = np.zeros((n_store_steps, n_rows, n_cols), dtype=np.float32)
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

        if self.step_idx < self.full_epi_arr.shape[0] and (self.step_idx * self.store_stride * self.dt) <= self.tf:
            if int(round(self.t_current / self.dt)) % self.store_stride == 0:
                self.full_epi_arr[self.step_idx] = self.cur_state_arr.copy()
                self.step_idx += 1

    def run(self):
        while self.t_current < self.tf:
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

