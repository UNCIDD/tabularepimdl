# `tabularepimdl` - a flexible, rule-based framework for constructing tabular epidemic models in Python. 
The model facilitates the simulation of complex outbreak dynamics across multiple populations and species, supporting modular processes such as infection, recovery, death, birth, and movement.

## Getting Started

### Installation
To install `tabularepimdl`, users can clone the repository and install it in editable mode:

```bash
# 1. Clone the repo
git clone https://github.com/UNCIDD/tabularepimdl.git
cd tabularepimdl
# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate # Windows
source .venv/bin/activate # macOS/Linux
# 3. Install the package in "editable" mode, with test and dev dependencies
pip install -e ".[test,dev]"
```

or from github:
```bash
# 1. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate # Windows
source .venv/bin/activate # macOS/Linux
# 2. Install the package
pip install "git+https://github.com/UNCIDD/tabularepimdl.git"
```

### Project Structure
```
tabularepimdl/
├── docs/                # High-level documents for each epidemic rule
├── tests/               # Unit tests
├── examples/            # Example simulations
├── src/
│   └── tabularepimdl/   # Individual process rules (infection, birth, death, etc.)
├── pyproject.toml       # Package configuration and dependencies
├── LICENSE              # MIT license
└── README.md            # Project description
```

## Features
**Modular Architecture**: Easily compose models with interchangeable processes.

**Multi-Population Support**: Simulate interactions across diverse populations and species.

**Configurable Rules**: Define custom processes using Pydantic-based configurations.

**Performance Optimized**: Leverage efficient data structures for large-scale simulations.

## Usage
Here's a minimal example to define and run a simple SIR model using `tabularepimdl`:
```python
import tabularepimdl as tepi
import pandas as pd

population_df = pd.DataFrame({'InfState': ['S', 'I'], 'N': [990.0, 10.0], 'T': [0, 0]})
infstate_compartments = ['S', 'I', 'R']

infect_rule = tepi.SimpleInfection_Vec_Encode(
    beta=0.5, column='InfState',
    infstate_compartments=infstate_compartments, column_categories=infstate_compartments,
)
recover_rule = tepi.SimpleTransition_Vec_Encode(
    column='InfState', from_st='I', to_st='R', rate=0.25,
    infstate_compartments=infstate_compartments, column_categories=infstate_compartments,
)
epi_mdl = tepi.EpiModel_Vec_Encode_1_5(init_state=population_df, rules=[infect_rule, recover_rule])

epi_mdl.do_timestep(dt=0.25)
print(epi_mdl.current_state())
```