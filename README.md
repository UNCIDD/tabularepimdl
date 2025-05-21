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
# 3. Install dependencies
pip install -r requirements.txt
# 4. Install the package in "editable" mode
pip install --editable .
```

or from github:
```bash
# 1. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate # Windows
source .venv/bin/activate #macOS/Linux
# 2. Install the package
pip install "git+https://github.com/UNCIDD/tabularepimdl.git"
```

### Project Structure
```
tabularepimdl/
├── docs/                # High-level documents for each epidemic rule
├── epitest/             # Unit tests
├── examples/            # Example simulations
├── tabularepimdl/       # Individual process rules (infection, birth, death, etc.)
├── setup.py             # Package configuration
└── README.md            # Project description
```

## Features
**Modular Architecture**: Easily compose models with interchangeable processes.

**Multi-Population Support**: Simulate interactions across diverse populations and species.

**Configurable Rules**: Define custom processes using Pydantic-based configurations.

**Performance Optimized**: Leverage efficient data structures for large-scale simulations.

## Usage
Here's a minimal example to define and run a simple SIR model using `tabularepimdl`:
```
import tabularepimdl as tepi

# Define a population DataFrame (population_df) as per the model requirements

infect_rule = tepi.SimpleInfection(beta=0.5, column='InfState')
recover_rule = tepi.SimpleTransition(column='InfState', from_st='I', to_st='R', rate=0.25)
epi_mdl = tepi.EpiModel(init_state = population_df, rules=[infect_rule, recover_rule])

epi_model.do_timestep(dt=0.25)

```