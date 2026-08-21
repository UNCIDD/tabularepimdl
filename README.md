# `tabularepimdl` - a flexible, rule-based framework for constructing tabular epidemic models 
`tabularepimdl` is a modular, rule-based framework facilitates the simulation of complex outbreak dynamics across multiple populations and species, supporting multi-strain infection with cross-protective immunity, environmental/reservoir-mediated transmission, and structured contact-matrix transmission between arbitrary population groupings (age, location, or any other category).

## Quick Start for Installation

```bash
git clone https://github.com/UNCIDD/tabularepimdl.git
cd tabularepimdl
uv sync
```

### Activate the virtual environment 
### Windows
```bash
.venv\Scripts\activate
```

### macOS/Linux
```bash
source .venv/bin/activate
```

## Project Structure
```
tabularepimdl/
├── docs/                # High-level documents for each epidemic rule, more to be edited
├── tests/               # Unit tests and Integration tests
├── examples/            # Example simulations built from rules and the model engine
├── legacy/              # initial pandas-based rules, model engine, and experimental stuff
├── src/
│   └── tabularepimdl/   # Individual process rules and the model engine
├── pyproject.toml       # Package configuration and dependencies
├── LICENSE              # MIT license
├── CONTRIBUTING.md      # guidelines and instructions for contributing
└── README.md            # Project description
```

## Features
**Modular Architecture**: Easily compose models with interchangeable processes.

**Multi-Population Support**: Simulate interactions across diverse populations and species.

**Configurable Rules**: Define custom processes using Pydantic-based configurations.

**Performance Optimized**: Leverage efficient data structures for large-scale simulations.