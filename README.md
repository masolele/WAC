# WAC — World AgroCommodities: Earth Observation for Deforestation-Free Supply Chains

This repository contains code, models, and utilities developed under the **World AgroCommodities (WAC)** project, which aims to support deforestation-free supply chains (EUDR) using satellite data and ancillary sources.

## Project Goals

- Map commodity parcels & crop types over demonstration sites
- Detect deforestation / land conversion after December 2020
- Provide scalable, open, reproducible methods for EU Member States by making use of **openEO** and **CDSE**
- Validate algorithms and support national uptake

## Installation & Dependencies


Clone the repository:
```git clone https://github.com/masolele/WAC
cd WAC
```


(Optional) Create and activate a virtual environment:

```
python3 -m venv venv
source venv/bin/activate
```

#### Users
From inside their virtual environments, users can install the package as follows:

```
pip install .
```

To be able to run the notebooks, the optional `notebooks` dependency has to be installed as well:

```
pip install .[notebooks]
```

#### Developers

The `world_agrocommodities` repository uses `uv` as package manager (https://github.com/astral-sh/uv). From the `uv.lock` file, developers can install all necessary dependencies (including the optional groups), by running

```
uv sync --all-extras
```

This will automatically create a venv used by `uv`, you can activate the venv from the root of the repository:

```
source .venv/bin/activate
```
Afterwards, developers can install `world_agrocommodities` in editable mode and install the git hooks

```
uv pip install -e . && uv run pre-commit install
```

Developers who want to avoid `uv` alltogether, can install the package and all optional dependencies in their favourite environment using `pip`

```
pip install -e .[dev,notebooks]
```
