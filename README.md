# Spectral Redistricting

An interface for the [GerryChain](https://gerrychain.readthedocs.io/en/latest/) library to
facilitate running redistricting chains.

## Installation
Run the following commands to install the required packages using
[Anaconda](https://www.anaconda.com/products/individual#Downloads):
```bash
$ conda create -n redist -c conda-forge -y pip=24.0 geopandas=0.14.3 networkx=3.2.1 mapclassify=2.6.1 matplotlib=3.8.3 tqdm=4.66.2 scipy=1.12.0 numpy=1.26.4
$ conda activate redist
$ pip install gerrychain==0.3.1
```

## Running the Script

[Redistricting](./redistricting/) contains the main python scripts used for redistricting runs.
Import the [Redistricting](./redistricting/redistricting.py) class to create an instance that can
then be run. Run the `demo.py` file for an example run.
```bash
$ python demo.py
```

## Data

All data that were collected can be found in the file `data.json`.
The file contains the following statistics for 79,932 generated plans, each with 7 districts:
- '*graph name*': The name of the plan's graph.
- '*proposal*': The proposal used to generate the plan.
- '*cut edges*': How many edges are cut by the plan.
- '*size disparity*': Ratio of the plan's largest and smallest districts' sizes.
- '*size deviation*': Pair-wise size tolerance achieved by the plan's districts.
- '*population disparity*': Ratio of the plan's largest and smallest districts' populations.
- '*population deviation*': Pair-wise population tolerance achieved by the plan's districts.
- '*contiguous*': Boolean indicating if the plan's districts are contiguous.
- '*time elapsed*': How long the plan took to generate (unavailable for RevRecom).