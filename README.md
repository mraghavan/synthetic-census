# Getting things to run

## Requirements
- Data (see [below](#getting-the-data))
- `python3`, and the following packages:
  - [numpy](https://numpy.org/)
  - [scipy](https://scipy.org/)
  - [pandas](https://pandas.pydata.org/)
  - [Gurobi](https://www.gurobi.com/), with a valid license and the python package `gurobipy` installed (`python3 -m pip install gurobipy`)

## Getting the data
We'll need a combination of data from NHGIS and PUMS from the census. For a particular state:
- Get PUMS data [here](https://www2.census.gov/census_2010/12-Stateside_PUMS/)
- For NHGIS:
  1. Start [here](https://data2.nhgis.org/main)
  2. Choose dataset `2010_SF1a`
  3. Choose geographic level `Block`
  4. Choose tables `P3`, `P16`, `P16A`--`P16G`, `P28`, and `P28A`--`P28G`. In general, we'd prefer to use the more granular `A`--`G` tables, but sometimes we'll have to fall back on the less granular ones.
  5. Click CONTINUE, and click CONTINUE again.
  6. Select the right state under GEOGRAPHIC EXTENTS
  7. SUBMIT

## Basic setup
Create a file called `params.json` in the `code/` directory.
The input data will need to be in `[data]/[state]`, and the output will be written to `output/state`.
`num_sols` specifies the maximum number of soluions to be returned for a block.
`write` controls whether output files will be written.
Here's a sample:
```
{
    "state": "VT",
    "data": "$HOME/Desktop/census_data/",
    "output": "$HOME/Desktop/census_data/output/",
    "num_sols": 100,
    "write": 0
}

```

## Storing the data
Put the data in the directory `[data]/[state]`, e.g., `$HOME/Desktop/census_data/VT`. The NHGIS file you downloaded should be `nhgisxxxx_csv.zip`. Unzip it. Rename the csv file it contains to `[data]/[state]/block_data.csv`.

Verify that the data files are in the right place by running `python3 config.py`.

## Pre-processing the data
We'll need to build both the microdata distribution and the block-level dataframe.
To test building the microdata distribution, run `python3 build_micro_dist.py` (optional).
To build the block-level dataframe, run `python3 build_block_df.py` (required). This will create the file `[data]/[state]/block_data_cleaned.csv`.

## Building the distribution
Make sure you've fulfilled the [requirements](#requirements).
Run `python3 partition_blocks.py`.
This will create `.pkl` file for each block in `[output]/[state]`.

### Running on the RC cluster with `slurm`
Run `sbatch runscript.sh`.

## Generating a dataset
TBD, will need to load each distribution and sample from it.

# How it works
To be written
