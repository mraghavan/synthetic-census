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
  4. Choose tables `P3`, `P5`, `P16`, `P16A`--`P16H`, `P28`, and `P28A`--`P28H`. In general, we'd prefer to use the more granular `A`--`H` tables, but sometimes we'll have to fall back on the less granular ones.
  5. Click CONTINUE, and click CONTINUE again.
  6. Select the right state under GEOGRAPHIC EXTENTS
  7. SUBMIT

## Basic setup
Create a file called `params.json` in the `code/` directory.
The input data will need to be in `[data]/[state]`, and the output will be written to `output/state`.
`num_sols` specifies the maximum number of soluions to be returned for a block.
`write` controls whether output files will be written.
Make sure you set `write` to 1 before you run.
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
Put the data in the directory `[data]/[state]`, e.g., `$HOME/Desktop/census_data/VT`.
The NHGIS file you downloaded should be `nhgisxxxx_csv.zip`.
Unzip it.
Rename the csv file it contains to `[data]/[state]/block_data.csv`.

Verify that the data files are in the right place by running `python3 config.py`.

## Pre-processing the data
We'll need to build both the microdata distribution and the block-level dataframe.

To test building the microdata distribution, run `python3 build_micro_dist.py` (optional).

To build the block-level dataframe, run `python3 build_block_df.py` (required). This will create the file `[data]/[state]/block_data_cleaned.csv`.
Computationally, this is fairly light and can be run locally.

## Building the distribution
Make sure you've fulfilled the [requirements](#requirements).
Run `python3 partition_blocks.py` (if running locally).
This will create `.pkl` file in `[output]/[state]/`.
To parallelize, run `python3 partition_blocks.py [i] [total]` for `i` in `1..total` on separate threads/machines (see below for `slurm` usage).
If given a third argument [name], files will be named `[name]_[i]_[total].pkl`.
Otherwise, they will be named `[i]_[total].pkl`
To turn the `.pkl` files into a dataset, run `python3 sample_from_dist.py [name]`.

## Running on the RC cluster with `slurm`
If using `slurm`, run `./generate_dataset.sh`.
This will use a map-reduce-style approach to generate a single dataset, which will be written to `[output]/[state]/[job_id]_synthetic.csv`, where `job_id` is the first line of the script's output.
Make sure the directory `code/out_files` exists before running.

### Map
The map phase is done by `sbatch runscript.sh`, which should look something like
```
#!/bin/bash
#SBATCH -c 4                # Number of cores (-c)
#SBATCH -t 0-0:15          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=5000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o out_files/census.%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e out_files/census.%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --array=1-50
#SBATCH --mail-type=END
module load python/3.8.5-fasrc01
module load gurobi/9.0.2-fasrc01
python3 -m pip install gurobipy
python3 partition_blocks.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT $SLURM_ARRAY_JOB_ID
```
This divides the blocks into a number of jobs (in this case 50) and, for each block, samples a household.
The results, along with the quality of the solution found, are written to `[output]/[state]/[job_id]_[i]_[total].pkl`.
The requirements are reasonably well-calibrated for Vermont, which contains 17,541 non-empty census blocks (takes at most 7 minutes per job).
Scale the number of parralel jobs (`#SBATCH --array=1-{num}`) appropriately.
To prevent too many from running at the same time, use `#SBATCH --array=1-{num}%{max}` where `max` is the maximum number of jobs to run at a time.
Gurobi can use multiple cores, which is why I've set the number of cores to 4. I haven't experimented with increasing/decreasing this.

### Reduce
The reduce phase is done by `sbatch sample_script.sh [job_id]`, where `[job_id]` should be the id of the map phase. `sample_script.sh` should look like
```
#!/bin/bash
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 0-0:15          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=1000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o out_files/samp.%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e out_files/samp.%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
module load python/3.8.5-fasrc01
python3 sample_from_dist.py $1
```
This will write the sampled dataset in the file `[output]/[state]/[job_id]_synthetic.csv`, where each row is a sampled household. This should run fairly quickly (~20 seconds for VT).

# Data schema
Each row in the synthetic dataset corresponds to a single household.
The synthetic dataset has the following columns:
| Name | Description|
|---|---|
|`YEAR`|(from NHGIS data) |
|`STATE`|(from NHGIS data) |
|`STATEA`|(from NHGIS data) |
|`COUNTY`|(from NHGIS data) |
|`COUNTYA`|(from NHGIS data) |
|`COUSUBA`|(from NHGIS data) |
|`TRACTA`|(from NHGIS data) |
|`BLKGRPA`|(from NHGIS data) |
|`BLOCKA`|(from NHGIS data) |
|`NAME`|(from NHGIS data) |
|`BLOCK_TOTAL`| Total number of people in that block |
|`BLOCK_18_PLUS`| Total number of 18+ people in that block (NOTE: this doesn't necessarily match NHGIS data. See `AGE_ACCURACY`.) |
|`TOTAL`| Total number of people in this household |
|`W`| Total number of people in this household of race 'White alone' |
|`B`|Total number of people in this household of race 'Black or African American alone' |
|`AI_AN`|Total number of people in this household of race 'American Indian and Alaska Native alone' |
|`AS`|Total number of people in this household of race 'Asian alone' |
|`H_PI`|Total number of people in this household of race 'Native Hawaiian and Other Pacific Islander alone' |
|`OTH`|Total number of people in this household of race 'Some Other Race alone' |
|`TWO_OR_MORE`|Total number of people in this household of race 'Two or More Races' |
|`NUM_HISP`| Number of Hispanic people in this household |
|`18_PLUS`| Total number of people in this household aged 18 or more |
|`HH_NUM`| Household index within block |
|`ACCURACY`| How accurate a solver was used (1, 2, or 3; see below)|
|`AGE_ACCURACY`| Whether this block had accurate age information |
|`identifier`| Unique block ID |

`ACCURACY = 1` means the households match the NHGIS data in terms of:
- Total number of people of each race X Hispanic combination in the block (see table `P5`)
- Number of households of each (householder race, family status, size) type (see tables `P28A`--`H`)
- (if `AGE_ACCURACY` is true) Number of people age 18+ living in a household for each householder race and ethnicity (see tables `P16A`--`H`)

`ACCURACY = 2` means the households match the NHGIS data in terms of:
- Total number of people of each race X Hispanic combination in the block (see table `P5`)
- Total number of households (see table `P28`)
- (if `AGE_ACCURACY` is true) Total number of people 18+ (see table `P16`)

`ACCURACY = 3` means the households match the NHGIS data in terms of:
- Total number of people of each race in the block (see table `P3`)
- Total number of Hispanic people of each race in the block (see table `P5`)
- (if `AGE_ACCURACY` is true) Total number of people 18+ (see table `P16`)

`AGE_ACCURACY` is true if tables `P3` and `P16` agree on the total number of residents in the block, and false otherwise. In VT, this is true for ~98% of blocks.

# Swapping

First, download the 2010 shapefiles for that state from [IPUMS](https://data2.nhgis.org/main) (more detailed instructions coming).

To get swapped data, name the synthetic dataset `[output]/[state]/[name]_synthetic.csv`.
Run `python3 swapping.py [name]`, which will run swap the dataset and write the resulting dataset to `[output]/[state]/[name]_swapped.csv`.

The swapping parameters (including the swap rate) can be found and edited in `swapping_params.json`.

## Running on slurm
To run on slurm, first create a `conda` environment with the following commands:
```
conda create --name geopandas
source activate geopandas
conda install geopandas
```
Then, run `sbatch swap_script.sh`.

(Harvard's cluster seems to require `conda` because the standard `python3` won't allow the installation of `geopandas` for some reason.
In the future, it may be worth re-trying to avoid the added hassle of `conda`.)

# ToyDown

This needs to be expanded.

Basic instructions:
- Run `python3 hh_to_person_microdata.py [name]`.
- Run `python3 run_toydown.py [name] 1 1 1 equal` (or with other parameters).

# How it works
To be written
