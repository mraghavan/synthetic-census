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
This will create `.pkl` file for each block in `[output]/[state]`.
To parallelize, run `python3 partition_blocks.py {i} {total}` for `i` in `1..total` on separate threads/machines (see below for `slurm` usage).

### Running on the RC cluster with `slurm`
If using `slurm`, run `sbatch runscript.sh`. `runscript.sh` should look something like
```
#!/bin/bash
#SBATCH -c 4                # Number of cores (-c)
#SBATCH -t 0-0:30          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=1000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o out_files/census.%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e out_files/census.%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --array=1-10
#SBATCH --mail-type=END
# Make sure the array end is the same as the number passed to partition_blocks
module load python/3.8.5-fasrc01
module load gurobi/9.0.2-fasrc01
python3 -m pip install gurobipy
python3 partition_blocks.py $SLURM_ARRAY_TASK_ID 10
```
Make sure the directory `code/out_files` exists before running.
The requirements are reasonably well-calibrated for Vermont, which contains 17,541 non-empty census blocks (takes roughly 15 minutes total).
I'd recommend scaling the number of parralel jobs (`#SBATCH --array=1-{num}` and `python3 partition_blocks.py $SLURM_ARRAY_TASK_ID {num}`) to be roughly 1 job for 2,000 blocks.
To prevent too many from running at the same time, use `#SBATCH --array=1-{num}%{max}` where `max` is the maximum number of jobs to run at a time.
Gurobi can use multiple cores, which is why I've set the number of cores to 4. I haven't experimented with increasing/decreasing this.

## Generating a dataset
Run `python3 sample_from_dist.py` if running locally (which may be prohibitively slow), or `sbatch sample_script.sh`, which should look like
```
#!/bin/bash
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 0-1:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=1000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o out_files/samp.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e out_files/samp.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
module load python/3.8.5-fasrc01
python3 sample_from_dist.py
```
This will write the sampled dataset in the file `[output]/[state]/synthetic.csv`, where each row is a sampled household.
This will take a fairly long time -- about 35 minutes for VT, which has ~17.5k non-empty blocks.
This suggests something like 2 minutes per thousand blocks, with some buffer to be safe.
The memory requirements may also need to be increased for larger states.
If this is too slow for larger states, consider parallelizing by:
- Sampling multiple datasets at a time.
- Sampling in parallel, and aggregating later.

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
|`BLOCK_18_PLUS`| Total number of 18+ people in that block |
|`TOTAL`| Total number of people in this household |
|`W`| Total number of people in this household of race 'White alone' |
|`B`|Total number of people in this household of race 'Black or African American alone' |
|`AI_AN`|Total number of people in this household of race 'American Indian and Alaska Native alone' |
|`AS`|Total number of people in this household of race 'Asian alone' |
|`H_PI`|Total number of people in this household of race 'Native Hawaiian and Other Pacific Islander alone' |
|`OTH`|Total number of people in this household of race 'Some Other Race alone' |
|`TWO_OR_MORE`|Total number of people in this household of race 'Two or More Races' |
|`18_PLUS`| Total number of people in this household aged 18 or more |
|`HH_NUM`| Household index within block |
|`ACCURACY`| How accurate a solver was used (1, 2, or 3; see below)|
|`AGE_ACCURACY`| Whether this block had accurate age information |
|`identifier`| Unique block ID |

`ACCURACY = 1` means the households match the NHGIS data in terms of:
- Total number of people of each race in the block (see table `P3`)
- Number of households of each (householder race, family status, size) type (see tables `P28A`--`G`)
- (if `AGE_ACCURACY` is true) Number of people age 18+ living in a household for each householder race (see tables `P16A`--`G`)

`ACCURACY = 2` means the households match the NHGIS data in terms of:
- Total number of people of each race in the block (see table `P3`)
- Total number of households (see table `P28`)
- (if `AGE_ACCURACY` is true) Total number of people 18+ (see table `P16`)

`ACCURACY = 3` means the households match the NHGIS data in terms of:
- Total number of people of each race in the block (see table `P3`)
- (if `AGE_ACCURACY` is true) Total number of people 18+ (see table `P16`)

`AGE_ACCURACY` is true if tables `P3` and `P16` agree on the total number of residents in the block, and false otherwise. In VT, this is true for ~98% of blocks.

# How it works
To be written
