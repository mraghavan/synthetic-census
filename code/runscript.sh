#!/bin/bash
#SBATCH -c 4                # Number of cores (-c)
#SBATCH -t 0-3:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=5000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o census.%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e census.%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --array=1-5
#SBATCH --mail-type=END
# Make sure the array end is the same as the number passed to partition_blocks
module load python/3.8.5-fasrc01
module load gurobi/9.0.2-fasrc01
python3 -m pip install gurobipy
python3 partition_blocks.py $SLURM_ARRAY_TASK_ID 5
