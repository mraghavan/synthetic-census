#!/bin/bash
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 0-4:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=100000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o out_files/swap.%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e out_files/swap.%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
module load Anaconda3/2020.11
source activate mragh_test
conda run python3 swapping.py
