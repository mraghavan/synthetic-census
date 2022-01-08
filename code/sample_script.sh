#!/bin/bash
#SBATCH -c 4                # Number of cores (-c)
#SBATCH -t 0-1:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=5000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o out_files/samp.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e out_files/samp.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
module load python/3.8.5-fasrc01
python3 sample_from_dist.py
