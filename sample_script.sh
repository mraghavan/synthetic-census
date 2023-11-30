#!/bin/bash
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 0-0:15          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=1000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o out_files/samp.%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e out_files/samp.%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
module load python/3.8.5-fasrc01
echo Reading from job $1
python3 aggregate_data_shards --from_params AL_params.json --task_name $1
