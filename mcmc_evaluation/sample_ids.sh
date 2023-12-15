#!/bin/bash
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 0-0:10          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p sched_mit_sloan_batch # Partition to submit to
#SBATCH -o out_files/ids.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e out_files/ids.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mem=1000           # Memory pool for all cores (see also --mem-per-cpu)
[ "$#" -eq 1 ] || { echo "No task name given" ; exit 1 ; }
module load python/3.8.5-fasrc01
python3 sample_identifiers.py --from_params ../AL_params.json --task_name $1
