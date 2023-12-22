#!/bin/bash
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 0-0:15          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p sched_mit_sloan_batch # Partition to submit to
#SBATCH --mem=1000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o out_files/samp.%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e out_files/samp.%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
if [ "$#" -eq 2 ]; then
    PARAM_FILE="$1"
    TASK_NAME=$2
else
    echo Missing arguments PARAM_FILE or TASK_NAME
    exit 1
fi
module load sloan/python/modules/python-3.6/gurobipy/9.0.1
echo Reading from job $1
python3 aggregate_data_shards --from_params "$1" --task_name $2
