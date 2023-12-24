#!/bin/bash
#SBATCH -c 1                # Number of cores (-c)
#SBATCH -t 0-0:10          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p sched_mit_sloan_batch # Partition to submit to
#SBATCH -o out_files/ids.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e out_files/ids.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mem=1000           # Memory pool for all cores (see also --mem-per-cpu)
if [ "$#" -eq 2 ]; then
    PARAM_FILE="$1"
    TASK_NAME=$2
else
    echo Missing arguments PARAM_FILE or TASK_NAME
    exit 1
fi
module load sloan/python/modules/python-3.6/gurobipy/9.0.1
python3 sample_identifiers.py --from_params "$PARAM_FILE" --task_name $TASK_NAME
