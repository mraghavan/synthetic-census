#!/bin/bash
#SBATCH -c 4                # Number of cores (-c)
#SBATCH -t 0-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p sched_mit_sloan_batch   # Partition to submit to
#SBATCH --mem=10000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o out_files/build.%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e out_files/build.%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END

if [ "$#" -eq 3 ]; then
    PARAM_FILE="$1"
    TASK_NUM="$2"
    TASKS="$3"
else
    echo Missing arguments PARAM_FILE TASK_NUM TASKS
    exit 1
fi

module load sloan/python/modules/python-3.6/gurobipy/9.0.1
python3 -m pip install gurobipy
python3 -m pip install --user networkx
python3 build_graphs.py --from_params "$PARAM_FILE" --task $TASK_NUM --num_tasks $TASKS
python3 analyze_graphs.py --from_params "$PARAM_FILE" --task $TASK_NUM --num_tasks $TASKS
