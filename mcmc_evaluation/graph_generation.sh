#!/bin/bash
#SBATCH -c 4                # Number of cores (-c)
#SBATCH -t 0-10:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p sched_mit_sloan_batch   # Partition to submit to
#SBATCH --mem=5000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o out_files/mcmc_graph.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e out_files/mcmc_graph.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END
if [ "$#" -eq 1 ]; then
    TASK_NAME=$1
else
    echo "No task name given"; exit 1
fi
module load sloan/python/modules/python-3.6/gurobipy/9.0.1
python3 -m pip install gurobipy
python3 build_graphs.py --from_params ../AL_params.json --task_name $TASK_NAME
