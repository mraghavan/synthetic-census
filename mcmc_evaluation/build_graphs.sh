#!/bin/bash
#SBATCH -c 4                # Number of cores (-c)
#SBATCH -t 0-10:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p sched_mit_sloan_batch   # Partition to submit to
#SBATCH --mem=5000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o out_files/build.%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e out_files/build.%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --array=1-50
#SBATCH --mail-type=END

module load sloan/python/modules/python-3.6/gurobipy/9.0.1
python3 -m pip install gurobipy
python3 build_graphs.py --from_params ../AL_params.json --task $SLURM_ARRAY_TASK_ID --num_tasks $SLURM_ARRAY_TASK_COUNT
