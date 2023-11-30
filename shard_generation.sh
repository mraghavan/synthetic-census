#!/bin/bash
#SBATCH -c 4                # Number of cores (-c)
#SBATCH -t 0-0:15          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue   # Partition to submit to
#SBATCH --mem=5000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o out_files/census.%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e out_files/census.%A_%a.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --array=1-50
#SBATCH --mail-type=END
if [ "$#" -eq 1 ]; then
    echo "No task name given. Echo defaulting to job id"
    TASK_NAME=$SLURM_ARRAY_JOB_ID
else
    TASK_NAME=$1
fi
module load python/3.8.5-fasrc01
module load gurobi/9.0.2-fasrc01
python3 -m pip install gurobipy
python3 generate_data_shard.py --from_params AL_params.json --task $SLURM_ARRAY_TASK_ID --num_tasks $SLURM_ARRAY_TASK_COUNT --task_name $TASK_NAME
