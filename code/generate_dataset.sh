#/bin/bash
JOB_ID=$(sbatch --parsable runscript.sh) || exit 1
echo $JOB_ID
sbatch --dependency=afterok:$JOB_ID sample_script.sh $JOB_ID
