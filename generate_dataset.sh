#/bin/bash
[ "$#" -eq 1 ] || { echo "No task name given" ; exit 1 ; }
PART_ID=$(sbatch --parsable ./shard_generation.sh $1) || exit 2
echo "Partition job: $PART_ID"
SAMP_ID=$(sbatch --parsable --dependency=afterok:$PART_ID ./shard_aggregation.sh $1) || exit 3
echo "Sample job: $SAMP_ID"
