#/bin/bash
if [ "$#" -eq 2 ]; then
    PARAM_FILE="$1"
    TASK_NAME="$2"
else
    echo Missing arguments PARAM_FILE or TASK_NAME
    exit 1
fi
PART_ID=$(sbatch --parsable ./shard_generation.sh "$PARAM_FILE" "$TASK_NAME") || exit 2
echo "Partition job: $PART_ID"
SAMP_ID=$(sbatch --parsable --dependency=afterok:$PART_ID ./shard_aggregation.sh "$PARAM_FILE" "$TASK_NAME") || exit 3
echo "Sample job: $SAMP_ID"
