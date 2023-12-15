#/bin/bash
[ "$#" -eq 1 ] || { echo "No task name given" ; exit 1 ; }
SAMP_ID=$(sbatch --parsable ./sample_ids.sh $1) || exit 2
echo "ID sampling job: $SAMP_ID"
BUILD_ID=$(sbatch --parsable --dependency=afterok:$SAMP_ID ./graph_generation.sh $1) || exit 3
echo "Graph building job: $BUILD_ID"
