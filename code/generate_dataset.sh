#/bin/bash
[ "$#" -eq 1 ] || { echo "No argument given" ; exit 1 ; }
PART_ID=$(sbatch --parsable runscript.sh $1) || exit 2
echo "Partition job: $PART_ID"
SAMP_ID=$(sbatch --parsable --dependency=afterok:$PART_ID sample_script.sh $1) || exit 3
echo "Sample job: $SAMP_ID"
SWAP_ID=$(sbatch --parsable --dependency=afterok:$SAMP_ID swap_script.sh $1) || exit 4
echo "Swap job: $SWAP_ID"
