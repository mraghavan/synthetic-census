#!/bin/bash

python3 test_sampling.py --from_params ../NV_params.json --task_name NV_eval
python3 test_sampling.py --from_params ../AL_params.json --task_name AL_final
\cp img/NV* ~/Dropbox/research/census-knapsack/forc/img
\cp img/AL* ~/Dropbox/research/census-knapsack/forc/img
