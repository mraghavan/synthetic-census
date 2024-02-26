#!/bin/bash

python3 process_graph_results.py --from_params ../NV_params.json
python3 process_graph_results.py --from_params ../AL_params.json
\cp img/NV* ~/Dropbox/research/census-knapsack/forc/img
\cp img/AL* ~/Dropbox/research/census-knapsack/forc/img
