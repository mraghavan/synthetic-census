import pandas as pd
import pickle as pkl
import sys
import numpy as np
from .guided_solver import SOLVER_PARAMS, SOLVER_RESULTS, SolverResults, solve
from ..utils.encoding import encode_hh_dist
from ..utils.census_utils import *
from ..preprocessing.build_micro_dist import read_microdata

def read_block_data(block_clean_file: str):
    return pd.read_csv(block_clean_file)

def sample_from_sol(sol):
    keys, probs = zip(*sol.items())
    keys = list(keys)
    if len(keys) > 1:
        return keys[np.random.choice(range(len(keys)), p=probs)]
    else:
        return keys[0]

def generate_data(
        micro_file: str,
        block_clean_file: str,
        num_sols: int,
        task: int,
        num_tasks: int,
        include_probs: bool = False,
        tmp_file: str= '',
        tmp_file_template: str = '',
        ):
    SOLVER_PARAMS.num_sols = num_sols

    df = read_block_data(block_clean_file)
    # non-empty rows
    df = df[df['H7X001'] > 0]
    n = len(df)
    first_ind = int((task-1) / num_tasks * n)
    last_ind = int(task/num_tasks * n)
    print('Processing indices', first_ind, 'through', last_ind)
    df = df.iloc[first_ind:last_ind+1]
    print(len(df), 'blocks to process')
    print(df.head())
    hh_dist = encode_hh_dist(read_microdata(micro_file))
    errors = []
    output = []
    if tmp_file:
        print('Loading tmp file', tmp_file)
        with open(tmp_file, 'rb') as f:
            output, errors = pkl.load(f)
    already_finished = set([o['id'] for o in output])

    for i, (ind, row) in enumerate(df.iterrows()):
        print()
        print('index', ind, 'id', row['identifier'])
        if row['identifier'] in already_finished:
            print(row['identifier'], 'already finished')
            continue
        identifier = str(row['identifier'])
        sol = solve(row, hh_dist)
        print(len(sol), 'unique solutions')
        chosen = sample_from_sol(sol)
        chosen = tuple(hh.to_sol() for hh in chosen)
        if hasattr(chosen[0], 'get_type'):
            chosen_types = tuple(c.get_type() for c in chosen)
            print(chosen_types)
        else:
            chosen_types = None
        if SOLVER_RESULTS.status == SolverResults.UNSOLVED:
            print(ind, SOLVER_RESULTS.status, file=sys.stderr)
            errors.append(ind)
        print('SOLVER LEVEL', SOLVER_RESULTS.level, 'USED AGE', SOLVER_RESULTS.use_age, 'STATUS', SOLVER_RESULTS.status)
        if len(sol) > 0:
            d = {
                    'id': identifier,
                    'sol': chosen,
                    'level': SOLVER_RESULTS.level,
                    'complete': SOLVER_RESULTS.status == SolverResults.OK,
                    'age': SOLVER_RESULTS.use_age,
                    'types': chosen_types,
                    }
            if include_probs:
                d['prob_list'] = list(sol.values())
            output.append(d)
            if i > 0 and i % 30 == 0 and tmp_file_template:
                print('Saving tmp file', tmp_file_template.format(i))
                with open(tmp_file_template.format(i), 'wb') as f:
                    pkl.dump((output, errors), f)
    return (output, errors)
