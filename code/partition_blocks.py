from guided_solver import *
from census_utils import *
from config import *
import pandas as pd
from build_micro_dist import read_microdata
import pickle
import sys
import psutil
import gc

def read_block_data():
    return pd.read_csv(get_block_out_file())

if __name__ == '__main__':
    print_config()
    SOLVER_PARAMS.num_sols = NUM_SOLS

    df = read_block_data()
    print(df.head())
    # non-empty rows
    df = df[df['H7X001'] > 0]
    print(df.head())
    print(len(df), 'blocks to process')
    all_dists, fallback_dist = read_microdata(get_micro_file())
    errors = []
    for ind, row in df.iterrows():
        print()
        print('index', ind)
        print('Current memory usage', psutil.Process().memory_info().rss / (1024 * 1024), 'MB')
        identifier = str(row['identifier'])
        id_file = get_dist_dir() + identifier + '.pkl'
        if os.path.exists(id_file):
            print(id_file, 'already exists')
            continue
        sol = solve(row, all_dists, fallback_dist)
        print(len(sol), 'unique solutions')
        if SOLVER_RESULTS.status in (SolverResults.BAD_DIST, SolverResults.BAD_COUNTS):
            print(ind, SOLVER_RESULTS.status, file=sys.stderr)
            errors.append(ind)
        print('SOLVER LEVEL', SOLVER_RESULTS.level, 'USED AGE', SOLVER_RESULTS.use_age, 'STATUS', SOLVER_RESULTS.status)
        if len(sol) > 0:
            if WRITE:
                with open(id_file, 'wb') as f:
                    pickle.dump({identifier: sol}, f)
        print('errors', errors, file=sys.stderr)
    print(len(errors), 'errors')
