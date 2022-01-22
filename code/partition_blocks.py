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
    if len(sys.argv) >= 3:
        task = int(sys.argv[1])
        num_tasks = int(sys.argv[2])
    else:
        task = 1
        num_tasks = 1
    print_config()
    SOLVER_PARAMS.num_sols = NUM_SOLS

    df = read_block_data()
    # non-empty rows
    df = df[df['H7X001'] > 0]
    n = len(df)
    first_ind = int((task-1) / num_tasks * n)
    last_ind = int(task/num_tasks * n)
    print('Processing indices', first_ind, 'through', last_ind)
    df = df.iloc[first_ind:last_ind+1]
    print(len(df), 'blocks to process')
    print(df.head())
    hh_dist = encode_hh_dist(read_microdata(get_micro_file()))
    # print(hh_dist.most_common(10))
    errors = []
    for ind, row in df.iterrows():
        print()
        print('index', ind)
        # print('Current memory usage', psutil.Process().memory_info().rss / (1024 * 1024), 'MB')
        identifier = str(row['identifier'])
        id_file = get_dist_dir() + identifier + '.pkl'
        if os.path.exists(id_file):
            print(id_file, 'already exists')
            continue
        sol = solve(row, hh_dist)
        print(len(sol), 'unique solutions')
        if SOLVER_RESULTS.status == SolverResults.UNSOLVED:
            print(ind, SOLVER_RESULTS.status, file=sys.stderr)
            errors.append(ind)
        print('SOLVER LEVEL', SOLVER_RESULTS.level, 'USED AGE', SOLVER_RESULTS.use_age, 'STATUS', SOLVER_RESULTS.status)
        if SOLVER_RESULTS.level > 2:
            break
        if len(sol) > 0:
            if WRITE:
                with open(id_file, 'wb') as f:
                    pickle.dump({
                        'id': identifier,
                        'sol': sol,
                        'level': SOLVER_RESULTS.level,
                        'complete': SOLVER_RESULTS.status == SolverResults.OK,
                        'age': SOLVER_RESULTS.use_age,
                        },
                        f)
        print('errors', errors, file=sys.stderr)
    print(len(errors), 'errors')
