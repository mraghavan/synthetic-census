from guided_solver import *
from census_utils import *
from config2 import *
import pandas as pd
from build_micro_dist import read_microdata
import pickle
import sys
# import psutil
parser_builder = ParserBuilder(
        {'state': True,
         'micro_file': True,
         'block_clean_file': True,
         'synthetic_output_dir': False,
         'num_sols': False,
         'task': False,
         'num_tasks': False,
         'task_name': False,
         })

def read_block_data(block_clean_file):
    return pd.read_csv(block_clean_file)

def sample_from_sol(sol):
    keys, probs = zip(*sol.items())
    keys = list(keys)
    if len(keys) > 1:
        return keys[np.random.choice(range(len(keys)), p=probs)]
    else:
        return keys[0]

if __name__ == '__main__':
    parser_builder.parse_args()
    print(parser_builder.args)
    parser_builder.verify_required_args()
    args = parser_builder.args
    task = args.task
    num_tasks = args.num_tasks
    if args.task_name != '':
        task_name = args.task_name + '_'
    else:
        task_name = ''
    if args.synthetic_output_dir != '':
        out_file = args.synthetic_output_dir + task_name + '%d_%d.pkl' % (task, num_tasks)
        if os.path.exists(out_file):
            print(out_file, 'already exists')
            sys.exit(0)
    else:
        out_file = ''
    print(args)
    SOLVER_PARAMS.num_sols = args.num_sols

    df = read_block_data(args.block_clean_file)
    # non-empty rows
    df = df[df['H7X001'] > 0]
    n = len(df)
    first_ind = int((task-1) / num_tasks * n)
    last_ind = int(task/num_tasks * n)
    print('Processing indices', first_ind, 'through', last_ind)
    df = df.iloc[first_ind:last_ind+1]
    print(len(df), 'blocks to process')
    print(df.head())
    hh_dist = encode_hh_dist(read_microdata(args.micro_file))
    errors = []
    output = []
    for ind, row in df.iterrows():
        if ind > 500:
            break
        print()
        print('index', ind, 'id', row['identifier'])
        # print('Current memory usage', psutil.Process().memory_info().rss / (1024 * 1024), 'MB')
        identifier = str(row['identifier'])
        sol = solve(row, hh_dist)
        print(len(sol), 'unique solutions')
        # print(sol.values())
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
            if out_file != '':
                output.append({
                        'id': identifier,
                        'sol': chosen,
                        'level': SOLVER_RESULTS.level,
                        'complete': SOLVER_RESULTS.status == SolverResults.OK,
                        'age': SOLVER_RESULTS.use_age,
                        'types': chosen_types,
                        'prob_list': list(sol.values())
                        })
        print('errors', errors, file=sys.stderr)
    if out_file != '':
        print('Writing to', out_file)
        with open(out_file, 'wb') as f:
            pickle.dump(output, f)
    print(len(errors), 'errors')
