from census_utils import *
import os
import pickle
import numpy as np
from collections import Counter, OrderedDict
from knapsack_utils import normalize
import sys
from config2 import *
import pandas as pd
import matplotlib.pyplot as plt
from get_synthetic_stats import add_tex_var, print_all_tex_vars

parser_builder = ParserBuilder(
        {'state': True,
         'synthetic_output_dir': False,
         'num_sols': True,
         'task_name': False,
         })

def evaluate_coverage(task_name, args):
    funcs = OrderedDict({
            'total_solutions': lambda pl, ml: len(pl),
            'last_mass': lambda pl, ml: pl[-1],
            'last_50_mass': lambda pl, ml: sum(pl[-50:]),
            'coverage_at_50': lambda pl, ml: sum(pl[:50]),
            'coverage_at_500': lambda pl, ml: sum(pl[:500]),
            'sorted_coverage_at_500': lambda pl, ml: sum(sorted(pl, reverse=True)[:500]),
            })
    # for i in np.linspace(50, args.num_sols, 50):
        # funcs['coverage_at_' + str(int(i))] = lambda pl, ml: sum(pl[:i])
    d = args.synthetic_output_dir
    big_results_df = pd.DataFrame(columns=list(funcs.keys()))
    unsorted_probs = np.zeros((args.num_sols,))
    sorted_probs = np.zeros((args.num_sols,))
    for fname in os.listdir(d):
        # TODO: remove the 1 later
        if re.match(task_name + '[0-9]+_[0-9]+.pkl', fname):
            print('Reading from', d+fname, file=sys.stderr)
            with open(d + fname, 'rb') as f:
                result_list = pickle.load(f)
                # Dataframe with dimensions (len(funcs), len(result_list)) with columns given by the keys in funcs filled with zeros
                results_df = pd.DataFrame(np.zeros((len(result_list), len(funcs))), columns=list(funcs.keys()))
                # print(results_df.shape, file=sys.stderr)
                for i, results in enumerate(result_list):
                    results_df.loc[i] = [f(results['prob_list'], args.num_sols) for f in funcs.values()]
                    if len(results['prob_list']) == args.num_sols:
                        unsorted_probs += results['prob_list']
                        sorted_probs += sorted(results['prob_list'], reverse=True)
            big_results_df = pd.concat([big_results_df, results_df], ignore_index=True)
    return big_results_df, unsorted_probs, sorted_probs

def fit_pl(solution_counts):
    solution_counts = sorted(solution_counts)
    # create a list of tuples (x, y) where x is the number of solutions and y is the number of blocks with at least that many solutions
    # TODO



def to_cumulative(probs):
    new_probs = [0] * len(probs)
    new_probs[0] = probs[0]
    for i in range(1, len(probs)):
        new_probs[i] = new_probs[i-1] + probs[i]
    return new_probs

STATE = ''
if __name__ == '__main__':
    parser_builder.parse_args()
    args = parser_builder.args
    STATE = args.state.upper()
    parser_builder.verify_required_args()
    task_name = args.task_name
    if task_name != '':
        task_name += '_'
    results_df, unsorted_probs, sorted_probs = evaluate_coverage(task_name, args)
    # print(results_df, file=sys.stderr)
    print(results_df.describe(), file=sys.stderr)
    solution_counts = results_df['total_solutions'].values
    fit_pl(solution_counts)
    # weights = np.ones(len(results_df)) / len(results_df)
    results_df['total_solutions'].hist()
    plt.xlabel('Number of solutions')
    plt.savefig(task_name + 'total_solutions.png')
    plt.clf()
    # Count the number of rows where total_solutions < num_sols

    add_tex_var('NumSolutions', args.num_sols)
    num_blocks = results_df.shape[0]

    add_tex_var('NumBlocks', num_blocks)
    fully_solved = int((results_df['total_solutions'] < args.num_sols).sum())
    add_tex_var('Coverage', fully_solved)
    add_tex_var('CoveragePercent', fully_solved / num_blocks * 100, precision=2)

    avg_last_mass = results_df[results_df['total_solutions'] == args.num_sols]['last_50_mass'].mean()
    add_tex_var('AvgLastMass', avg_last_mass*100, precision=2)
    unfinished_df = results_df[results_df['total_solutions'] == args.num_sols]
    diff = unfinished_df['sorted_coverage_at_500'] - unfinished_df['coverage_at_500']
    add_tex_var('AvgDiff', diff.mean()*100, precision=2)
    unsorted_probs = to_cumulative(unsorted_probs/len(unfinished_df))
    sorted_probs = to_cumulative(sorted_probs/len(unfinished_df))
    plt.plot(unsorted_probs, label='Heuristic')
    plt.plot(sorted_probs, label='Optimal')
    plt.xlabel('Number of solutions')
    plt.ylabel('Cumulative probability')
    plt.savefig(task_name + 'cumulative_probs.png')
    
    print_all_tex_vars(STATE)
