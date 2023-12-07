import sys
import os
import pickle
import lzma
import pandas as pd
import numpy as np
import re
sys.path.append('../')
from syn_census.utils.config2 import ParserBuilder
from syn_census.mcmc_analysis.build_mcmc_graphs import build_graph, build_graph_simple, IncompleteError
from syn_census.preprocessing.build_micro_dist import read_microdata
from syn_census.utils.encoding import encode_hh_dist, encode_row
from syn_census.utils.ip_distribution import ip_solve
from syn_census.utils.knapsack_utils import is_eligible

parser_builder = ParserBuilder(
        {'state': True,
         'micro_file': True,
         'block_clean_file': True,
         'mcmc_output_dir': True,
         'synthetic_output_dir': False,
         'num_sols': False,
         'task_name': True,
         })

def get_relevant_blocks(results_dir: str, task_name: str, min_sols: int, max_sols: int):
    """
    Returns a list of block ids that have between min_sols and max_sols solutions
    and their corresponding solution counts.
    """
    pattern = re.compile(rf'{task_name}_\d+_\d+.pkl')
    matching_ids = []
    sol_counts = {}
    for fname in os.listdir(results_dir):
        if pattern.match(fname):
            with open(results_dir + fname, 'rb') as f:
                sols = pickle.load(f)
                for sol in sols:
                    if min_sols <= len(sol['prob_list']) <= max_sols:
                        matching_ids.append(sol['id'])
                        sol_counts[sol['id']] = len(sol['prob_list'])
    return matching_ids, sol_counts

def make_simple_graphs(row: pd.Series, dist: dict, gammas: list, max_sols=0):
    counts = encode_row(row)
    simple_dist = {k: v for k, v in dist.items() if is_eligible(k, counts)}
    # should time-bound this
    graphs, sol_map = build_graph_simple(simple_dist, counts, gammas, total_solutions=max_sols)
    return {gamma: (graphs[gamma], sol_map) for gamma in gammas}

def make_reduced_graphs(dist: dict, sol: list, sol_map: dict, ks: list):
    reduced_graphs = {}
    for k in ks:
        print('Building reduced graph for k =', k, 'with', len(sol), 'solutions')
        reduced_graphs[k] = build_graph(dist, sol, sol_map, k=k)
    return reduced_graphs

def all_files_exist(file_dir: str, template: str, identifier: str, params: list):
    for param in params:
        if not os.path.exists(file_dir + template.format(identifier=identifier, param=param)):
            return False
    return True

if __name__ == '__main__':
    # set numpy random seed
    parser_builder.parse_args()
    print(parser_builder.args)
    args = parser_builder.args
    min_sols = 30
    max_sols = 100
    sample_size = 10
    block_ids, sol_counts = get_relevant_blocks(args.synthetic_output_dir, args.task_name, min_sols, max_sols)
    print('Number of blocks with between {} and {} solutions: {}'.format(min_sols, max_sols, len(block_ids)))
    random_sample_of_block_ids = set(np.random.choice(block_ids, size=sample_size, replace=False))

    df = pd.read_csv(args.block_clean_file)
    # non-empty rows
    df = df[df['H7X001'] > 0]
    print(df.head())
    dist = encode_hh_dist(read_microdata(args.micro_file))
    matching_df = df[df['identifier'].isin(random_sample_of_block_ids)]
    print('Number of matching rows:', len(matching_df))

    gammas = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    ks = [2, 3, 4]
    simple_fname_template = '{identifier}_{param}_simple_graph.xz'
    reduced_fname_template = '{identifier}_{param}_reduced_graph.xz'
    failure_file = os.path.join(args.mcmc_output_dir, 'failures.txt')
    prev_failures = set()
    with open(failure_file, 'r') as f:
        for line in f:
            prev_failures.add(line.strip())
    failures = []

    # TODO consider adding computation time to the outputs
    for i, row in matching_df.iterrows():
        print('Processing', row['identifier'])
        if row['identifier'] in prev_failures:
            print('Skipping', row['identifier'], 'because it failed previously')
            continue
        if not all_files_exist(args.mcmc_output_dir, simple_fname_template, row['identifier'], gammas):
            try:
                simple_graphs = make_simple_graphs(row, dist, gammas, sol_counts[row['identifier']])
            except IncompleteError as e:
                print('IncompleteError:', e)
                failures.append(row['identifier'])
                continue
            for gamma, (graph, sol_map) in simple_graphs.items():
                fname = os.path.join(args.mcmc_output_dir, simple_fname_template.format(identifier=row['identifier'], param=gamma))
                with lzma.open(fname, 'wb') as f:
                    print('Writing to', fname)
                    pickle.dump((graph, sol_map), f)
        else:
            print('Simple graphs already exist for', row['identifier'])
        if not all_files_exist(args.mcmc_output_dir, reduced_fname_template, row['identifier'], ks):
            sol = ip_solve(encode_row(row), dist, num_solutions=max_sols)
            sol_map = {v: i for i, v in enumerate(sol)}
            reduced_graphs = make_reduced_graphs(dist, sol, sol_map, ks)
            for k, graph in reduced_graphs.items():
                fname = os.path.join(args.mcmc_output_dir, reduced_fname_template.format(identifier=row['identifier'], param=k))
                with lzma.open(fname, 'wb') as f:
                    print('Writing to', fname)
                    pickle.dump(graph, f)
        else:
            print('Reduced graphs already exist for', row['identifier'])
    with open(failure_file, 'a') as f:
        for failure in failures:
            f.write(failure + '\n')
