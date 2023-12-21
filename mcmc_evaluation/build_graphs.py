import sys
import os
import pickle
import lzma
import pandas as pd
# import numpy as np
import re
sys.path.append('../')
from syn_census.utils.config2 import ParserBuilder
from syn_census.mcmc_analysis.build_mcmc_graphs import build_graph_reduced, build_graph_simple, IncompleteError, build_graph_gibbs
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
         'num_sols': True,
         'task': False,
         'num_tasks': False,
         })

def make_gibbs_graphs(row: pd.Series, dist: dict, gammas: list, max_sols=0):
    counts = encode_row(row)
    gibbs_dist = {k: v for k, v in dist.items() if is_eligible(k, counts)}
    # should time-bound this
    graphs, sol_map = build_graph_gibbs(gibbs_dist, counts, gammas, total_solutions=max_sols)
    return {gamma: (graphs[gamma], sol_map) for gamma in gammas}

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
        reduced_graphs[k] = build_graph_reduced(dist, sol, sol_map, k=k)
    return reduced_graphs

def all_files_exist(file_dir: str, template: str, identifier: str, params: list):
    for param in params:
        if not os.path.exists(file_dir + template.format(identifier=identifier, param=param)):
            return False
    return True

def get_ids_from_file(fname: str, task: int, num_tasks: int):
    with open(fname, 'r') as f:
        full_random_sample_of_block_ids = f.read().splitlines()
    # get the IDs for this task
    step_size = len(full_random_sample_of_block_ids) // num_tasks
    random_sample_of_block_ids = set(full_random_sample_of_block_ids[(task-1) * step_size: task * step_size])
    return random_sample_of_block_ids


def save_graphs(graphs: dict, fname_template: str, identifier: str, params: list, file_dir: str):
    for param, graph in graphs.items():
        fname = os.path.join(file_dir, fname_template.format(identifier=identifier, param=param))
        with lzma.open(fname, 'wb') as f:
            print('Writing to', fname)
            pickle.dump(graph, f)

if __name__ == '__main__':
    parser_builder.parse_args()
    print(parser_builder.args)
    args = parser_builder.args
    print(f'Task {args.task} of {args.num_tasks}')

    in_file = os.path.join(args.mcmc_output_dir, 'sampled_block_ids.txt')
    random_sample_of_block_ids = get_ids_from_file(in_file, args.task, args.num_tasks)


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
    gibbs_fname_template = '{identifier}_{param}_gibbs_graph.xz'
    reduced_fname_template = '{identifier}_{param}_reduced_graph.xz'
    failure_file = os.path.join(args.mcmc_output_dir, f'failures{args.task}.txt')
    failures = []

    # TODO consider adding computation time to the outputs
    for i, row in matching_df.iterrows():
        print('Processing', row['identifier'])

        if not all_files_exist(args.mcmc_output_dir, gibbs_fname_template, row['identifier'], gammas):
            sol = ip_solve(encode_row(row), dist, num_solutions=args.num_sols)
            num_sols = len(sol)
            try:
                gibbs_graphs = make_gibbs_graphs(row, dist, gammas, max_sols=num_sols)
            except IncompleteError as e:
                print('IncompleteError:', e)
                failures.append(row['identifier'])
                continue
            save_graphs(gibbs_graphs, gibbs_fname_template, row['identifier'], gammas, args.mcmc_output_dir)
        else:
            print('Gibbs graphs already exist for', row['identifier'])

        # if not all_files_exist(args.mcmc_output_dir, simple_fname_template, row['identifier'], gammas):
            # sol = ip_solve(encode_row(row), dist, num_solutions=args.num_sols)
            # num_sols = len(sol)
            # try:
                # simple_graphs = make_simple_graphs(row, dist, gammas, max_sols=num_sols)
            # except IncompleteError as e:
                # print('IncompleteError:', e)
                # failures.append(row['identifier'])
                # continue
            # save_graphs(simple_graphs, simple_fname_template, row['identifier'], gammas, args.mcmc_output_dir)
        # else:
            # print('Simple graphs already exist for', row['identifier'])

        if not all_files_exist(args.mcmc_output_dir, reduced_fname_template, row['identifier'], ks):
            sol = ip_solve(encode_row(row), dist, num_solutions=args.num_sols)
            sol_map = {v: i for i, v in enumerate(sol)}
            reduced_graphs = make_reduced_graphs(dist, sol, sol_map, ks)
            save_graphs(reduced_graphs, reduced_fname_template, row['identifier'], ks, args.mcmc_output_dir)
        else:
            print('Reduced graphs already exist for', row['identifier'])
    with open(failure_file, 'a') as f:
        for failure in failures:
            f.write(failure + '\n')

    # filter_larger = {k: v for k, v in sol_counts.items() if v > max_sols}
    # big_sample_of_block_ids = set(np.random.choice(list(filter_larger.keys()), size=big_sample_size, replace=False))
    # print(big_sample_of_block_ids)
    # big_matching_df = df[df['identifier'].isin(big_sample_of_block_ids)]
    # print('Number of matching rows:', len(big_matching_df))
    # for i, row in big_matching_df.iterrows():
        # print('Processing', row['identifier'])
        # if row['identifier'] in prev_failures:
            # print('Skipping', row['identifier'], 'because it failed previously')
            # continue
        # if not all_files_exist(args.mcmc_output_dir, reduced_fname_template, row['identifier'], ks):
            # sol = ip_solve(encode_row(row), dist, num_solutions=args.num_sols)
            # sol_map = {v: i for i, v in enumerate(sol)}
            # reduced_graphs = make_reduced_graphs(dist, sol, sol_map, ks)
            # for k, graph in reduced_graphs.items():
                # fname = os.path.join(args.mcmc_output_dir, reduced_fname_template.format(identifier=row['identifier'], param=k))
                # with lzma.open(fname, 'wb') as f:
                    # print('Writing to', fname)
                    # pickle.dump(graph, f)
        # else:
            # print('Reduced graphs already exist for', row['identifier'])
