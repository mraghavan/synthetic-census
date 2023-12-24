import sys
import os
import pickle
import lzma
import pandas as pd
from collections import OrderedDict
from math import ceil, floor
# import numpy as np
# import re
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

def make_gibbs_graphs(row: pd.Series, dist: dict, gammas: list, **kwargs):
    max_sols = kwargs.get('max_sols', 0)
    counts = encode_row(row)
    gibbs_dist = OrderedDict({k: v for k, v in dist.items() if is_eligible(k, counts)})
    # should time-bound this
    graphs, sol_map = build_graph_gibbs(gibbs_dist, counts, gammas, total_solutions=max_sols)
    return {gamma: (graphs[gamma], sol_map) for gamma in gammas}

def make_simple_graphs(row: pd.Series, dist: dict, gammas: list, **kwargs):
    max_sols = kwargs.get('max_sols', 0)
    counts = encode_row(row)
    simple_dist = OrderedDict({k: v for k, v in dist.items() if is_eligible(k, counts)})
    # should time-bound this
    graphs, sol_map = build_graph_simple(simple_dist, counts, gammas, total_solutions=max_sols)
    return {gamma: (graphs[gamma], sol_map) for gamma in gammas}

def make_reduced_graphs(row: pd.Series, dist: dict, ks: list, **kwargs):
    sol = kwargs.get('sol', None)
    sol_map = kwargs.get('sol_map', None)
    assert sol is not None
    assert sol_map is not None
    reduced_graphs = {}
    for k in ks:
        print('Building reduced graph for k =', k, 'with', len(sol), 'solutions')
        try:
            reduced_graphs[k] = build_graph_reduced(dist, sol, sol_map, k=k)
        except IncompleteError:
            reduced_graphs[k] = None
    return reduced_graphs

def all_files_exist(file_dir: str, template: str, identifier: str, params: list):
    for param in params:
        if not os.path.exists(file_dir + template.format(identifier=identifier, param=param)):
            # print('File', file_dir + template.format(identifier=identifier, param=param), 'does not exist')
            return False
    return True

def get_jobs_from_file(fname: str, task: int, num_tasks: int):
    with open(fname, 'r') as f:
        full_random_sample_of_block_ids = f.read().splitlines()
    # get the IDs for this task
    step_size = ceil(len(full_random_sample_of_block_ids) / num_tasks)
    random_sample_of_block_ids = full_random_sample_of_block_ids[(task-1) * step_size: task * step_size]
    return random_sample_of_block_ids


def save_graphs(graphs: dict, fname_template: str, identifier: str, file_dir: str):
    for param, graph in graphs.items():
        fname = os.path.join(file_dir, fname_template.format(identifier=identifier, param=param))
        with lzma.open(fname, 'wb') as f:
            print('Writing to', fname)
            pickle.dump(graph, f)

gammas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
ks = [2, 3, 4]
simple_fname_template = '{identifier}_{param}_simple_graph.xz'
gibbs_fname_template = '{identifier}_{param}_gibbs_graph.xz'
reduced_fname_template = '{identifier}_{param}_reduced_graph.xz'

templates = {
    'simple': simple_fname_template,
    'gibbs': gibbs_fname_template,
    'reduced': reduced_fname_template,
    }
params = {
    'simple': gammas,
    'gibbs': gammas,
    'reduced': ks,
    }

if __name__ == '__main__':
    parser_builder.parse_args()
    print(parser_builder.args)
    args = parser_builder.args
    print(f'Task {args.task} of {args.num_tasks}')

    in_file = os.path.join(args.mcmc_output_dir, 'sampled_block_ids.txt')
    jobs = get_jobs_from_file(in_file, args.task, args.num_tasks)
    print(jobs)


    df = pd.read_csv(args.block_clean_file)
    # non-empty rows
    df = df[df['H7X001'] > 0]
    print(df.head())
    dist = encode_hh_dist(read_microdata(args.micro_file))
    print('Number of jobs', len(jobs))
    graph_funcs = {
            'gibbs': make_gibbs_graphs,
            'simple': make_simple_graphs,
            'reduced': make_reduced_graphs,
            }
    # failure_file = os.path.join(args.mcmc_output_dir, f'failures{args.task}.{args.num_tasks}.txt')
    failure_file = os.path.join(args.mcmc_output_dir, f'failures{args.task}.txt')
    prev_failures = set()
    if os.path.exists(failure_file):
        with open(failure_file, 'r') as f:
            prev_failures = set(f.read().splitlines())
    failures = []
    print('failed', prev_failures)

    # TODO consider adding computation time to the outputs
    for job in jobs:
        identifier, job_type = job.split(',')
        if identifier in prev_failures:
            failures.append(identifier)
            print('Skipping', identifier, job_type)
            continue
        row = df[df['identifier'] == identifier].iloc[0]
        print('Processing', identifier, job_type)
        template = templates[job_type]
        param = params[job_type]
        if not all_files_exist(args.mcmc_output_dir, template, identifier, param):
            sol = ip_solve(encode_row(row), dist, num_solutions=args.num_sols)
            num_sols = len(sol)
            sol_map = {v: i for i, v in enumerate(sol)}
            try:
                kwargs = {
                        'sol': sol,
                        'sol_map': sol_map,
                        'max_sols': num_sols,
                        }
                graphs = graph_funcs[job_type](row, dist, param, **kwargs)
                if len(graphs) < len(param):
                    print('IncompleteError: not all graphs were built')
                    failures.append(row['identifier'])
                    continue
            except IncompleteError as e:
                print('IncompleteError:', e)
                failures.append(row['identifier'])
                continue
            save_graphs(graphs, template, identifier, args.mcmc_output_dir)
        else:
            print(job_type, 'graphs already exist for', row['identifier'])

    with open(failure_file, 'w') as f:
        print('Writing failures to', failure_file)
        for failure in failures:
            f.write(failure + '\n')
