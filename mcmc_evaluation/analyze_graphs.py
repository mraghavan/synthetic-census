import sys
import os
import pickle
import lzma
import re
import pandas as pd
import networkx as nx
sys.path.append('../')
from syn_census.utils.config2 import ParserBuilder
from syn_census.synthetic_data_generation.mcmc_sampler import SimpleMCMCSampler
from syn_census.preprocessing.build_micro_dist import read_microdata
from syn_census.utils.encoding import encode_hh_dist, encode_row
from syn_census.utils.ip_distribution import ip_solve
from syn_census.utils.knapsack_utils import is_eligible
from syn_census.mcmc_analysis.analyze_mcmc_graphs import is_connected, get_mixing_time_bounds, get_solution_density, get_conductance_ub

parser_builder = ParserBuilder(
        {'state': True,
         'micro_file': True,
         'block_clean_file': True,
         'mcmc_output_dir': True,
         'num_sols': False,
         })

simple_re = re.compile(r'(\d+-\d+-\d+)_(\d+(.\d+)?)_simple_graph.xz')
reduced_re = re.compile(r'(\d+-\d+-\d+)_(\d+)_reduced_graph.xz')

def get_all_matching_files(results_dir: str):
    simple_files = []
    reduced_files = []
    for fname in os.listdir(results_dir):
        if simple_re.match(fname):
            simple_files.append(fname)
        elif reduced_re.match(fname):
            reduced_files.append(fname)
    return sorted(simple_files), sorted(reduced_files)

def get_simple_params(fname: str):
    match = simple_re.match(fname)
    assert match is not None
    return match.group(1), float(match.group(2))

def get_reduced_params(fname: str):
    match = reduced_re.match(fname)
    assert match is not None
    return match.group(1), int(match.group(2))

def do_common_analyses(G: nx.DiGraph):
    results = {}
    results['mixing_time_bounds'] = get_mixing_time_bounds(G, 1/len(G))
    results['num_states'] = len(G)
    return results

def do_simple_analyses(graph: dict, sol_map: dict, dist: dict, counts: tuple, gamma: float):
    results = {}
    G = nx.DiGraph(graph)
    reverse_sol_map = {v: k for k, v in sol_map.items()}
    results['solution_density'], results['num_solutions'] = get_solution_density(G, gamma, dist, reverse_sol_map, counts)
    results.update(do_common_analyses(G))
    return results

def do_reduced_analyses(graph: dict):
    results = {}
    G = nx.DiGraph(graph)
    results['is_connected'] = is_connected(G)
    results['num_solutions'] = len(G)
    results.update(do_common_analyses(G))
    return results

if __name__ == '__main__':
    parser_builder.parse_args()
    print(parser_builder.args)
    args = parser_builder.args

    df = pd.read_csv(args.block_clean_file)
    # non-empty rows
    df = df[df['H7X001'] > 0]
    print(df.head())
    dist = encode_hh_dist(read_microdata(args.micro_file))

    simple_files, reduced_files = get_all_matching_files(args.mcmc_output_dir)
    print(f'Found {len(simple_files)} simple files and {len(reduced_files)} reduced files')

    simple_results_template = '{identifier}_{param}_simple_results.pkl'
    reduced_results_template = '{identifier}_{param}_reduced_results.pkl'

    for simple_fname in simple_files:
        identifier, param = get_simple_params(simple_fname)
        results_fname = os.path.join(
                args.mcmc_output_dir,
                simple_results_template.format(identifier=identifier, param=param)
                )
        if os.path.exists(results_fname):
            print(f'Simple results for {identifier} with gamma={param} already exist')
            continue
        print(f'Analyzing simple {identifier} with gamma={param}')
        counts = encode_row(df[df['identifier'] == identifier].iloc[0])
        simple_dist = {k: v for k, v in dist.items() if is_eligible(k, counts)}
        gamma = float(param)
        graph, sol_map = None, None
        with lzma.open(os.path.join(args.mcmc_output_dir, simple_fname), 'rb') as f:
            graph, sol_map = pickle.load(f)
        print(f'Graph has {len(graph)} nodes')
        results = do_simple_analyses(graph, sol_map, simple_dist, counts, gamma)
        with open(results_fname, 'wb') as f:
            print(f'Writing results to {results_fname}')
            pickle.dump(results, f)

    for reduced_fname in reduced_files:
        identifier, param = get_reduced_params(reduced_fname)
        results_fname = os.path.join(
                args.mcmc_output_dir,
                reduced_results_template.format(identifier=identifier, param=param)
                )
        if os.path.exists(results_fname):
            print(f'Reduced results for {identifier} with k={param} already exist')
            continue
        print(f'Analyzing reduced {identifier} with k={param}')
        k = int(param)
        graph, sol_map = None, None
        with lzma.open(os.path.join(args.mcmc_output_dir, reduced_fname), 'rb') as f:
            graph, sol_map = pickle.load(f)
        print(f'Graph has {len(graph)} nodes')
        results = do_reduced_analyses(graph)
        with open(results_fname, 'wb') as f:
            print(f'Writing results to {results_fname}')
            pickle.dump(results, f)
