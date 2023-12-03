import sys
import os
import numpy as np
import pickle
import re
import pandas as pd
import networkx as nx
sys.path.append('../')
from syn_census.utils.config2 import ParserBuilder
from syn_census.synthetic_data_generation.mcmc_sampler import get_log_prob
from syn_census.preprocessing.build_micro_dist import read_microdata
from syn_census.utils.encoding import encode_hh_dist, encode_row
from syn_census.utils.ip_distribution import ip_solve
from syn_census.utils.knapsack_utils import tup_sum, tup_minus, logsumexp, exp_normalize
from get_graph_stats import is_simple

parser_builder = ParserBuilder(
        {'state': True,
         'micro_file': True,
         'block_clean_file': True,
         'mcmc_output_dir': True,
         'num_sols': False,
         })

SIMPLE_PATTERN = r'simple_(\d+(\.\d+)?)'
K_PATTERN = r'k_(\d+)'

def get_conductance_ub(G: nx.DiGraph, dist: dict, sol_map: dict, gamma: float, counts: tuple):
    print('Getting conductance')
    reverse_sol_map = {v: k for k, v in sol_map.items()}
    sol_num_to_distance = {}
    for sol, sol_num in sol_map.items():
        if sol != ():
            sol_num_to_distance[sol_num] = sum(tup_minus(counts, tup_sum(sol)))
        else:
            sol_num_to_distance[sol_num] = sum(counts)
    print('All distances', set(sol_num_to_distance.values()))
    log_pi = {}
    log_pi_list_by_layer = {}
    for i, sol_num in enumerate(G.nodes()):
        sol = reverse_sol_map[sol_num]
        log_pi[sol_num] = get_log_prob(sol, dist) - gamma * sol_num_to_distance[sol_num]
        # print(tuple(str(s) for s in sol), log_pi[sol_num])
        if sol_num_to_distance[sol_num] not in log_pi_list_by_layer:
            log_pi_list_by_layer[sol_num_to_distance[sol_num]] = []
        log_pi_list_by_layer[sol_num_to_distance[sol_num]].append(log_pi[sol_num])
    log_normalizing_constant = logsumexp(list(log_pi.values()))
    normalized_log_pi = {sol_num: log_pi[sol_num] - log_normalizing_constant for sol_num in log_pi}
    log_pi_by_layer = {l: logsumexp(log_pi_list_by_layer[l]) for l in log_pi_list_by_layer}
    # print(log_pi_list_by_layer)
    print('Before normalization', log_pi_by_layer)
    pi_by_layer = exp_normalize(log_pi_by_layer)
    print('After', pi_by_layer)
    all_layers = sorted(list(pi_by_layer.keys()))
    print('All layers', all_layers)
    phis = []
    for l in all_layers[:-1]:
        phi = get_phi(G, l, pi_by_layer, normalized_log_pi, sol_num_to_distance)
        print('Layer', l, 'phi', phi)
        if not np.isinf(phi):
            phis.append(phi)
    return min(phis)

def get_phi(G: nx.DiGraph, l: int, pi_by_layer: dict, normalized_log_pi: dict, sol_num_to_distance: dict):
    print('Getting phi for layer', l)
    S_l = [sol_num for sol_num in G.nodes() if sol_num_to_distance[sol_num] <= l]
    S_l_complement = set(sol_num for sol_num in G.nodes() if sol_num_to_distance[sol_num] > l)
    pi_S = sum(pi_by_layer[l_prime] for l_prime in pi_by_layer if l_prime <= l)
    Q = 0
    for sol_num in S_l:
        crossing_prob = 0
        for _, neighbor in G.edges(sol_num): #type: ignore
            if neighbor in S_l_complement:
                crossing_prob += G.edges[sol_num, neighbor]['weight']
        Q += np.exp(normalized_log_pi[sol_num]) * crossing_prob
    return Q/(pi_S * (1-pi_S))

if __name__ == '__main__':
    parser_builder.parse_args()
    print(parser_builder.args)
    args = parser_builder.args
    #TODO some analysis of expansion of solution set
    graphs = {}
    sol_maps = {}

    fname = '{}_graph.pkl'
    # get all files matching the pattern
    formats = [fname.format(SIMPLE_PATTERN), fname.format(K_PATTERN)]
    matching_files = []
    for form in formats:
        matching_files += [f for f in os.listdir(args.mcmc_output_dir) if re.match(form, f)]
    print(matching_files)
    df = pd.read_csv(args.block_clean_file)
    # non-empty rows
    df = df[df['H7X001'] > 0]
    print(df.head())
    dist = encode_hh_dist(read_microdata(args.micro_file))
    test_row = 3
    row = df.iloc[test_row]
    counts = encode_row(row)

    for file in matching_files:
        with open(os.path.join(args.mcmc_output_dir, file), 'rb') as f:
            results = {}
            analyses_to_run = None
            param = 0
            if is_simple(file):
                m = re.match(SIMPLE_PATTERN, file)
                assert m is not None
                param = float(m.group(1)) # this is gamma
                if param != 1.0:
                    continue
                print(f'Loading {file}')
                g, sol_map = pickle.load(f)
                graphs[file] = nx.DiGraph(g)
                sol_maps[file] = sol_map
                print('conductance', get_conductance_ub(graphs[file], dist, sol_map, param, counts))
                break
