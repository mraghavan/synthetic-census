import pandas as pd
import pickle
import re
import networkx as nx
import numpy as np
import scipy.sparse as sp
from collections import Counter
# from build_mcmc_graphs import get_neighbors_simple, get_d_maxes
import random
from math import ceil, floor
from ..synthetic_data_generation.mcmc_sampler import get_log_prob, MCMCSampler, SimpleMCMCSampler, get_prob_diff_sum
from ..utils.knapsack_utils import is_eligible, logsumexp, tup_sum, tup_minus, exp_normalize
from ..utils.config2 import ParserBuilder
from ..utils.encoding import encode_hh_dist, encode_row
from ..preprocessing.build_micro_dist import read_microdata
from ..utils.ip_distribution import ip_solve

parser_builder = ParserBuilder(
        {'state': True,
         'micro_file': True,
         'block_clean_file': True,
         'synthetic_output_dir': False,
         'num_sols': False,
         })

def read_block_data(block_clean_file: str):
    return pd.read_csv(block_clean_file)

def is_connected(graph: nx.DiGraph):
    seen = set()
    starting_node = next(iter(graph))
    stack = [starting_node]
    while len(stack) > 0:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend([n for n in graph.neighbors(node)])
    return len(seen) == len(graph)

def mixing_test(P: sp.csr_array, sampler: MCMCSampler | SimpleMCMCSampler, counts: tuple, sol_map: dict, sol, num_tries=10000):
    actual = P[[sol_map[sol]]]
    # print('Actual transitions', actual)
    # csr matrix of shape (1, P.shape[1])
    emp_vector = sp.lil_matrix((1, P.shape[1]))
    for _ in range(num_tries):
        # increment the count of the next state
        # TODO use a Counter instead?
        next_sol = sampler.get_next_state(counts, sol)
        emp_vector[0, sol_map[next_sol]] += 1 #type: ignore
        if actual[0, sol_map[next_sol]] == 0:
            print('Empirical transition to 0-probability state')
            print_sol(sol)
            print_sol(next_sol)
            print_sol((Counter(sol) - Counter(next_sol)).elements())
            print('Next sol number', sol_map[next_sol])
            raise ValueError('Empirical transition to 0-probability state')
    emp_vector = emp_vector/np.sum(emp_vector) #type: ignore
    # print('Empirical transitions', emp_vector)
    actual_coo = actual.tocoo()
    diffs = []
    for j, v in zip(actual_coo.col, actual_coo.data):
        diffs.append(np.round(v - emp_vector[0, j], 5))
    print('Diffs', diffs)
    print('TVD', np.sum(np.abs(actual - emp_vector))/2)

def print_sol(sol):
    for k, v in Counter(sol).items():
        print(str(k), v)
    print()

def get_mixing_time_bounds(G: nx.DiGraph, pi_min: float, eps=1/(2*np.exp(1))):
    # TODO for large graphs will have to start working with sparse matrices
    P = nx.to_scipy_sparse_array(G)
    ls = sp.linalg.eigs(P, k=2, which='LM', return_eigenvectors=False)
    l2 = sorted(np.abs(ls), reverse=True)[1]
    print('l2', l2)
    t_rel = 1/(1-l2)
    print('t_rel', t_rel)
    mixing_lb = floor((t_rel-1) * np.log(1/(2*eps)))
    mixing_ub = ceil(t_rel * np.log(1/(eps*pi_min)))
    return (mixing_lb, mixing_ub)

def get_conductance_ub(G: nx.DiGraph, dist: dict, sol_map: dict, gamma: float, counts: tuple):
    print('Getting conductance')
    reverse_sol_map = {v: k for k, v in sol_map.items()}
    sol_num_to_distance = {}
    for sol, sol_num in sol_map.items():
        if sol != ():
            sol_num_to_distance[sol_num] = sum(tup_minus(counts, tup_sum(sol)))
        else:
            sol_num_to_distance[sol_num] = sum(counts)
    log_pi = {}
    log_pi_list_by_layer = {}
    for i, sol_num in enumerate(G.nodes()):
        sol = reverse_sol_map[sol_num]
        log_pi[sol_num] = get_log_prob(sol, dist) - gamma * sol_num_to_distance[sol_num]
        if sol_num_to_distance[sol_num] not in log_pi_list_by_layer:
            log_pi_list_by_layer[sol_num_to_distance[sol_num]] = []
        log_pi_list_by_layer[sol_num_to_distance[sol_num]].append(log_pi[sol_num])
    log_normalizing_constant = logsumexp(list(log_pi.values()))
    normalized_log_pi = {sol_num: log_pi[sol_num] - log_normalizing_constant for sol_num in log_pi}
    log_pi_by_layer = {l: logsumexp(log_pi_list_by_layer[l]) for l in log_pi_list_by_layer}
    # print('Before normalization', log_pi_by_layer)
    pi_by_layer = exp_normalize(log_pi_by_layer)
    # print('After', pi_by_layer)
    all_layers = sorted(list(pi_by_layer.keys()))
    # print('All layers', all_layers)
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
    return Q/(min(pi_S, 1-pi_S))

def get_solution_density(G: nx.DiGraph, gamma: float, dist: dict, reverse_sol_map: dict, counts: tuple):
    true_sol_probs = []
    bad_sol_probs = []
    num_exact_solutions = 0
    for sol_number in G.nodes():
        sol = reverse_sol_map[sol_number]
        p = get_log_prob(sol, dist)
        diff = get_prob_diff_sum(counts, Counter(sol))
        p -= gamma * diff
        if diff == 0:
            num_exact_solutions += 1
            true_sol_probs.append(p)
        else:
            bad_sol_probs.append(p)
    # sum together the true and bad probs using logsumexp
    true_sum = logsumexp(true_sol_probs)
    total_sum = logsumexp(true_sol_probs + bad_sol_probs)
    # print('count', count)
    return np.exp(true_sum - total_sum), num_exact_solutions

# if __name__ == '__main__':
    # parser_builder.parse_args()
    # print(parser_builder.args)
    # args = parser_builder.args
    # #TODO larger k
    # #TODO some analysis of expansion of solution set
    # graphs = {}
    # sol_maps = {}


    # fname = '{}_graph.pkl'
    # # get all files matching the pattern
    # formats = [fname.format(r'simple_(\d+(\.\d+)?)'), fname.format(r'k_\d+')]
    # matching_files = []
    # for form in formats:
        # matching_files += [f for f in os.listdir(args.synthetic_output_dir) if re.match(form, f)]
    # print(matching_files)
    # df = read_block_data(args.block_clean_file)
    # # non-empty rows
    # df = df[df['H7X001'] > 0]
    # print(df.head())
    # dist = encode_hh_dist(read_microdata(args.micro_file))
    # test_row = 3
    # row = df.iloc[test_row]
    # counts = encode_row(row)
    # simple_dist = {k: v for k, v in dist.items() if is_eligible(k, counts)}

    # sol = ip_solve(counts, dist, num_solutions=args.num_sols)
    # # sol_map = {v: i for i, v in enumerate(sol)}
    # # sol_map_copy = sol_map.copy()

    # gammas = {}

    # for file in matching_files:
        # with open(file, 'rb') as f:
            # print(f'Loading {file}')
            # g, sol_map = pickle.load(f)
            # graphs[file] = nx.DiGraph(g)
            # sol_maps[file] = sol_map
    # for graph in graphs:
        # m = re.match(r'simple_(\d+(\.\d+)?)', graph)
        # if m:
            # gammas[graph] = float(m.group(1))

    # # reverse_sol_map = {v: k for k, v in sol_maps['simple'].items()}
    
    # # Test connectivity for k_* graphs
    # print('Testing connectivity')
    # for graph in graphs:
        # if re.match(r'k_\d', graph):
            # graph_is_connected = is_connected(graphs[graph])
            # print(f'{graph} is connected: {graph_is_connected}')


    # test_transition = False
    # if test_transition:
        # print('Testing tranistion matrices')
        # num_to_test = 10
        # for graph in graphs:
            # print('Testing', graph)
            # test_sols = random.choices(list(sol_maps[graph].keys()), k=num_to_test)
            # print('Test numbers', [sol_maps[graph][s] for s in test_sols])
            # if re.match(r'k_\d', graph):
                # for s in test_sols:
                    # k = int(graph.split('_')[1])
                    # mixing_test(nx.to_scipy_sparse_array(graphs[graph], nodelist=sorted(graphs[graph].nodes())), MCMCSampler(dist, k=k), counts, sol_maps[graph], s)
            # else:
                # if graph not in gammas:
                    # continue
                # for s in test_sols:
                    # mixing_test(nx.to_scipy_sparse_array(graphs[graph], nodelist=sorted(graphs[graph].nodes())), SimpleMCMCSampler(simple_dist), counts, sol_maps[graph], s)

    # test_mixing_time = True
    # mixing_times = {}
    # if test_mixing_time:
        # print('Testing mixing time')
        # for graph in graphs:
            # if not graph in gammas:
                # continue
            # print('Testing', graph)
            # mixing_time = get_mixing_time(graphs[graph], 1/len(graphs[graph]))
            # print('Mixing time', mixing_time)
            # if graph in gammas:
                # mixing_times[gammas[graph]] = mixing_time

    # densities = {}
    # test_solution_density = True
    # if test_solution_density:
        # print('Testing solution density')
        # for graph in graphs:
            # if not graph in gammas:
                # continue
            # print('Testing', graph)
            # gamma = gammas[graph]
            # print('gamma', gamma)
            # reverse_sol_map = {v: k for k, v in sol_maps[graph].items()}
            # density = get_solution_density(graphs[graph], gamma, simple_dist, reverse_sol_map, SimpleMCMCSampler(simple_dist, gamma), counts)
            # densities[gamma] = density
            # print('Solution density', density)
    # sorted_gammas = sorted(gammas.values())
    # if len(densities) > 0 and len(mixing_times) > 0:
        # fig, ax1 = plt.subplots()
        # line1, = ax1.plot(sorted_gammas, [1/densities[gamma] for gamma in sorted_gammas], label='1/solution density')
        # ax1.set_yscale('log')
        # ax1.set_xlabel(r'$\gamma$')
        # ax1.set_ylabel('1/solution density')
        # ax2 = ax1.twinx()
        # line2, = ax2.plot(sorted_gammas, [mixing_times[gamma] for gamma in sorted_gammas], label='mixing time', color='r')
        # ax2.set_yscale('log')
        # ax2.set_ylabel('mixing time')
        # lines = [line1, line2]
        # labels = [line.get_label() for line in lines]
        # ax1.legend(lines, labels)
        # plt.savefig('mixing_time_vs_solution_density.png')
        # plt.show()
