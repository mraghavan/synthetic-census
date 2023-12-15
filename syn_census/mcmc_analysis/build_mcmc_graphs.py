import pandas as pd
from itertools import combinations
from collections import Counter
from scipy.special import comb
from scipy.linalg import eig
import numpy as np
from ..utils.config2 import ParserBuilder
from ..synthetic_data_generation.mcmc_sampler import get_log_prob, get_prob_diff_sum
from ..utils.knapsack_utils import tup_sum, tup_minus, is_eligible, tup_plus, is_feasible
# from ..utils.census_utils import approx_equal
from ..utils.ip_distribution import ip_enumerate
# from ..utils.encoding import encode_hh_dist, encode_row
# from ..preprocessing.build_micro_dist import read_microdata

parser_builder = ParserBuilder(
        {'state': True,
         'micro_file': True,
         'block_clean_file': True,
         'synthetic_output_dir': False,
         'num_sols': False,
         })

class IncompleteError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def read_block_data(block_clean_file: str):
    return pd.read_csv(block_clean_file)

def get_neighbors(dist: dict, s: tuple, sol_map: dict, k: int):
    neighbors = {} #type: dict[int, dict]
    cache = set()
    nchoosek = comb(len(s), k)
    total_weight = 0
    x_prob = get_log_prob(s, dist)
    for combo in combinations(s, k):
        if tuple(sorted(combo)) in cache:
            continue
        solutions = ip_enumerate(tup_sum(combo), tuple(dist.keys()), num_solutions=len(sol_map))
        cache.add(tuple(sorted(combo)))
        assert len(solutions) > 0
        for sol in solutions:
            neighbor = list((Counter(s) - Counter(combo) + Counter(sol)).elements())
            neighbor = tuple(sorted(neighbor))
            # deal with self-loops later
            if s == neighbor:
                continue
            if neighbor not in sol_map:
                # sol_map[neighbor] = len(sol_map)
                # print('Adding to sol_map', len(sol_map))
                print(tup_sum(s), tup_sum(neighbor))
                print(tup_minus(tup_sum(s), tup_sum(neighbor)))
                raise ValueError('neighbor not in sol_map')
            xprime_prob = get_log_prob(neighbor, dist)
            ratio = np.exp(xprime_prob - x_prob)
            # factor of 2 because it's lazy
            prob = 1/(2*(len(solutions)-1)*nchoosek) * min(1, ratio)
            if sol_map[neighbor] not in neighbors:
                neighbors[sol_map[neighbor]] = {'weight': prob}
            else:
                neighbors[sol_map[neighbor]]['weight'] += prob
            total_weight += prob
    neighbors[sol_map[s]] = {'weight': 1 - total_weight}
    return neighbors

def build_graph(dist: dict, sol, sol_map: dict, k: int):
    graph = {}
    for s in sol:
        graph[sol_map[s]] = get_neighbors(dist, s, sol_map, k)
        if sol_map[s] % 50 == 0:
            print(sol_map[s])
        # print(sol_map[s], graph[sol_map[s]])
        # print('len', len(sol_map))
    return graph, sol_map

def get_d_maxes(dist: dict, counts: tuple):
    d_maxes = {}
    for item in dist:
        d_max = 0
        cur_item = item
        while is_eligible(cur_item, counts):
            d_max += 1
            cur_item = tup_plus(cur_item, item)
        d_maxes[item] = d_max
    return d_maxes

def build_graph_simple(dist: dict, counts: tuple, gammas: list, total_solutions=0, fail_time = 1000000):
    empty_sol = tuple()
    # same sol_map for each gamma
    sol_map = {}
    reverse_sol_map = {}
    sol_map[empty_sol] = len(sol_map)
    reverse_sol_map[sol_map[empty_sol]] = empty_sol
    graphs = {g: {} for g in gammas}
    first_gamma = gammas[0]
    first_graph = graphs[first_gamma]
    stack = [empty_sol]
    get_neighbors_simple.num_exact = 0
    d_maxes = get_d_maxes(dist, counts)
    dist = {k: v for k, v in dist.items() if d_maxes[k] > 0}
    while len(stack) > 0:
        node = stack.pop()
        if sol_map[node] in first_graph:
            continue
        if len(node) > fail_time:
            print('Too many states. Aborting')
            raise IncompleteError('Failed to find exact solutions')
        if len(first_graph) % 1000 == 0:
            print('Processed', len(first_graph), 'exact solutions', get_neighbors_simple.num_exact, '/', total_solutions, ('projected {}'.format(int(len(first_graph)*(total_solutions+1)/max(1, get_neighbors_simple.num_exact))) if total_solutions > 0 else ''))
        all_neighbors = get_neighbors_simple(dist, node, counts, sol_map, reverse_sol_map, d_maxes, gammas)
        for gamma in gammas:
            graphs[gamma][sol_map[node]] = all_neighbors[gamma]
        # graph[sol_map[node]] = get_neighbors_simple(dist, node, counts, sol_map, reverse_sol_map, d_maxes, gammas)
        stack.extend([reverse_sol_map[n] for n in first_graph[sol_map[node]]])
    return graphs, sol_map

def get_neighbors_simple(dist: dict, s: tuple, counts: tuple, sol_map: dict, reverse_sol_map: dict, d_maxes: dict, gammas: list):
    if len(s) > 0:
        if all(i == j for i, j in zip(tup_sum(s), counts)):
            get_neighbors_simple.num_exact += 1
            # print('Exact solution found', get_neighbors_simple.num_exact)
    neighbors_by_gamma = {g: {} for g in gammas}
    first_gamma = gammas[0]
    total_weight_by_gamma = {g: 0 for g in gammas}
    for item in dist:
        d_max = d_maxes[item]
        if d_max == 0:
            # item can't be added
            continue
        for i in range(0, d_max+1):
            if len(s) > 0:
                new_s_counter = Counter(s)
            else:
                new_s_counter = Counter()
            new_s_counter[item] = i
            new_s = tuple(sorted(list(new_s_counter.elements())))
            if new_s == s:
                continue
            if len(new_s) > 0 and not is_feasible(new_s, counts):
                continue
            if new_s not in sol_map:
                sol_map[new_s] = len(sol_map)
                reverse_sol_map[sol_map[new_s]] = new_s
            if sol_map[new_s] in neighbors_by_gamma[first_gamma]:
                raise ValueError('Got the same solution')
            # make sure not to change these in the for loop
            x_prob_diff_sum = get_prob_diff_sum(counts, Counter(s))
            xprime_prob_diff_sum = get_prob_diff_sum(counts, new_s_counter)
            for gamma in gammas:
                w = 1/(2*len(dist) * d_max)
                x_prob = get_log_prob(s, dist) - gamma * x_prob_diff_sum
                xprime_prob = get_log_prob(new_s, dist) - gamma * xprime_prob_diff_sum
                ratio = np.exp(xprime_prob - x_prob)
                A = min(1, ratio)
                w *= A
                total_weight_by_gamma[gamma] += w
                neighbors_by_gamma[gamma][sol_map[new_s]] = {'weight': w}
    for gamma in gammas:
        # set the self loop probability
        neighbors_by_gamma[gamma][sol_map[s]] = {'weight': 1 - total_weight_by_gamma[gamma]}
    return neighbors_by_gamma
get_neighbors_simple.num_exact = 0

# def is_stochastic(G: nx.DiGraph):
    # for node in G.nodes():
        # outgoing_edges = [e for e in G.edges(node)]
        # probabilities = [G[e[0]][e[1]]['weight'] for e in outgoing_edges]
        # if not approx_equal(sum(probabilities), 1.0):
            # print(sum(probabilities))
            # return False
    # return True

def get_tvd_at_iterations(P: np.ndarray, t: int):
    # print(P)
    initial = np.zeros((P.shape[0],))
    initial[0] = 1
    P_t = np.linalg.matrix_power(P, t)
    ls, vl = eig(P, left=True, right=False) #type: ignore
    i = 0
    # TODO find largest eignevalue instead
    for i, l in enumerate(ls):
        if np.abs(np.real(l) - 1.0) < .00001:
            break
    pi = vl[:,i]
    pi = pi/np.sum(pi)
    v_t = P_t.T.dot(initial)
    v_t = v_t / np.sum(v_t)
    return np.sum(np.abs(v_t - pi))/2
