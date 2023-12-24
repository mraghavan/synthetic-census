import pandas as pd
import time
from itertools import combinations
from collections import Counter, OrderedDict
from scipy.special import comb
from scipy.linalg import eig
import numpy as np
from ..utils.config2 import ParserBuilder
from ..synthetic_data_generation.guided_solver import recompute_probs, recompute_probs_gamma
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

def get_neighbors_reduced(dist: dict, s: tuple, sol_map: dict, k: int):
    neighbors = {} #type: dict[int, dict]
    cache = set()
    nchoosek = comb(len(s), k)
    total_weight = 0
    # x_prob = get_log_prob(s, dist)
    elements = tuple(dist.keys())
    for combo in combinations(s, k):
        tup_combo = tuple(sorted(combo))
        if tup_combo in cache:
            continue
        solutions = ip_enumerate(tup_sum(combo), elements, num_solutions=len(sol_map))
        cache.add(tup_combo)
        assert len(solutions) > 0
        all_sols = []
        for sol in solutions:
            neighbor = list((Counter(s) - Counter(combo) + Counter(sol)).elements())
            neighbor = tuple(sorted(neighbor))
            if neighbor not in sol_map:
                print(tup_sum(s), tup_sum(neighbor))
                print(tup_minus(tup_sum(s), tup_sum(neighbor)))
                raise ValueError('neighbor not in sol_map')
            all_sols.append(neighbor)
        sols_with_probs = recompute_probs(all_sols, dist)
        for sol, pi in sols_with_probs.items():
            prob = pi / (2*nchoosek)
            if sol_map[sol] not in neighbors:
                neighbors[sol_map[sol]] = {'weight': prob}
            else:
                neighbors[sol_map[sol]]['weight'] += prob
    del neighbors[sol_map[s]]
    total_weight = sum([v['weight'] for v in neighbors.values()])
    assert total_weight <= 0.5
    neighbors[sol_map[s]] = {'weight': 1 - total_weight}
    return neighbors

def build_graph_reduced(dist: dict, sol, sol_map: dict, k: int, fail_time=3*60*60):
    graph = {}
    start_time = time.time()
    for s in sol:
        graph[sol_map[s]] = get_neighbors_reduced(dist, s, sol_map, k)
        elapsed = time.time() - start_time
        projected = int(len(sol)*elapsed/max(1, len(graph)))
        if sol_map[s] % 50 == 0:
            print('Processed', len(graph), 'solutions.', f'Elapsed {elapsed:.2f} seconds.', f'Projected {projected} seconds.', f'fail_time {fail_time} seconds.')
        if len(graph) > 200 and projected > fail_time:
            print('Too much time. Aborting')
            raise IncompleteError('Failed to find exact solutions')
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

def build_graph_gibbs(dist: OrderedDict, counts: tuple, gammas: list, total_solutions=0, fail_time = 8*60*60):
    empty_sol = tuple()
    # one sol_map for each gamma
    sol_map = {}
    reverse_sol_map = {}
    sol_map[empty_sol] = len(sol_map)
    reverse_sol_map[sol_map[empty_sol]] = empty_sol
    graphs = {g: {} for g in gammas}
    first_gamma = gammas[0]
    first_graph = graphs[first_gamma]
    stack = [empty_sol]
    get_neighbors_gibbs.num_exact = 0
    start_time = time.time()
    while len(stack) > 0:
        node = stack.pop()
        if sol_map[node] in first_graph:
            continue
        # if sol_map[node] > fail_time:
            # print('Too many states. Aborting')
            # raise IncompleteError('Failed to find exact solutions')
        if len(first_graph) % 1000 == 0:
            elapsed = time.time() - start_time
            projected = int(elapsed*(total_solutions+1)/max(1, get_neighbors_gibbs.num_exact))
            print('Processed', len(first_graph), 'states.',
                  'Found {}/{} exact solutions.'.format(get_neighbors_gibbs.num_exact, total_solutions) if total_solutions > 0 else '',
                  f'Elapsed {elapsed:.2f} seconds.',
                  f'Projected {projected} seconds.',
                  f'fail_time {fail_time} seconds.')
            if len(first_graph) > 20000 and projected > fail_time:
                print('Too much time. Aborting')
                raise IncompleteError('Failed to find exact solutions')
            # print('Processed', len(first_graph), 'exact solutions', get_neighbors_gibbs.num_exact, '/', total_solutions, ('projected {}'.format(projected) if total_solutions > 0 else ''))
        all_neighbors = get_neighbors_gibbs(dist, node, counts, sol_map, reverse_sol_map, gammas)
        for gamma in gammas:
            graphs[gamma][sol_map[node]] = all_neighbors[gamma]
        stack.extend([reverse_sol_map[n] for n in first_graph[sol_map[node]]])
    return graphs, sol_map

def get_neighbors_gibbs(dist: dict, s: tuple, counts: tuple, sol_map: dict, reverse_sol_map: dict, gammas: list):
    if len(s) > 0:
        if all(i == j for i, j in zip(tup_sum(s), counts)):
            get_neighbors_gibbs.num_exact += 1
            # print('Exact solution found', get_neighbors_simple.num_exact)
    all_neighbors_by_gamma = {g: {} for g in gammas}
    first_gamma = gammas[0]
    for item in dist:
        neighbors_by_gamma = {g: {} for g in gammas}
        if len(s) > 0:
            base_s_counter = Counter(s)
        else:
            base_s_counter = Counter()
        base_s_counter[item] = 0
        base_s = tuple(sorted(list(base_s_counter.elements())))
        if not base_s:
            remainder = counts
        else:
            remainder = tup_minus(counts, tup_sum(base_s))
        j = 0
        item_sum = (0,)*len(counts)
        s_norm_without_i = sum(base_s_counter.values())
        v_i_norm = sum(item)
        prev_probs = {g: 1 for g in gammas}
        all_neighbors = []
        while all(a <= b for a, b in zip(item_sum, remainder)):
            new_s = base_s + (item,)*j
            new_s = tuple(sorted(new_s))
            if new_s not in sol_map:
                sol_map[new_s] = len(sol_map)
                reverse_sol_map[sol_map[new_s]] = new_s
            all_neighbors.append(new_s)
            if j == 0:
                for g in gammas:
                    neighbors_by_gamma[g][sol_map[new_s]] = 1
                    prev_probs[g] = 1
            else:
                for g in gammas:
                    neighbors_by_gamma[g][sol_map[new_s]] = prev_probs[g] * ((s_norm_without_i + j) / j) * dist[item] * np.exp(g*v_i_norm)
                    prev_probs[g] = neighbors_by_gamma[g][sol_map[new_s]]
            item_sum = tup_plus(item_sum, item)
            j += 1
        # normalize each neighbors_by_gamma[g]
        weight_by_gamma = {g: sum(neighbors_by_gamma[g].values()) for g in gammas}
        for g in gammas:
            for sol_num in neighbors_by_gamma[g]:
                if sol_num != sol_map[s]:
                    # divide by 2 to make the chain lazy
                    all_neighbors_by_gamma[g][sol_num] = {'weight': neighbors_by_gamma[g][sol_num] / (2*weight_by_gamma[g]*len(dist))}

        # neighbors_by_gamma = {g: recompute_probs_gamma(all_neighbors, dist, counts, g) for g in gammas}
        # # weight_by_gamma = {g: sum(neighbors_by_gamma[g].values()) for g in gammas}
        # for g in gammas:
            # for neighbor in neighbors_by_gamma[g]:
                # if neighbor != s:
                    # # divide by 2 to make the chain lazy
                    # all_neighbors_by_gamma[g][sol_map[neighbor]] = {'weight': neighbors_by_gamma[g][neighbor] / (2*len(dist))}

    total_weight_by_gamma = {g: 0 for g in gammas}
    for gamma in gammas:
        total_weight_by_gamma[gamma] = sum(all_neighbors_by_gamma[gamma][k]['weight']for k in all_neighbors_by_gamma[gamma])
        assert total_weight_by_gamma[gamma] <= 0.5
    for gamma in gammas:
        # set the self loop probability
        all_neighbors_by_gamma[gamma][sol_map[s]] = {'weight': 1 - total_weight_by_gamma[gamma]}
    return all_neighbors_by_gamma
get_neighbors_gibbs.num_exact = 0

        # for i in range(0, d_max+1):
            # if len(s) > 0:
                # new_s_counter = Counter(s)
            # else:
                # new_s_counter = Counter()
            # new_s_counter[item] = i
            # new_s = tuple(sorted(list(new_s_counter.elements())))
            # if new_s == s:
                # continue
            # if len(new_s) > 0 and not is_feasible(new_s, counts):
                # continue
            # if new_s not in sol_map:
                # sol_map[new_s] = len(sol_map)
                # reverse_sol_map[sol_map[new_s]] = new_s
            # if sol_map[new_s] in neighbors_by_gamma[first_gamma]:
                # raise ValueError('Got the same solution')
            # make sure not to change these in the for loop
            # x_prob_diff_sum = get_prob_diff_sum(counts, Counter(s))
            # xprime_prob_diff_sum = get_prob_diff_sum(counts, new_s_counter)
            # for gamma in gammas:
                # w = 1/(2*len(dist) * d_max)
                # x_prob = get_log_prob(s, dist) - gamma * x_prob_diff_sum
                # xprime_prob = get_log_prob(new_s, dist) - gamma * xprime_prob_diff_sum
                # ratio = np.exp(xprime_prob - x_prob)
                # A = min(1, ratio)
                # w *= A
                # total_weight_by_gamma[gamma] += w
                # neighbors_by_gamma[gamma][sol_map[new_s]] = {'weight': w}
    # for gamma in gammas:
        # # set the self loop probability
        # neighbors_by_gamma[gamma][sol_map[s]] = {'weight': 1 - total_weight_by_gamma[gamma]}
    # return neighbors_by_gamma
# get_neighbors_gibbs.num_exact = 0

def build_graph_simple(dist: OrderedDict, counts: tuple, gammas: list, total_solutions=0, fail_time = 1000000):
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
    dist = OrderedDict({k: v for k, v in dist.items() if d_maxes[k] > 0})
    while len(stack) > 0:
        node = stack.pop()
        if sol_map[node] in first_graph:
            continue
        if sol_map[node] > fail_time:
            print('Too many states. Aborting')
            raise IncompleteError('Failed to find exact solutions')
        if len(first_graph) % 1000 == 0:
            projected = int(len(first_graph)*(total_solutions+1)/max(1, get_neighbors_simple.num_exact))
            print('Processed', len(first_graph), 'exact solutions', get_neighbors_simple.num_exact, '/', total_solutions, ('projected {}'.format(projected) if total_solutions > 0 else ''))
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
