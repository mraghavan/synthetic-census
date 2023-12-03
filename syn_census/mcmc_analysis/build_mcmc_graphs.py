import pandas as pd
from itertools import combinations
from collections import Counter
import networkx as nx
from scipy.special import comb
from scipy.linalg import eig
import numpy as np
from ..utils.config2 import ParserBuilder
from ..synthetic_data_generation.mcmc_sampler import get_log_prob, SimpleMCMCSampler
from ..utils.knapsack_utils import tup_sum, tup_minus, is_eligible, tup_plus, is_feasible
from ..utils.census_utils import approx_equal
from ..utils.ip_distribution import ip_solve
# from ..utils.encoding import encode_hh_dist, encode_row
# from ..preprocessing.build_micro_dist import read_microdata

parser_builder = ParserBuilder(
        {'state': True,
         'micro_file': True,
         'block_clean_file': True,
         'synthetic_output_dir': False,
         'num_sols': False,
         })

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
        solutions = ip_solve(tup_sum(combo), dist)
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
            prob = 1/(2*len(solutions)*nchoosek) * min(1, ratio)
            if sol_map[neighbor] not in neighbors:
                neighbors[sol_map[neighbor]] = {'weight': prob}
            else:
                neighbors[sol_map[neighbor]]['weight'] += prob
            total_weight += prob
    neighbors[sol_map[s]] = {'weight': 1 - total_weight}
    return neighbors

def build_graph(dist: dict, sol: tuple|list, sol_map: dict, k: int):
    graph = {}
    for s in sol:
        graph[sol_map[s]] = get_neighbors(dist, s, sol_map, k)
        if sol_map[s] % 10 == 0:
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

def build_graph_simple(dist: dict, counts: tuple, sampler: SimpleMCMCSampler, total_solutions=0):
    empty_sol = tuple()
    sol_map = {}
    reverse_sol_map = {}
    sol_map[empty_sol] = len(sol_map)
    reverse_sol_map[sol_map[empty_sol]] = empty_sol
    graph = {}
    stack = [empty_sol]
    get_neighbors_simple.num_exact = 0
    d_maxes = get_d_maxes(dist, counts)
    dist = {k: v for k, v in dist.items() if d_maxes[k] > 0}
    while len(stack) > 0:
        node = stack.pop()
        if sol_map[node] in graph:
            continue
        if len(graph) % 1000 == 0:
            print('Processed', len(graph), 'exact solutions', get_neighbors_simple.num_exact, ('projected {}'.format(int(len(graph)*(total_solutions+1)/max(1, get_neighbors_simple.num_exact))) if total_solutions > 0 else ''))
        graph[sol_map[node]] = get_neighbors_simple(dist, node, counts, sol_map, reverse_sol_map, d_maxes, sampler)
        stack.extend([reverse_sol_map[n] for n in graph[sol_map[node]]])
    return graph, sol_map

def get_neighbors_simple(dist: dict, s: tuple, counts: tuple, sol_map: dict, reverse_sol_map: dict, d_maxes: dict, sampler: SimpleMCMCSampler):
    if len(s) > 0:
        if all(i == 0 for i in tup_minus(tup_sum(s), counts)):
            get_neighbors_simple.num_exact += 1
            # print('Exact solution found', get_neighbors_simple.num_exact)
    neighbors = {}
    total_weight = 0
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
            if sol_map[new_s] in neighbors:
                raise ValueError('Got the same solution')
            w = 1/(2*len(dist) * (d_max + 1))
            x_prob_diff_sum = sampler.get_prob_diff_sum(counts, Counter(s))
            x_prob = get_log_prob(s, sampler.dist) - sampler.gamma * x_prob_diff_sum
            xprime_prob_diff_sum = sampler.get_prob_diff_sum(counts, new_s_counter)
            xprime_prob = get_log_prob(new_s, sampler.dist) - sampler.gamma * xprime_prob_diff_sum
            ratio = np.exp(xprime_prob - x_prob)
            A = min(1, ratio)
            w *= A
            total_weight += w
            neighbors[sol_map[new_s]] = {'weight': w}
    neighbors[sol_map[s]] = {'weight': 1 - total_weight}
    return neighbors
get_neighbors_simple.num_exact = 0

def is_stochastic(G: nx.DiGraph):
    for node in G.nodes():
        outgoing_edges = [e for e in G.edges(node)]
        probabilities = [G[e[0]][e[1]]['weight'] for e in outgoing_edges]
        if not approx_equal(sum(probabilities), 1.0):
            print(sum(probabilities))
            return False
    return True

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
    # print(ls)
    # print(i)
    # print(ls[i])
    pi = vl[:,i]
    pi = pi/np.sum(pi)
    v_t = P_t.T.dot(initial)
    # print(np.sum(v_t))
    v_t = v_t / np.sum(v_t)
    # print(pi)
    # print(v_t)
    # print(sum(pi))
    # print(sum(v_t))
    return np.sum(np.abs(v_t - pi))/2

# if __name__ == '__main__':
    # parser_builder.parse_args()
    # print(parser_builder.args)
    # parser_builder.verify_required_args()
    # args = parser_builder.args
    # #TODO larger k
    # #TODO some analysis of expansion of solution set
    # # TODO modify SimpleMCMCSampler to incorporate probabilities and gamma


    # graph = None
    # fname = '{}_graph.pkl'
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
    # sol_map = {v: i for i, v in enumerate(sol)}
    # sol_map_copy = sol_map.copy()

    # gammas = [0, 0.1, 0.2, 0.5, 1]
    # ks = [2, 3]

    # graphs_to_build = {f'simple_{gamma}': (lambda param=gamma: build_graph_simple(dist, counts, SimpleMCMCSampler(simple_dist, gamma=param), total_solutions=len(sol))) for gamma in gammas}
    # graphs_to_build.update({f'k_{k}': (lambda param=k: build_graph(dist, sol, sol_map_copy, k=param)) for k in ks})

    # for g, func in graphs_to_build.items():
        # if not os.path.exists(fname.format(g)):
            # print('Building graph', g)
            # graph = func()
            # with open(fname.format(g), 'wb') as f:
                # pickle.dump(graph, f)
    
    # graph, sol_map, reverse_sol_map = build_graph_simple(dist, counts, total_solutions=len(sol))
    # print('total number of nodes', len(graph))
    # print(len(reverse_sol_map))
    # print('Number of exact solutions', sum(1 for n in graph if len(reverse_sol_map[n]) > 0 and tup_sum(reverse_sol_map[n]) == counts))
    # for n in graph:
        # if len(reverse_sol_map[n]) == 0:
            # continue
        # diff = tup_minus(counts, tup_sum(reverse_sol_map[n]))
        # print(sum(diff))

    # print(first_sol)
    # print(len(sol), 'solutions')
    # if os.path.exists(fname):
        # with open(fname, 'rb') as f:
            # graph = pickle.load(f)
    # else:
        # graph = build_graph(dist, sol, sol_map_copy, k)
        # print(graph)
        # with open(fname, 'wb') as f:
            # pickle.dump(graph, f)

    # print('connected?', is_connected(graph))
    # G = nx.DiGraph(graph)
    # print('is stochastic?', is_stochastic(G))
    
    # # print('Number of new states', len(sol_map) - len(sol_map_copy))
    # sols_with_probs = recompute_probs(sol, dist)
    # pi_min = min(sols_with_probs.values())
    # for s in sol:
        # mixing_test(nx.to_numpy_array(G), MCMCSampler(dist, k=k), sol_map, s)

    # eps = 1/(2*np.exp(1))
    # mt = get_mixing_time(G, pi_min, eps=eps)
    # print('Mixing time', mt)
    # P = nx.to_numpy_array(G)
    # ts = [1, 10, 100, 1000, 10000]
    # tvds = [get_tvd_at_iterations(P, t) for t in ts]
    # plt.plot(ts, tvds)
    # plt.xscale('log')
    # plt.ylabel('TVD')
    # plt.plot([mt, mt], [0, 1], color='k', ls=':')
    # plt.plot([min(ts), max(ts)], [eps, eps], color='k', ls=':')
    # plt.show()

    # for t in [1, 10, 100, 1000, 10000]:
        # print('tvd at', t, get_tvd_at_iterations(nx.to_numpy_array(G), t))
    # print(sorted(nx.adjacency_spectrum(G), reverse=True)[1])
    # G.remove_edges_from(nx.selfloop_edges(G))
    # nx.draw(G, node_size=30, pos=nx.spring_layout(G))
    # plt.show()
