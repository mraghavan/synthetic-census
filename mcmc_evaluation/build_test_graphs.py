import pickle
import os
from functools import lru_cache
from itertools import product
import sys
sys.path.append('../')
from syn_census.mcmc_analysis.build_mcmc_graphs import get_d_maxes, get_neighbors, get_neighbors_simple
from syn_census.synthetic_data_generation.mcmc_sampler import SimpleMCMCSampler
from syn_census.utils.ip_distribution import ip_solve
from syn_census.utils.knapsack_utils import normalize

def make_vector_two_ones(i: int, j: int, d: int):
    l = [0] * d
    l[i] = 1
    l[j] = 1
    return tuple(l)

def make_vector_one_two(i: int, d: int):
    l = [0] * d
    l[i] = 2
    return tuple(l)

@lru_cache(maxsize=None)
def make_dist_and_counts(c:int, d: int):
    dist = {}
    for i in range(d):
        dist[make_vector_one_two(i, d)] = 1
        for j in range(i+1, d):
            dist[make_vector_two_ones(i, j, d)] = 1
    dist = normalize(dist)
    counts = tuple(c for _ in range(d))
    return dist, counts

def get_all_solutions(c, d):
    dist, counts = make_dist_and_counts(c, d)
    return ip_solve(counts, dist, num_solutions=10000)

def build_simple_graph(c: int, d: int, gamma: float):
    """
    Build a graph corresopnsing to the d-dimensional knapsack problem
    with capacity c in each dimension.
    """
    assert c % 2 == 0
    get_neighbors_simple.num_exact = 0
    dist, counts = make_dist_and_counts(c, d)
    d_maxes = get_d_maxes(dist, counts)
    empty_sol = tuple()
    stack = [empty_sol]
    graph = {}
    sol_map = {}
    sol_map[empty_sol] = len(sol_map)
    reverse_sol_map = {sol_map[empty_sol]: empty_sol}
    sampler = SimpleMCMCSampler(dist, gamma)
    while len(stack) > 0:
        node = stack.pop()
        if sol_map[node] in graph:
            continue
        # if len(graph) % 1000 == 0:
            # print('Processed', len(graph), 'solutions')
        graph[sol_map[node]] = get_neighbors_simple(dist, node, counts, sol_map, reverse_sol_map, d_maxes, sampler)
        stack.extend([reverse_sol_map[n] for n in graph[sol_map[node]]])
    return graph, sol_map

def build_reduced_graph(dist, k: int, all_solutions: tuple|list):
    graph = {}
    sol_map = {}
    for i, s in enumerate(sorted(all_solutions)):
        sol_map[s] = i
    for s in all_solutions:
        graph[sol_map[s]] = get_neighbors(dist, s, sol_map, k)
        # if sol_map[s] % 10 == 0:
            # print(sol_map[s])
        # print(sol_map[s], graph[sol_map[s]])
        # print('len', len(sol_map))
    return graph, sol_map

def save_graph(graph, sol_map, fname):
    print('Saving graph to', fname)
    with open(fname, 'wb') as f:
        pickle.dump((graph, sol_map), f)

if __name__ == '__main__':
    cs = [4, 6, 8]
    ds = [3, 4, 5, 6]
    gammas = [0.0, .1, 0.2, .3, .4, .5, .6, .7, .8, 1.0]
    ks = [2]
    failures = set()

    save_dir = 'test_graphs/'
    save_name = 'test_mcmc_{c}_{d}_{param}.pkl'

    # make reduced graphs
    for c, d, k, in product(cs, ds, ks):
        if (c, d) in failures:
            print('Skipping', c, d)
            continue
        fname = os.path.join(save_dir, save_name.format(c=c, d=d, param=k))
        if os.path.exists(fname):
            print('File exists', fname)
            continue
        print('c, d, k', c, d, k)
        dist, counts = make_dist_and_counts(c, d)
        all_solutions = get_all_solutions(c, d)
        print('Number of solutions', len(all_solutions))
        try:
            graph, sol_map = build_reduced_graph(dist, k, all_solutions)
            save_graph(graph, sol_map, fname)
        except ValueError as e:
            print('Error:', e)
            print('Likely cause: solution limit too low')
            failures.add((c, d))
        # G = nx.DiGraph(graph)
        # nx.draw(G, with_labels=True)
        # plt.show()

    for c, d, gamma in product(cs, ds, gammas):
        if (c, d) in failures:
            print('Skipping', c, d)
            continue
        fname = os.path.join(save_dir, save_name.format(c=c, d=d, param=gamma))
        if os.path.exists(fname):
            print('File exists', fname)
            continue
        print('c, d, gamma', c, d, gamma)
        graph, sol_map = build_simple_graph(c, d, gamma)
        save_graph(graph, sol_map, fname)
        # G = nx.DiGraph(graph)
        # nx.draw(G, with_labels=True)
        # plt.show()
