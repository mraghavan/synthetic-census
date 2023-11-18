from config2 import ParserBuilder
import pandas as pd
from build_micro_dist import read_microdata
from encoding import encode_hh_dist, encode_row
from ip_distribution import ip_solve
from itertools import combinations
from collections import Counter
from knapsack_utils import tup_sum, tup_minus
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os
from scipy.special import comb

parser_builder = ParserBuilder(
        {'state': True,
         'micro_file': True,
         'block_clean_file': True,
         'synthetic_output_dir': False,
         'num_sols': False,
         })

def read_block_data(block_clean_file: str):
    return pd.read_csv(block_clean_file)

def is_connected(graph: dict):
    seen = set()
    starting_node = next(iter(graph))
    stack = [starting_node]
    while len(stack) > 0:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend([n for n in graph[node]])
    return len(seen) == len(graph)

def get_neighbors(dist: dict, s: tuple, sol_map: dict, k: int):
    neighbors = {}
    cache = set()
    nchoosek = comb(len(s), k)
    total_weight = 0
    for combo in combinations(s, k):
        if tuple(sorted(combo)) in cache:
            continue
        solutions = ip_solve(tup_sum(combo), dist)
        cache.add(tuple(sorted(combo)))
        # print(solutions)
        assert len(solutions) > 0
        for sol in solutions:
            neighbor = list((Counter(s) - Counter(combo) + Counter(sol)).elements())
            neighbor = tuple(sorted(neighbor))
            # deal with self-loops later
            if s == neighbor:
                continue
            if neighbor not in sol_map:
                print(tup_sum(s), tup_sum(neighbor))
                print(tup_minus(tup_sum(s), tup_sum(neighbor)))
                raise ValueError('neighbor not in sol_map')
            if sol_map[neighbor] not in neighbors:
                neighbors[sol_map[neighbor]] = {'weight': 1/len(solutions)/nchoosek}
            else:
                neighbors[sol_map[neighbor]]['weight'] += 1/len(solutions)/nchoosek
            total_weight += 1/len(solutions)/nchoosek
    neighbors[sol_map[s]] = {'weight': 1 - total_weight}
    return neighbors

def build_graph(dist: dict, sol: tuple|list, sol_map: dict, k: int):
    graph = {}
    for s in sol:
        graph[sol_map[s]] = get_neighbors(dist, s, sol_map, k)
        print(graph[sol_map[s]])
        # print(graph[sol_map[s]][sol_map[s]])
        # print(sum(d['weight'] for d in graph[sol_map[s]].values()))
        # print()
        # G = nx.DiGraph(graph)
        # pos = nx.spring_layout(G)
        # nx.draw_networkx_nodes(G,pos)
        # labels = nx.get_edge_attributes(G,'weight')
        # nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
        # plt.show()
    validate_graph(graph)
    return graph

def validate_graph(graph: dict):
    edge_weights = {}
    hits = 0
    for node, edges in graph.items():
        for node2 in edges:
            tup = tuple(sorted((node, node2)))
            if tup in edge_weights:
                if edge_weights[tup] != edges[node2]['weight']:
                    raise ValueError('Difference: {}, {}, {}, {}'.format(node, node2, edge_weights[tup], edges[node2]['weight']))
                hits += 1
            else:
                edge_weights[tup] = edges[node2]['weight']
    print('Should be the same:', len(edge_weights), hits)

if __name__ == '__main__':
    parser_builder.parse_args()
    print(parser_builder.args)
    parser_builder.verify_required_args()
    args = parser_builder.args
    #TODO analyze mixing time. Not sure about this this adjacency_spectrum, especially because eventually we'll have an asymmetric transition matrix
    # Need to read a bit more about what matrix/eigenvalues we're actually interested in
    #TODO larger k
    #TODO some analysis of expansion of solution set

    graph = None
    k = 3
    if os.path.exists('graph.pkl'):
        with open('graph.pkl', 'rb') as f:
            graph = pickle.load(f)
    else:
        df = read_block_data(args.block_clean_file)
        # non-empty rows
        df = df[df['H7X001'] > 0]
        print(df.head())
        dist = encode_hh_dist(read_microdata(args.micro_file))
        test_row = 3
        row = df.iloc[test_row]
        counts = encode_row(row)
        sol = ip_solve(counts, dist, num_solutions=args.num_sols)
        print(len(sol), 'solutions')
        sol_map = {v: i for i, v in enumerate(sol)}
        graph = build_graph(dist, sol, sol_map, k)
        print(graph)
        with open('graph.pkl', 'wb') as f:
            pickle.dump(graph, f)

    print('connected?', is_connected(graph))
    G = nx.Graph(graph)
    print(sorted(nx.adjacency_spectrum(G), reverse=True)[1])
    G.remove_edges_from(nx.selfloop_edges(G))
    nx.draw(G, node_size=30)
    plt.show()
