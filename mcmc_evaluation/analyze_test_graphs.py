import pickle
import numpy as np
import re
import os
import sys
import networkx as nx
import pandas as pd
sys.path.append('../')
from syn_census.synthetic_data_generation.mcmc_sampler import SimpleMCMCSampler
from syn_census.mcmc_analysis.analyze_mcmc_graphs import get_mixing_time_bounds, get_solution_density, get_conductance_ub
from build_test_graphs import make_dist_and_counts


PATTERN = r'test_mcmc_(\d+)_(\d+)_(\d+(.\d+)?).pkl'

def is_simple(key):
    return type(key[2]) == float

def load_all_graphs(open_dir):
    graphs = {}
    sol_maps = {}
    for filename in os.listdir(open_dir):
        m = re.match(PATTERN, filename)
        if m:
            with open(os.path.join(open_dir, filename), 'rb') as f:
                print('Loading', filename)
                c = int(m.group(1))
                d = int(m.group(2))
                param = m.group(3)
                if '.' in param:
                    param = float(param)
                else:
                    param = int(param)
                graph, sol_map = pickle.load(f)
                graphs[(c, d, param)] = graph
                sol_maps[(c, d, param)] = sol_map
    return graphs, sol_maps

def get_all_graph_files(open_dir):
    files = []
    for filename in os.listdir(open_dir):
        m = re.match(PATTERN, filename)
        if m:
            files.append(filename)
    return files

if __name__ == '__main__':
    open_dir = 'test_graphs/'
    all_graph_files = get_all_graph_files(open_dir)
    # graphs, sol_maps = load_all_graphs(open_dir)
    # print(graphs.keys())
    
    results_out = 'test_mcmc_results_{c}_{d}_{param}.pkl'
    for filename in all_graph_files:
        m = re.match(PATTERN, filename)
        assert m is not None
        c = int(m.group(1))
        d = int(m.group(2))
        param = m.group(3)
        if '.' in param:
            param = float(param)
        else:
            param = int(param)
        results_out_file = os.path.join(open_dir, results_out.format(c=c, d=d, param=param))
        if os.path.exists(results_out_file):
            print('Skipping', (c, d, param))
            continue
        with open(os.path.join(open_dir, filename), 'rb') as f:
            print('Loading', filename)
            graph, sol_map = pickle.load(f)
            reverse_sol_map = {v: k for k, v in sol_map.items()}
            dist, counts = make_dist_and_counts(c, d)
            results = {}
            print((c, d, param), 'graph len', len(graph))
            G = nx.DiGraph(graph)
            print('G len', len(G), np.log(len(G)))
            results['mixing_time'] = get_mixing_time_bounds(G, 1/len(G))
            results['num_states'] = len(G)
            if is_simple((c, d, param)):
                gamma = param
                sampler = SimpleMCMCSampler(dist, gamma)
                results['solution_density'] = get_solution_density(G, gamma, dist, reverse_sol_map, sampler, counts)
                results['conductance_ub'] = get_conductance_ub(G, dist, sol_map, gamma, counts)
            print((c, d, param), results)
            with open(results_out_file, 'wb') as f_out:
                pickle.dump(results, f_out)
