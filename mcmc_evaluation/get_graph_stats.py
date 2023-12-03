import sys
import os
import pickle
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

SIMPLE_PATTERN = r'simple_(\d+(\.\d+)?)'
K_PATTERN = r'k_(\d+)'

def is_simple(fname):
    return re.match(SIMPLE_PATTERN, fname)

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
    # Pick a row with a non-trivial number of solutions, probably < 1000 though
    # TODO try a different row
    test_row = 3
    row = df.iloc[test_row]
    counts = encode_row(row)
    simple_dist = {k: v for k, v in dist.items() if is_eligible(k, counts)}

    sol = ip_solve(counts, dist, num_solutions=args.num_sols)

    gammas = {}
    common_analyses_to_run = {
            'mixing_time': lambda g, sm, param: get_mixing_time_bounds(g, 1/len(g)),
            }
    simple_analyses_to_run = {
            'solution_density': lambda g, sm, param: get_solution_density(g, param, simple_dist, {v: k for k, v in sm.items()}, SimpleMCMCSampler(simple_dist, param), counts),
            'conductance_ub': lambda g, sm, param: get_conductance_ub(g, simple_dist, sm, param, counts),
            }
    simple_analyses_to_run.update(common_analyses_to_run)
    k_analyses_to_run = {
            'connectivity': lambda g, sm, param: is_connected(g),
            }
    k_analyses_to_run.update(common_analyses_to_run)

    for file in matching_files:
        with open(os.path.join(args.mcmc_output_dir, file), 'rb') as f:
            results_file = os.path.join(args.mcmc_output_dir, file.replace('.pkl', '_results.pkl'))
            if os.path.exists(results_file):
                print(f'{results_file} exists, skipping')
                continue
            results = {}
            analyses_to_run = None
            param = 0
            if is_simple(file):
                analyses_to_run = simple_analyses_to_run
                m = re.match(SIMPLE_PATTERN, file)
                assert m is not None
                param = float(m.group(1)) # this is gamma
            else:
                analyses_to_run = k_analyses_to_run
                m = re.match(K_PATTERN, file)
                assert m is not None
                param = int(m.group(1)) # this is k
            print(f'Loading {file}')
            g, sol_map = pickle.load(f)
            graphs[file] = nx.DiGraph(g)
            sol_maps[file] = sol_map
            for name, func in analyses_to_run.items():
                print(f'Running {name} on {file}')
                results[name] = func(graphs[file], sol_map, param)
                print(f'{file}\t{name} result: {results[name]}')
            with open(results_file, 'wb') as f:
                pickle.dump(results, f)

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
