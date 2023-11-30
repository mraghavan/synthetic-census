import sys
import os
import pickle
import pandas as pd
sys.path.append('../')
from syn_census.utils.config2 import ParserBuilder
from syn_census.synthetic_data_generation.mcmc_sampler import SimpleMCMCSampler
from syn_census.mcmc_analysis.build_mcmc_graphs import build_graph, build_graph_simple
from syn_census.preprocessing.build_micro_dist import read_microdata
from syn_census.utils.encoding import encode_hh_dist, encode_row
from syn_census.utils.ip_distribution import ip_solve
from syn_census.utils.knapsack_utils import is_eligible

parser_builder = ParserBuilder(
        {'state': True,
         'micro_file': True,
         'block_clean_file': True,
         'mcmc_output_dir': True,
         'synthetic_output_dir': False,
         'num_sols': False,
         })


if __name__ == '__main__':
    parser_builder.parse_args()
    print(parser_builder.args)
    args = parser_builder.args
    fname = '{}_graph.pkl'
    df = pd.read_csv(args.block_clean_file)
    # non-empty rows
    df = df[df['H7X001'] > 0]
    print(df.head())
    dist = encode_hh_dist(read_microdata(args.micro_file))
    test_row = 3
    row = df.iloc[test_row]
    counts = encode_row(row)
    simple_dist = {k: v for k, v in dist.items() if is_eligible(k, counts)}
    sol = ip_solve(counts, dist, num_solutions=args.num_sols)
    sol_map = {v: i for i, v in enumerate(sol)}
    sol_map_copy = sol_map.copy()

    gammas = [0, 0.1, 0.2, 0.5, 1]
    ks = [2, 3]

    graphs_to_build = {f'simple_{gamma}': (lambda param=gamma: build_graph_simple(dist, counts, SimpleMCMCSampler(simple_dist, gamma=param), total_solutions=len(sol))) for gamma in gammas}
    graphs_to_build.update({f'k_{k}': (lambda param=k: build_graph(dist, sol, sol_map_copy, k=param)) for k in ks})

    for g, func in graphs_to_build.items():
        full_name = args.mcmc_output_dir + fname.format(g)
        if not os.path.exists(full_name):
            print('Building graph', g)
            graph = func()
            with open(full_name, 'wb') as f:
                pickle.dump(graph, f)
        else:
            print('Graph', g, 'already exists')
