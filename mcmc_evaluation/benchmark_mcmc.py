import sys
import random
import numpy as np
import os
import pandas as pd
from time import time
import pickle
sys.path.append('../')
from sample_identifiers import get_relevant_blocks
from syn_census.synthetic_data_generation.mcmc_sampler import MCMCSampler, SimpleMCMCSampler, run_test
from syn_census.utils.config2 import ParserBuilder
from syn_census.preprocessing.build_micro_dist import read_microdata
from syn_census.utils.encoding import encode_hh_dist, encode_row
from syn_census.utils.ip_distribution import ip_solve

parser_builder = ParserBuilder(
        {'state': True,
         'micro_file': True,
         'block_clean_file': True,
         'synthetic_output_dir': False,
         'num_sols': True,
         'task_name': True,
         })

if __name__ == '__main__':
    parser_builder.parse_args()
    print(parser_builder.args)
    args = parser_builder.args
    np.random.seed(123)
    random.seed(123)

    num_iter = 100000
    num_attempts = 20
    k = 3
    gamma = 0.8

    # Load Census data
    results_df = get_relevant_blocks(args.synthetic_output_dir, args.task_name)
    eligible_df = results_df[
            (results_df['level'] == 1) &\
            (results_df['num_elements'] > k) &\
            (results_df['num_sols'] > 1)]
    ilp_eligible_df = results_df[
            (results_df['level'] == 1) &\
            (results_df['num_sols'] == args.num_sols)]
    df = pd.read_csv(args.block_clean_file)
    # non-empty rows
    df = df[df['H7X001'] > 0]

    
    # dist_pickle_file = './dist.pkl'
    # if os.path.exists(dist_pickle_file):
        # print('Loading from file')
        # with open(dist_pickle_file, 'rb') as f:
            # dist = pickle.load(f)
    # else:
    dist = encode_hh_dist(read_microdata(args.micro_file))
    print(len(dist), 'households')
        # with open(dist_pickle_file, 'wb') as f:
            # pickle.dump(dist, f)
    random_blocks = eligible_df.sample(num_attempts)['identifier'].values
    ilp_random_blocks = ilp_eligible_df.sample(num_attempts)['identifier'].values

    ilp_times = []
    for i, random_block in enumerate(ilp_random_blocks):
        print('trial {}'.format(i))
        row = df[df['identifier'] == random_block].iloc[0]
        start = time()
        sols = ip_solve(encode_row(row), dist, num_solutions=args.num_sols)
        end = time()
        print(len(sols))
        print('Time taken: {}'.format(end - start))
        ilp_times.append(end - start)

    unit = 100000
    simple_times = []
    for i, random_block in enumerate(random_blocks):
        print('trial {}'.format(i))
        row = df[df['identifier'] == random_block].iloc[0]
        sampler = SimpleMCMCSampler(dist, gamma=gamma)
        print(encode_row(row))
        start = time()
        sampler.mcmc_iterate(encode_row(row), num_iterations=num_iter, cache={})
        end = time()
        print('Time taken: {}'.format(end - start))
        simple_times.append(end - start)

    reduced_times = []
    for i, random_block in enumerate(random_blocks):
        print('trial {}'.format(i))
        row = df[df['identifier'] == random_block].iloc[0]
        sampler = MCMCSampler(dist, num_iterations=num_iter, k=k)
        MCMCSampler.ip_solve_cached.cache_clear()
        print(encode_row(row))
        start = time()
        sampler.mcmc_solve(encode_row(row))
        end = time()
        print('Time taken: {}'.format(end - start))
        reduced_times.append(end - start)

    print('ILP')
    print('Average time taken {}'.format(sum(ilp_times) / len(ilp_times)))
    print('Standard deviation: {}'.format(np.std(ilp_times)))
    print()

    print('Simple')
    print('Average time taken per {} iterations: {}'.format(unit, (sum(simple_times) / len(simple_times) / num_iter * unit)))
    print('Standard deviation: {}'.format(np.std(simple_times)))
    print()

    print('Reduced')
    print('Average time taken per {} iterations: {}'.format(unit, (sum(reduced_times) / len(reduced_times) / num_iter * unit)))
    print('Standard deviation: {}'.format(np.std(reduced_times)))
