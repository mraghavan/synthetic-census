import sys
import os
import pickle
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.special import comb
sys.path.append('../')
from syn_census.utils.config2 import ParserBuilder
from syn_census.preprocessing.build_micro_dist import read_microdata, t_to_str
from syn_census.utils.encoding import encode_hh_dist, encode_row, TYPES

parser_builder = ParserBuilder(
        {'state': True,
         'micro_file': True,
         'block_clean_file': True,
         'synthetic_output_dir': True,
         'num_sols': False,
         'task_name': True,
         })

def get_num_sols(results_dir: str, task_name: str):
    pattern = re.compile(rf'{task_name}_\d+_\d+.pkl')
    sol_counts = {}
    for fname in os.listdir(results_dir):
        if pattern.match(fname):
            with open(results_dir + fname, 'rb') as f:
                sols = pickle.load(f)
                for sol in sols:
                    if sol['level'] != 1:
                        continue
                    # if sol['id'] == bad_id:
                        # print(sol)
                    sol_counts[sol['id']] = len(sol['prob_list'])
    return sol_counts

def get_log_ub(counts: tuple, elements: tuple):
    """
    Returns the upper bound on the number of solutions for given counts
    and set of elements.
    """
    log_num_sols_type = 0.0
    for t in TYPES:
        r = getattr(counts, t_to_str(t))
        if r > 0:
            num_matching = len([e for e in elements if getattr(e, t_to_str(t)) > 0])
            log_num_sols_type += np.log(comb(r + num_matching - 1, num_matching - 1))

    r = getattr(counts, 'num_hh')
    # print(r)
    # print(len(elements))
    log_num_sols_hh = np.log(comb(r + len(elements) - 1, len(elements) - 1))
    assert r * np.log(len(elements) + r) >= log_num_sols_hh
    if log_num_sols_hh < log_num_sols_type:
        print('New bound better')
    return min(log_num_sols_type, log_num_sols_hh)

if __name__ == '__main__':
    parser_builder.parse_args()
    print(parser_builder.args)
    args = parser_builder.args
    bad_id = '001-021000-2126'
    sol_counts = get_num_sols(args.synthetic_output_dir, args.task_name)
    print(len(sol_counts))
    df = pd.read_csv(args.block_clean_file)
    df = df[df['H7X001'] > 0]
    print(df.head())
    dist = encode_hh_dist(read_microdata(args.micro_file))
    elements = tuple(dist.keys())
    all_counts = {}
    all_ubs = {}
    num_rows = 100
    # iterate over df in a random order
    for i, row in df.sample(n=num_rows, replace=False).iterrows():
        if row['identifier'] not in sol_counts:
            continue
        # if row['identifier'] != bad_id:
            # continue
        all_counts[row['identifier']] = encode_row(row)
        all_ubs[row['identifier']] = get_log_ub(all_counts[row['identifier']], elements)
        # print(all_counts[bad_id])
        # if row['identifier'] == bad_id:
            # break
    all_data = []
    for identifier in all_counts:
        all_data.append({
                'identifier': identifier,
                'log_ub': all_ubs[identifier],
                'num_sols': sol_counts[identifier],
                })
    df = pd.DataFrame(all_data)
    print(df['log_ub'].max())
    if not all(df['num_sols'] < np.exp(df['log_ub'])):
        # get the identifiers that have a problem
        bad_ids = df[~(df['num_sols'] < np.exp(df['log_ub']))]['identifier'].values
        print('Problem with upper bound', bad_ids)
    # assert all(df['num_sols'] < np.exp(df['log_ub'])), 'Problem with upper bound'
    solved_df = df[df['num_sols'] < 1000].copy()
    solved_df['log_ratio'] = np.log(np.exp(solved_df['log_ub']) / solved_df['num_sols'])
    plt.hist(solved_df['log_ratio'])
    plt.xlabel('log ratio of upper bound to number of solutions')
    plt.ylabel('Frequency')
    plt.show()

    plt.hist(solved_df['log_ub'])
    plt.xlabel('log upper bound')
    plt.ylabel('Frequency')
    plt.show()

    plt.scatter(solved_df['num_sols'], solved_df['log_ratio'])
    plt.xlabel('Number of solutions')
    plt.ylabel('log ratio of upper bound to number of solutions')
    plt.show()
    # bound_list = []
    # num_sol_list = []
    # for identifier in all_counts:
        # bound_list.append(all_ubs[identifier])
        # num_sol_list.append(sol_counts[identifier])
        # if np.exp(all_ubs[identifier]) < sol_counts[identifier]:
            # print('Problem with bound', identifier)
    # bound_array = np.array(bound_list)
    # num_sol_array = np.array(num_sol_list)
    # # histogram of ratio
    # plt.hist(np.log(np.exp(bound_array) / num_sol_array))
    # plt.xlabel('log ratio of upper bound to number of solutions')
    # plt.ylabel('Frequency')
    # plt.show()
    # plt.scatter(num_sol_list, bound_list)
    # plt.xlabel('Number of solutions')
    # plt.ylabel('Upper bound on log number of solutions')
    # plt.show()
