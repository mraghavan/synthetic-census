import sys
import os
import pickle
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import re
from collections import OrderedDict
sys.path.append('../')
from syn_census.utils.config2 import ParserBuilder

parser_builder = ParserBuilder(
        {'state': True,
         'micro_file': True,
         'block_clean_file': True,
         'mcmc_output_dir': True,
         'synthetic_output_dir': False,
         'num_sols': True,
         'task_name': True,
         'num_samples': True,
         })

def get_relevant_blocks(results_dir: str, task_name: str):
    pattern = re.compile(rf'{task_name}_\d+_\d+.pkl')
    results_list = []
    for fname in os.listdir(results_dir):
        if pattern.match(fname):
            with open(results_dir + fname, 'rb') as f:
                sols = pickle.load(f)
                for sol in sols:
                    results_list.append({
                        'identifier': sol['id'],
                        'num_sols': len(sol['prob_list']),
                        'num_elements': len(sol['sol']),
                        'level': sol['level']
                        })
    results_df = pd.DataFrame(results_list)
    return results_df

# def plot_results(results_df: pd.DataFrame, max_sols: int):
    # # plot num_elements vs fraction with num_sols < max_sols
    # elements = []
    # fractions = []
    # sizes = []
    # for num_elements in results_df['num_elements'].unique():
        # df = results_df[results_df['num_elements'] == num_elements]
        # elements.append(num_elements)
        # fractions.append(sum(df['num_sols'] < max_sols) / len(df))
        # sizes.append(len(df))
    # plt.scatter(elements, fractions, s=np.array(sizes)/8)
    # plt.xlabel('Number of households')
    # plt.ylabel(f'Fraction of blocks with < {max_sols} solutions')
    # plt.show()

def get_common_sample(filtered_df: pd. DataFrame, num_samples: int):
    # get a sample of num_samples common blocks
    np.random.seed(123)
    filtered_sol_ids = sorted(list(filtered_df['identifier']))
    random_permutation = np.random.permutation(filtered_sol_ids)
    random_sample_of_block_ids = random_permutation[:num_samples]
    return random_sample_of_block_ids

def generate_sample(df: pd. DataFrame, **kwargs):
    df = df[(kwargs['min_hh'] <= df['num_elements']) &\
            (df['num_elements'] <= kwargs['max_hh'])]
    np.random.seed(123)
    filtered_sol_ids = sorted(list(df['identifier']))
    random_permutation = np.random.permutation(filtered_sol_ids)
    random_sample_of_block_ids = random_permutation[:kwargs['num_samples']]
    return [(identifier, kwargs['type']) for identifier in random_sample_of_block_ids]

if __name__ == '__main__':
    parser_builder.parse_args()
    print(parser_builder.args)
    args = parser_builder.args

    sample_size = args.num_samples
    results_df = get_relevant_blocks(args.synthetic_output_dir, args.task_name)

    samples = [
            {'min_hh': 5, 'max_hh': 20, 'num_samples': sample_size, 'type': 'gibbs'},
            {'min_hh': 5, 'max_hh': 20, 'num_samples': sample_size, 'type': 'reduced'},
            {'min_hh': 14, 'max_hh': 35, 'num_samples': sample_size, 'type': 'reduced'},
            ]

    print('Total number of blocks', len(results_df))
    filtered_df = results_df[(results_df['level'] == 1) & (results_df['num_sols'] < args.num_sols)]

    largest_max_hh = max([sample['max_hh'] for sample in samples])
    print('Overall eligible', len(filtered_df[filtered_df['num_elements'] <= largest_max_hh]) / len(results_df))

    full_sample = []
    for sample in samples:
        full_sample += generate_sample(filtered_df, **sample)

    out_file = os.path.join(args.mcmc_output_dir, 'sampled_block_ids.txt')
    df_out_file = os.path.join(args.mcmc_output_dir, 'all_blocks.csv')
    print('Writing df to', df_out_file)
    filtered_df.to_csv(df_out_file, index=False)
    with open(out_file, 'w') as f:
        print('Writing %d jobs to' % len(full_sample), out_file)
        for identifier, sample_type in full_sample:
            f.write(f'{identifier},{sample_type}\n')
