import sys
import os
import pickle
import numpy as np
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

def get_relevant_blocks(results_dir: str, task_name: str, max_num_sols: int):
    """
    Returns a list of block ids that have between min_sols and max_sols solutions
    and their corresponding solution counts.
    """
    pattern = re.compile(rf'{task_name}_\d+_\d+.pkl')
    sol_lens = OrderedDict()
    for fname in os.listdir(results_dir):
        if pattern.match(fname):
            with open(results_dir + fname, 'rb') as f:
                sols = pickle.load(f)
                for sol in sols:
                    if sol['level'] == 1 and len(sol['prob_list']) < max_num_sols:
                        sol_lens[sol['id']] = len(sol['sol'])
    return sol_lens

if __name__ == '__main__':
    parser_builder.parse_args()
    print(parser_builder.args)
    args = parser_builder.args
    min_hh = 5
    max_hh = 20
    sample_size = args.num_samples
    sol_lens = get_relevant_blocks(args.synthetic_output_dir, args.task_name, args.num_sols)
    np.random.seed(123)
    filtered_sol_ids = sorted([k for k, v in sol_lens.items() if min_hh <= v <= max_hh])
    filtered_sol_lens = {k: v for k, v in sol_lens.items() if min_hh <= v <= max_hh}
    print('Number of blocks with between {} and {} hhs: {}'.format(min_hh, max_hh, len(filtered_sol_lens)))
    random_permutation = np.random.permutation(filtered_sol_ids)
    random_sample_of_block_ids = random_permutation[:sample_size]
    print(random_sample_of_block_ids)
    out_file = os.path.join(args.mcmc_output_dir, 'sampled_block_ids.txt')
    with open(out_file, 'w') as f:
        for block_id in random_sample_of_block_ids:
            f.write(block_id + '\n')
