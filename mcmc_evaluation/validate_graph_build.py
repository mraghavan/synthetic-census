import sys
import os
import pandas as pd
sys.path.append('../')
from syn_census.utils.config2 import ParserBuilder
from syn_census.utils.census_utils import get_num_hhs
from syn_census.utils.encoding import encode_row, encode_hh_dist
from syn_census.preprocessing.build_micro_dist import read_microdata
from syn_census.utils.ip_distribution import ip_solve
from build_graphs import all_files_exist, params, get_jobs_from_file
from analyze_graphs import results_templates

parser_builder = ParserBuilder(
        {'state': True,
         'micro_file': True,
         'block_clean_file': True,
         'mcmc_output_dir': True,
         'synthetic_output_dir': False,
         'num_sols': True,
         'num_tasks': True,
         })

def get_stats_on_row(row: pd.Series, dist: dict, num_sols):
    num_hh = get_num_hhs(row)
    counts = encode_row(row)
    sols = ip_solve(counts, dist, num_solutions=num_sols)
    return (num_hh, len(sols))

if __name__ == '__main__':
    parser_builder.parse_args()
    print(parser_builder.args)
    args = parser_builder.args

    in_file = os.path.join(args.mcmc_output_dir, 'sampled_block_ids.txt')
    all_jobs = {}
    for i in range(1, args.num_tasks + 1):
        all_jobs[i] = get_jobs_from_file(in_file, i, args.num_tasks)

    unknown_failures = set()
    known_failures = set()
    unknown_failed_jobs = set()
    known_failed_jobs = set()
    unfinished_jobs = set()

    for i in all_jobs:
        # TODO this will have to be changed with new format to:
        # fail_file = os.path.join(args.mcmc_output_dir, f'failures{i}.{args.num_tasks}.txt')
        fail_file = os.path.join(args.mcmc_output_dir, f'failures{i}.txt')
        if os.path.exists(fail_file):
            with open(fail_file) as f:
                for line in f:
                    known_failures.add(line.strip())
                    print('Known failure', line.strip(), 'in job', i)
        else:
            unfinished_jobs.add(i)
            print('Job', i, 'did not finish')

    for i, jobs in all_jobs.items():
        flag = False
        for jobs in jobs:
            identifier, job_type = jobs.split(',')
            if not all_files_exist(args.mcmc_output_dir, results_templates[job_type], identifier, params[job_type]):
                if identifier not in known_failures:
                    unknown_failures.add(identifier)
                    if i not in unfinished_jobs:
                        flag = True
                print('Missing', identifier, job_type, i)
        if flag:
            unknown_failed_jobs.add(i)

    print('Jobs with unknown failures:', sorted(unknown_failed_jobs))
    print('Unfinished jobs:', sorted(unfinished_jobs))
    print('Total', len(known_failures) + len(unknown_failures), 'failures')
    print('Known failures:', len(known_failures))
    all_failures = known_failures.union(unknown_failures)

    df = pd.read_csv(args.block_clean_file)
    # non-empty rows
    df = df[df['H7X001'] > 0]
    print(df.head())
    matching_df = df[df['identifier'].isin(all_failures)]
    print('Number of matching rows:', len(matching_df))
    dist = encode_hh_dist(read_microdata(args.micro_file))
    print('%10s\tnum_hhs\tnum_sols' % 'identifier')
    for _, row in matching_df.iterrows():
        num_hhs, num_sols = get_stats_on_row(row, dist, args.num_sols)
        print('%s\t%d\t%d' % (row['identifier'], num_hhs, num_sols))
