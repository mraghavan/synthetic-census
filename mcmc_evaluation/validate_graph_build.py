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
        jobs = get_jobs_from_file(in_file, i, args.num_tasks)
        l = {}
        for job in jobs:
            identifier, job_type = job.split(',')
            l[identifier] = job_type
        all_jobs[i] = l

    unknown_failures = set()
    known_failures = set()
    known_failures_with_type = {}
    unknown_failed_jobs = set()
    known_failed_jobs = set()
    unfinished_jobs = set()
    all_failures_with_type = set()

    for i, jobs in all_jobs.items():
        # TODO this will have to be changed with new format to:
        # fail_file = os.path.join(args.mcmc_output_dir, f'failures{i}.{args.num_tasks}.txt')
        fail_file = os.path.join(args.mcmc_output_dir, f'failures{i}.txt')
        if os.path.exists(fail_file):
            with open(fail_file) as f:
                for line in f:
                    identifier = line.strip()
                    known_failures.add((identifier, jobs[identifier]))
                    known_failures_with_type[identifier] = jobs[identifier]
                    print('Known failure', identifier, 'in job', i, jobs[identifier])
        else:
            unfinished_jobs.add(i)
        flag = False
        for identifier, job_type in jobs.items():
            if not all_files_exist(args.mcmc_output_dir, results_templates[job_type], identifier, params[job_type]):
                if (identifier, job_type) not in known_failures:
                    unknown_failures.add((identifier, job_type))
                    if i not in unfinished_jobs:
                        flag = True
                print('Missing', identifier, job_type, i)
                all_failures_with_type.add((identifier, job_type))
        if flag:
            unknown_failed_jobs.add(i)

    # for i, jobs in all_jobs.items():
        # flag = False
        # for identifier, job_type in jobs.items():
            # if not all_files_exist(args.mcmc_output_dir, results_templates[job_type], identifier, params[job_type]):
                # if identifier not in known_failures:
                    # unknown_failures.add(identifier)
                    # if i not in unfinished_jobs:
                        # flag = True
                # print('Missing', identifier, job_type, i)
                # all_failures_with_type.add((identifier, job_type))
        # if flag:
            # unknown_failed_jobs.add(i)

    print('Jobs with unknown failures:', sorted(unknown_failed_jobs))
    print('Unfinished jobs:', sorted(unfinished_jobs), f'({len(unfinished_jobs)} total)')
    print('Total', len(all_failures_with_type), 'failures')
    print('Known failures:', len(known_failures))
    all_failures = known_failures.union(unknown_failures)

    df = pd.read_csv(args.block_clean_file)
    # non-empty rows
    df = df[df['H7X001'] > 0]
    print(df.head())
    # matching_df = df[df['identifier'].isin(all_failures)]
    # print('Number of matching rows:', len(matching_df))
    dist = encode_hh_dist(read_microdata(args.micro_file))
    print('identifier\tjob_type\tnum_hhs\tnum_sols')
    for identifier, job_type in all_failures_with_type:
        matching_row = df[df['identifier'] == identifier]
        num_hhs, num_sols = get_stats_on_row(matching_row.iloc[0], dist, args.num_sols)
        print('%s\t%10s\t%d\t%d' % (identifier, job_type, num_hhs, num_sols))
    # for _, row in matching_df.iterrows():
        # num_hhs, num_sols = get_stats_on_row(row, dist, args.num_sols)
        # print('%s\t%d\t%d' % (row['identifier'], num_hhs, num_sols))
