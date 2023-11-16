from guided_solver import *
from census_utils import *
from config2 import *
import pandas as pd
from build_micro_dist import read_microdata
import multiprocessing as mp
from mcmc_sampler import MCMCSampler
import pickle

parser_builder = ParserBuilder(
        {'state': True,
         'micro_file': True,
         'block_clean_file': True,
         'synthetic_output_dir': False,
         })

def read_block_data(block_clean_file):
    return pd.read_csv(block_clean_file)

def sample_from_sol(sol):
    keys, probs = zip(*sol.items())
    keys = list(keys)
    if len(keys) > 1:
        return keys[np.random.choice(range(len(keys)), p=probs)]
    else:
        return keys[0]

def get_n_mcmc_samples(counts, dist, n, num_iterations, k):
    sampler = MCMCSampler(dist, num_iterations=num_iterations, k=k)
    print('Running MCMC with num_iterations={}, k={}'.format(num_iterations, k))
    mcmc_dist = Counter()
    for i in range(n):
        if i % 100 == 0:
            print(i)
        mcmc_sol = tuple(sorted(sampler.mcmc_solve(counts)))
        mcmc_dist[mcmc_sol] += 1
    return mcmc_dist

def get_dist_over_n_random_samples(sol, n, num_attempts=1000):
    all_dists = []
    for _ in range (num_attempts):
        random_dist = Counter()
        for i in range(n):
            random_sol = sample_from_sol(sol)
            random_dist[random_sol] += 1
        all_dists.append(random_dist)
    return all_dists

def get_tvd(d1: Counter, d2: Counter):
    tvd = 0
    for k in d1:
        tvd += abs(d1[k] - d2[k])
    return tvd

def get_missing_mass(d1: Counter, d2: Counter):
    # d2 is the true distribution
    missing_mass = 0
    for k in d2:
        if k not in d1:
            missing_mass += d2[k]
    return missing_mass

def main():
    parser_builder.parse_args()
    print(parser_builder.args)
    parser_builder.verify_required_args()
    args = parser_builder.args
    SOLVER_PARAMS.num_sols = args.num_sols

    df = read_block_data(args.block_clean_file)
    # non-empty rows
    df = df[df['H7X001'] > 0]
    print(df.head())
    hh_dist = encode_hh_dist(read_microdata(args.micro_file))
    test_row = 3
    pool = mp.Pool(mp.cpu_count())
    num_samples = 1000
    print('num cpus', mp.cpu_count())
    print('total number of samples', num_samples * mp.cpu_count())
    row = df.iloc[test_row]
    sol = solve(row, hh_dist)

    ks = [2, 3, 4, 5]
    num_iterations = [1, 10, 100, 1000, 10000, 100000]

    combos_to_test = [(i, k) for i in num_iterations for k in ks]
    all_random_dists = get_dist_over_n_random_samples(sol, num_samples * mp.cpu_count())
    print('using', sum(all_random_dists[0].values()), 'random samples')
    counter_sol = Counter(sol)
    random_tvds = [get_tvd(Counter(normalize(d)), counter_sol) for d in all_random_dists]
    random_missing_masses = [get_missing_mass(Counter(normalize(d)), counter_sol) for d in all_random_dists]
    print('Got random TVDs', 'mean', np.mean(random_tvds), 'std', np.std(random_tvds))
    all_tvds = {}
    all_missing_masses = {}
    for num_iterations, k in combos_to_test:
        print('num_iterations', num_iterations, 'k', k)
        dists = pool.starmap(get_n_mcmc_samples, [(encode_row(row), hh_dist, num_samples, num_iterations, k)] * mp.cpu_count())
        mcmc_dist = sum(dists, Counter())
        mcmc_dist = Counter(normalize(mcmc_dist))
        all_tvds[(num_iterations, k)] = get_tvd(mcmc_dist, counter_sol)
        all_missing_masses[(num_iterations, k)] = get_missing_mass(mcmc_dist, counter_sol)
        print(all_tvds[(num_iterations, k)], all_missing_masses[(num_iterations, k)])

    with open (args.synthetic_output_dir + 'mcmc_results.pkl', 'wb') as f:
        pickle.dump(
                {
                    'random_tvds': random_tvds,
                    'random_missing_masses': random_missing_masses,
                    'all_tvds': all_tvds,
                    'all_missing_masses': all_missing_masses,
                    'num_random_samples': num_samples * mp.cpu_count(),
                    },
                f)

if __name__ == '__main__':
    main()
