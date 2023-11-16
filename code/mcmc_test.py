from guided_solver import *
from census_utils import *
from config2 import *
import pandas as pd
from build_micro_dist import read_microdata
import random
import sys
import multiprocessing as mp
# from knapsack_sampler import knapsack_solve
from mcmc_sampler import MCMCSampler
import pickle
# import psutil
parser_builder = ParserBuilder(
        {'state': True,
         'micro_file': True,
         'block_clean_file': True,
         'synthetic_output_dir': False,
         'num_sols': False,
         'task': False,
         'num_tasks': False,
         'task_name': False,
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
    print(mcmc_dist)
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
    num_samples = 10
    row = df.iloc[test_row]
    sol = solve(row, hh_dist)
    # ks = [2, 3, 4, 5]
    # num_iterations = [1, 10, 100, 1000, 10000, 100000]

    ks = [2, 3]
    num_iterations = [0, 1]
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
        # mcmc_sol = tuple(sorted(sampler.mcmc_solve(encode_row(row))))
        # mcmc_dist[mcmc_sol] += 1
        mcmc_dist = Counter(normalize(mcmc_dist))
        # random_dist = normalize(random_dist)
        all_tvds[(num_iterations, k)] = get_tvd(mcmc_dist, counter_sol)
        all_missing_masses[(num_iterations, k)] = get_missing_mass(mcmc_dist, counter_sol)
        print(all_tvds[(num_iterations, k)], all_missing_masses[(num_iterations, k)])
        # for hhs in mcmc_dist:
            # print(mcmc_dist[hhs], sol[hhs])
            # tvd += abs(mcmc_dist[hhs] - sol[hhs])
        # for hhs in random_dist:
            # random_tvd += abs(random_dist[hhs] - sol[hhs])
        # vs = []
        # for hhs in sol:
            # if hhs not in mcmc_dist:
                # vs.append(sol[hhs])
        # print(sorted(vs, reverse=True)[:10])
        # print('missing mass:' , sum(vs))
        # print('MCMC TVD', tvd)

    with open ('./mcmc_results.pkl', 'wb') as f:
        pickle.dump({
            'random_tvds': random_tvds,
            'random_missing_masses': random_missing_masses,
            'all_tvds': all_tvds,
            'all_missing_masses': all_missing_masses,
            'num_random_samples': num_samples * mp.cpu_count(),
            },
                    f)

        # print('Random TVD', random_tvd)
    # for i in range(num_samples * mp.cpu_count()):
        # random_sol = sample_from_sol(sol)
        # random_dist[random_sol] += 1
        # if i % 10 == 0:
            # print(i)

    # dists = pool.starmap(get_n_mcmc_samples, [(encode_row(row), hh_dist, num_samples, num_iterations, k)] * mp.cpu_count())
    # mcmc_dist = sum(dists, Counter())
        # # mcmc_sol = tuple(sorted(sampler.mcmc_solve(encode_row(row))))
        # # mcmc_dist[mcmc_sol] += 1
    # mcmc_dist = normalize(mcmc_dist)
    # random_dist = normalize(random_dist)
    # tvd = 0
    # random_tvd = 0
    # for hhs in mcmc_dist:
        # print(mcmc_dist[hhs], sol[hhs])
        # tvd += abs(mcmc_dist[hhs] - sol[hhs])
    # for hhs in random_dist:
        # random_tvd += abs(random_dist[hhs] - sol[hhs])
    # vs = []
    # for hhs in sol:
        # if hhs not in mcmc_dist:
            # vs.append(sol[hhs])
    # print(sorted(vs, reverse=True)[:10])
    # print('missing mass:' , sum(vs))
    # print('MCMC TVD', tvd)
    # print('Random TVD', random_tvd)

if __name__ == '__main__':
    main()

# Results from 1000 iterations, k=4
# 0.037 0.03121646035210474
# 0.207 0.2166816857556673
# 0.256 0.25006390268818246
# 0.138 0.11718303380880735
# 0.082 0.0749756095584433
# 0.055 0.0572129236311069
# 0.041 0.040021362219553956
# 0.018 0.017308528777916648
# 0.044 0.04957529897810555
# 0.013 0.01204170112561645
# 0.01 0.006125911532616289
# 0.008 0.012070095097283967
# 0.034 0.04016232473747479
# 0.001 0.0008249349260761519
# 0.001 5.738902114693842e-05
# 0.002 0.0016537601959956294
# 0.001 0.000300972902156959
# 0.009 0.01355806585229372
# 0.008 0.01540895217795312
# 0.008 0.01311506040255677
# 0.004 0.0021606917133611334
# 0.009 0.005652123025387443
# 0.004 0.003101993441892539
# 0.003 0.0074180286178422
# 0.001 0.00024977895637169894
# 0.003 0.0009266308145421554
# 0.001 0.001684642497214131
# 0.001 0.0026400492687198764
# 0.001 3.3274908121801675e-05
# TVD 0.1096290670479247
# TVD from true random: 0.071, 0.102, 0.077, 0.103, 0.114, 0.090
