# Only going to evaluate ACCURACY = 1
from ip_distribution import ip_solve_eval
import numpy as np
import pandas as pd
from census_utils import *
from knapsack_utils import *
from build_micro_dist import read_microdata
from guided_solver import recompute_probs, decode_solution
from encoding import *
import matplotlib.pyplot as plt

def ind_above(cum_probs, pct):
    for i, p in enumerate(cum_probs):
        if p > pct:
            return i

def read_block_data():
    return pd.read_csv(get_block_out_file())

def plot_prob_over_solutions(solution, dist):
    with_probs = recompute_probs(solution, dist)
    probs = []
    for sol in solution:
        probs.append(with_probs[sol])
    sorted_probs = sorted(probs, reverse=True)
    cum_probs = [sum(probs[:i]) for i in range(len(probs) + 1)]
    sorted_cum_probs = [sum(sorted_probs[:i]) for i in range(len(probs) + 1)]
    sorted_advantage = np.array(sorted_cum_probs) - np.array(cum_probs)
    plt.plot(cum_probs, label='our solution')
    plt.plot(sorted_cum_probs, label='optimal solution')
    plt.xlabel('solution number')
    plt.ylabel('probability mass captured')
    plt.legend()
    plt.show()
    area_between = sum(sorted_advantage) / len(sorted_advantage)
    # plt.plot(cum_probs)
    # plt.show()
    return ind_above(cum_probs, .99), area_between

def evaluate_gurobi(row, all_dists):
    use_age = has_valid_age_data(row)
    if not use_age:
        return
    hhs = row_to_hhs(row)
    if len(hhs) == 0:
        return
    # elif sum(hhs.values()) == 1 and use_age:
        # return
    hh_ordering = list(sorted(hhs.keys()))
    counts = get_race_counts(row)
    counts += get_over_18_counts(row)
    full_counts = counts + tuple(hhs[hh] for hh in hh_ordering)
    full_dist = {}
    n = len(hh_ordering)
    for i, hh in enumerate(hh_ordering):
        try:
            dist = all_dists[hh]
        except KeyError as e:
            print('Missing key', e)
            return
        i_tup = make_one_hot(i, n)
        for hh_counts, prob in dist.items():
            hh_encoded = hh_counts[:-1]
            over_18_tup = [0] * len(Race)
            over_18_tup[hh[0].value - 1] = hh_counts[-1]
            hh_encoded += tuple(over_18_tup)
            hh_encoded += i_tup
            full_dist[hh_encoded] = prob

    print('full_counts', full_counts)
    print('hhs', hhs)
    num_sols = 5000
    if row['H7X001'] > 100:
        return np.inf, np.inf, num_sols
    solution = ip_solve_eval(full_counts, full_dist, num_solutions=num_sols)
    if len(solution) == 0:
        return
    if len(solution) >= num_sols:
        print('Not all found')
        return np.inf, np.inf, len(solution)
    min_ind, area_between = plot_prob_over_solutions(solution, full_dist)
    return min_ind, area_between, len(solution)
    # solution = recompute_probs(solution, full_dist)

    # solution = normalize(decode_solution(solution, decode_1))
    # return solution

if __name__ == '__main__':
    df = read_block_data()
    all_dists, fallback_dist = read_microdata(get_micro_file())
    num_trials = int(len(df) * .05)
    print(num_trials, 'trials')
    min_inds = []
    areas = []
    sizes = []
    sol_counts = []
    for i in range(num_trials):
        print('Trial', i)
        ind = np.random.choice(range(len(df)))
        row = df.iloc[ind]
        answer = evaluate_gurobi(row, all_dists)
        if answer is not None:
            min_ind, area, num = answer
            print('Index', ind)
            print('99 quantile', min_ind)
            print('Area between', area)
            min_inds.append(min_ind)
            areas.append(area)
            sizes.append(row['H7X001'])
            sol_counts.append(num)
            print()
    min_inds = sorted(min_inds)
    print('min_inds', min_inds)
    print('total trials', len(min_inds))
    finite_min_inds = [a for a in min_inds if not np.isinf(a)]
    finite_areas = [a for a in areas if not np.isinf(a)]
    print('finite inds', len(finite_min_inds), len(finite_min_inds) / len(min_inds))
    print('mean finite area', np.mean(finite_areas))
    print('99 quantile:', max(finite_min_inds))
    # plt.scatter(sizes, sol_counts)
    # plt.show()
