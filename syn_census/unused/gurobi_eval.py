# Only going to evaluate ACCURACY = 1
import numpy as np
import pandas as pd
from build_micro_dist import read_microdata
from guided_solver import *
import matplotlib.pyplot as plt

def ind_above(cum_probs, pct):
    for i, p in enumerate(cum_probs):
        if p > pct:
            return i

def read_block_data():
    return pd.read_csv(get_block_out_file())

def plot_prob_over_solutions(solution, plot=False):
    # assumes solution is sorted in the order returned by Gurobi
    probs = []
    for sol in solution:
        probs.append(solution[sol])
    sorted_probs = sorted(probs, reverse=True)
    cum_probs = [sum(probs[:i]) for i in range(len(probs) + 1)]
    sorted_cum_probs = [sum(sorted_probs[:i]) for i in range(len(probs) + 1)]
    sorted_advantage = np.array(sorted_cum_probs) - np.array(cum_probs)
    if plot:
        plt.plot(cum_probs, label='our solution')
        plt.plot(sorted_cum_probs, label='optimal solution')
        plt.xlabel('solution number')
        plt.ylabel('probability mass captured')
        plt.legend()
        plt.show()
    area_between = sum(sorted_advantage) / len(sorted_advantage)
    return ind_above(cum_probs, .99), area_between

def evaluate_gurobi(row, dist):
    use_age = has_valid_age_data(row)
    if not use_age:
        return
    hhs = row_to_hhs(row)
    if len(hhs) == 0:
        return
    solution = solve(row, dist)
    min_ind, area_between = plot_prob_over_solutions(solution)
    return min_ind, area_between, len(solution)

if __name__ == '__main__':
    df = read_block_data()
    hh_dist = encode_hh_dist(read_microdata(get_micro_file()))
    num_trials = int(len(df) * .001)
    print(num_trials, 'trials')
    min_inds = []
    areas = []
    sizes = []
    sol_counts = []
    for i in range(num_trials):
        print('Trial', i, 'of', num_trials)
        ind = np.random.choice(range(len(df)))
        row = df.iloc[ind]
        answer = evaluate_gurobi(row, hh_dist)
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
    print('min_inds at .99', min_inds)
    print('total trials', len(min_inds))
    finite_min_inds = [a for a in min_inds if not np.isinf(a)]
    finite_areas = [a for a in areas if not np.isinf(a)]
    print('finite inds', len(finite_min_inds), len(finite_min_inds) / len(min_inds))
    print('mean finite area', np.mean(finite_areas))
    print('99 quantile:', max(finite_min_inds))
    # plt.scatter(sizes, sol_counts)
    # plt.show()
