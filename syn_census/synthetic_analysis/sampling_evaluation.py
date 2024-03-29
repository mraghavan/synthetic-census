import os
import pickle
import numpy as np
from collections import OrderedDict, Counter
import sys
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import lognorm
import re
import matplotlib.pyplot as plt
from matplotlib import ticker
# from ..utils.config2 import ParserBuilder
from .representativeness import add_tex_var, print_all_tex_vars

def evaluate_coverage(task_name: str, synthetic_output_dir: str, num_sols: int):
    funcs = OrderedDict({
            'total_solutions': lambda pl, sol: len(pl),
            'last_mass': lambda pl, sol: pl[-1],
            'last_50_mass': lambda pl, sol: sum(pl[-50:]),
            'coverage_at_50': lambda pl, sol: sum(pl[:50]),
            'coverage_at_500': lambda pl, sol: sum(pl[:500]),
            'sorted_coverage_at_500': lambda pl, sol: sum(sorted(pl, reverse=True)[:500]),
            'num_hh': lambda pl, sol: len(sol),
            })
    d = synthetic_output_dir
    big_results_df = pd.DataFrame(columns=list(funcs.keys()))
    unsorted_probs = np.zeros((num_sols,))
    sorted_probs = np.zeros((num_sols,))
    for fname in sorted(os.listdir(d)):
        if re.match(task_name + '[0-9]+_[0-9]+.pkl', fname):
            print('Reading from', d+fname, file=sys.stderr)
            with open(d + fname, 'rb') as f:
                result_list = pickle.load(f)
                # Dataframe with dimensions (len(funcs), len(result_list)) with columns given by the keys in funcs filled with zeros
                results_df = pd.DataFrame(np.zeros((len(result_list), len(funcs) + 2)), columns=(list(funcs.keys())) + ['level', 'age'])
                for i, results in enumerate(result_list):
                    results_df.loc[i] = [f(results['prob_list'], results['sol']) for f in funcs.values()] + [results['level'], results['age']]
                    if len(results['prob_list']) == num_sols:
                        unsorted_probs += results['prob_list']
                        sorted_probs += sorted(results['prob_list'], reverse=True)
            big_results_df = pd.concat([big_results_df, results_df], ignore_index=True)
    return big_results_df, unsorted_probs, sorted_probs

def to_cumulative(probs):
    new_probs = [0] * len(probs)
    new_probs[0] = probs[0]
    for i in range(1, len(probs)):
        new_probs[i] = new_probs[i-1] + probs[i]
    return new_probs

def fit_truncated_pl(xs, ys):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    # Fit log-normal distribution to CCDF data
    # params = lognorm.fit(xs)

    # Get the fitted parameters
    # shape, loc, scale = params

    # # Generate a smooth curve using the fitted parameters
    # smooth_xs = np.linspace(min(xs), max(xs), 100)
    # fit_ccdf = 1 - lognorm.cdf(smooth_xs, shape, loc=loc, scale=scale)

    # # Plot the original CCDF data and the fitted log-normal distribution
    # plt.step(xs, ys / ys.max(), label='Original CCDF', where='post')
    # plt.plot(smooth_xs, fit_ccdf, label=f'Log-Normal Fit: shape={shape:.2f}, scale={scale:.2f}', color='red')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('X')
    # plt.ylabel('CCDF (P(X >= x))')
    # plt.legend()
    # plt.show()

    # x0 = xs[0]

    # def F(x, a, b):
        # return np.power((x-x0)+1.0, -a) * np.exp(-b*(x-x0))

    # def logF(x, a, b):
        # return -a*np.log((x-x0)+1.0) + (-b*(x-x0))
    def power_law(x, a, b):
        return a * np.power(x, b)
    def power_law_ccdf(x, a, b):
        return a * np.power(x, -b)
    

    # Fit the power-law function to the data
    params, covariance = curve_fit(power_law_ccdf, xs, ys)


    # Get the fitted parameters
    a, b = params
    smooth_xs = np.linspace(min(xs), max(xs), 100)
    # fit_ys = power_law(smooth_xs, a, b)
    fit_ccdf = power_law_ccdf(smooth_xs, a, b)

    plt.scatter(xs, ys, label='Original Data')
    # plt.step(xs, ys, label='Original Cumulative Data', where='post')
    plt.plot(smooth_xs, fit_ccdf, label=f'Power law fit', color='red')
    # dashed vertical line at x = max(xs)
    plt.axvline(x=max(xs), color='black', linestyle='--')
    # plt.plot(smooth_xs, fit_ys, label=f'Power Law Fit: a={a:.2f}, b={b:.2f}', color='red')
    plt.xscale('log')
    plt.yscale('log')
    # plt.ylim(bottom=0.0000001)
    plt.xlabel(r'$s$')
    plt.ylabel(r'Fraction of blocks with $|\mathcal{X}_b| \geq s$')
    # TODO don't use sci notation for y axis

    plt.legend()
    plt.tight_layout()
    plt.gca().yaxis.set_minor_formatter(ticker.FormatStrFormatter("%.1f"))
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    # plt.gca().get_yaxis().set_major_formatter(FormatStrFormatter('%g'))
    # plt.gca().get_yaxis().set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    # plt.gca().get_yaxis().set_major_formatter(ScalarFormatter(useMathText=False))
    # plt.ticklabel_format(style='plain', axis='both')
    # plt.ticklabel_format(style='plain', axis='both')



    # popt, pcov = curve_fit(F, xs, ys)
    # print(popt)

    # plt.plot(xs, ys, 'b*', label='data')
    # plt.plot(xs, logF(xs, *popt), 'g-', label='fit')
    # plt.show()

def print_results(
        state: str,
        synthetic_output_dir: str,
        num_sols: int,
        task_name: str=''):
    state = state.upper()
    if task_name != '':
        task_name += '_'
    fname = os.path.join(synthetic_output_dir, task_name + 'sampling_results.pkl')
    results_df, unsorted_probs, sorted_probs = None, None, None
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            results_df, unsorted_probs, sorted_probs = pickle.load(f)
    else:
        results_df, unsorted_probs, sorted_probs = evaluate_coverage(task_name, synthetic_output_dir, num_sols)
        with open(fname, 'wb') as f:
            pickle.dump((results_df, unsorted_probs, sorted_probs), f)
    print(results_df.describe(), file=sys.stderr)

    original_results_df = results_df.copy()

    age_accurate = results_df[results_df['age'] == True]
    print('age accurate fraction', len(age_accurate)/len(results_df), file=sys.stderr)
    
    results_df = results_df[results_df['age'] == True]

    level_1 = results_df[results_df['level'] == 1]
    level_2 = results_df[results_df['level'] != 1]
    print('level 1', len(level_1), file=sys.stderr)
    print('level > 1', len(level_2), file=sys.stderr)
    print('level 1 fraction', len(level_1)/len(results_df), file=sys.stderr)

    results_df = results_df[results_df['level'] == 1]

    solution_counts = results_df['total_solutions'].values
    results_df['total_solutions'].hist()
    plt.xlabel('Number of solutions')
    plt.savefig(task_name + 'total_solutions.png')
    plt.clf()
    # Count the number of rows where total_solutions < num_sols
    
    ub = 50
    results_df['num_hh'].describe()
    num_hh = results_df[results_df['num_hh'] <= ub]['num_hh'].values
    plt.hist(num_hh, bins=ub)
    plt.xlabel('Number of households')
    plt.xlim(left=0, right=ub)
    plt.show()
    print(f'Greater than {ub}', len(results_df[results_df['num_hh'] > ub])/len(results_df))

    # histogram = Counter()
    # for count in results_df['num_hh'].values:
        # histogram[count] += 1
    # # del histogram[num_sols]
    # values, counts = zip(*histogram.items())

    # # Sort values in descending order
    # sorted_values = np.array(sorted(values, reverse=True))

    # # Calculate cumulative sums of counts
    # vals = np.array([histogram[val] for val in sorted_values])
    # vals = vals / vals.sum()
    # cumulative_sums = np.cumsum(vals)
    # fit_truncated_pl(sorted_values, cumulative_sums)
    # # plt.savefig(f'./img/{task_name}power_law_solution_count.png', dpi=300)
    # plt.show()



    # histogram = Counter({k: 0 for k in range(1, num_sols+1)})
    histogram = Counter()
    for count in solution_counts:
        histogram[count] += 1
    # del histogram[num_sols]
    values, counts = zip(*histogram.items())

    # Sort values in descending order
    sorted_values = np.array(sorted(values, reverse=True))

    # Calculate cumulative sums of counts
    vals = np.array([histogram[val] for val in sorted_values])
    vals = vals / vals.sum()
    cumulative_sums = np.cumsum(vals)
    fit_truncated_pl(sorted_values, cumulative_sums)
    print('Saving to', f'./img/{task_name}power_law_solution_count.png')
    plt.savefig(f'./img/{task_name}power_law_solution_count.png', dpi=300)
    plt.show()
    

    add_tex_var('NumSolutions', num_sols)
    num_blocks = results_df.shape[0]

    add_tex_var('NumBlocks', num_blocks)
    fully_solved = int((results_df['total_solutions'] < num_sols).sum())
    add_tex_var('Coverage', fully_solved)
    add_tex_var('CoveragePercent', fully_solved / num_blocks * 100, precision=2)

    avg_last_mass = results_df[results_df['total_solutions'] == num_sols]['last_50_mass'].mean()
    add_tex_var('AvgLastMass', avg_last_mass*100, precision=2)
    unfinished_df = results_df[results_df['total_solutions'] == num_sols]
    diff = unfinished_df['sorted_coverage_at_500'] - unfinished_df['coverage_at_500']
    add_tex_var('AvgDiff', diff.mean()*100, precision=2)
    unsorted_probs = to_cumulative(unsorted_probs/len(unfinished_df))
    sorted_probs = to_cumulative(sorted_probs/len(unfinished_df))
    plt.plot(unsorted_probs, label='Heuristic')
    plt.plot(sorted_probs, label='Optimal')
    plt.xlabel('Number of solutions')
    plt.ylabel('Cumulative probability')
    plt.savefig(task_name + 'cumulative_probs.png')
    
    print_all_tex_vars(state)
