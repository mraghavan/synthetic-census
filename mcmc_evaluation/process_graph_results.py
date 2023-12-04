import sys
import os
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
sys.path.append('../')
from syn_census.utils.config2 import ParserBuilder

simple_re = re.compile(r'(\d+-\d+-\d+)_(\d+(.\d+)?)_simple_results.pkl')
reduced_re = re.compile(r'(\d+-\d+-\d+)_(\d+(.\d+)?)_reduced_results.pkl')

parser_builder = ParserBuilder(
        {'state': True,
         'micro_file': True,
         'block_clean_file': True,
         'mcmc_output_dir': True,
         'num_sols': False,
         })

def extract_params(all_results, pattern, t=float):
    # returns a dictionary mapping parameter values to filenames
    params = {}
    for k in all_results:
        m = re.match(pattern, k)
        if m is not None:
            params[t(m.group(1))] = k
    return params

def plot_solution_density_and_mixing_time(all_results):
    simple_results = {k: v for k, v in all_results.items() if is_simple(k)}
    gammas = extract_params(simple_results, SIMPLE_PATTERN, float)
    gamma_list = sorted(gammas.keys())
    print(gammas)
    mixing_times_lbs = [simple_results[gammas[g]]['mixing_time'][0] for g in gamma_list]
    mixing_times_ubs = [simple_results[gammas[g]]['mixing_time'][1] for g in gamma_list]
    densities = [simple_results[gammas[g]]['solution_density'] for g in gamma_list]
    ax1: Axes = None #type: ignore
    _, ax1 = plt.subplots() #type: ignore
    # color_cycle = ax1._get_lines.color_cycle
    line1, = ax1.plot(gamma_list, [1/d for d in densities], label='1/solution density', marker='o')
    ax1.set_yscale('log') #type: ignore
    ax1.set_xlabel(r'$\gamma$')
    ax1.set_ylabel('1/solution density')
    ax2 = ax1.twinx()
    line2, = ax2.plot(gamma_list, mixing_times_lbs, label='mixing time LB', color='r', marker='o')
    line3, = ax2.plot(gamma_list, mixing_times_ubs, label='mixing time UB', color='g', marker='o')
    ax2.set_yscale('log')
    ax2.set_ylabel('mixing time')
    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper center')
    plt.tight_layout()
    plt.savefig('img/mixing_time_vs_solution_density.png')
    plt.show()

    conductances = [simple_results[gammas[g]]['conductance_ub'] for g in gamma_list]
    conductance_mixings = [(1/(2*c) - 1) for c in conductances]
    conductance_lbs = [mt*1/d for mt, d in zip(conductance_mixings, densities)]

    k_results = {k: v for k, v in all_results.items() if not is_simple(k)}
    ks = extract_params(k_results, K_PATTERN, int)
    k_list = sorted(ks.keys())

    expected_time_lbs = [mt * 1/d for mt, d in zip(mixing_times_lbs, densities)]
    expected_time_ubs = [mt * 1/d for mt, d in zip(mixing_times_ubs, densities)]
    k_mixing_time_lbs = [k_results[ks[k]]['mixing_time'][0] for k in k_list]
    k_mixing_time_ubs = [k_results[ks[k]]['mixing_time'][1] for k in k_list]
    _, ax1 = plt.subplots() #type: ignore
    lines = []
    l, = ax1.plot(k_list, k_mixing_time_lbs, label='complex LB', marker='o')
    lines.append(l)
    l, = ax1.plot(k_list, k_mixing_time_ubs, label='complex UB', marker='o')
    lines.append(l)
    ax1.set_yscale('log') #type: ignore
    ax1.set_xlabel('$k$')
    ax1.set_ylabel('expected #iterations')
    ax2 = ax1.twiny()
    l, = ax2.plot(gamma_list, expected_time_lbs, label='simple LB', color='r', marker='o')
    lines.append(l)
    l, = ax2.plot(gamma_list, expected_time_ubs, label='simple UB', color='g', marker='o')
    lines.append(l)
    l, = ax2.plot(gamma_list, conductance_lbs, label='conductance LB', color='y', marker='o')
    lines.append(l)
    ax2.set_xlabel(r'$\gamma$')
    labels = [line.get_label() for line in lines]
    plt.title("Expected iterations to generate a valid solution")
    ax1.legend(lines, labels)
    plt.tight_layout()
    ax1.grid(axis='y')
    plt.savefig('img/expected_time_k_gamma.png')
    plt.show()

def load_results(results_dir: str):
    simple_results = []
    reduced_results = []
    for file in os.listdir(results_dir):
        m1 = simple_re.match(file)
        m2 = reduced_re.match(file)
        if m1 is not None:
            with open(os.path.join(results_dir, file), 'rb') as f:
                results = pickle.load(f)
                results['identifier'] = m1.group(1)
                results['gamma'] = float(m1.group(2))
                results['mixing_time_lb'] = results['mixing_time_bounds'][0]
                results['mixing_time_ub'] = results['mixing_time_bounds'][1]
                del results['mixing_time_bounds']
                simple_results.append(results)
        elif m2 is not None:
            with open(os.path.join(results_dir, file), 'rb') as f:
                results = pickle.load(f)
                results['identifier'] = m2.group(1)
                results['k'] = int(m2.group(2))
                results['mixing_time_lb'] = results['mixing_time_bounds'][0]
                results['mixing_time_ub'] = results['mixing_time_bounds'][1]
                del results['mixing_time_bounds']
                reduced_results.append(results)
    simple_df = pd.DataFrame(simple_results)
    reduced_df = pd.DataFrame(reduced_results)
    return simple_df, reduced_df

def add_exp_mixing_time(df: pd.DataFrame):
    df['exp_mixing_time_lb'] = df['mixing_time_lb'] / df['solution_density']
    df['exp_mixing_time_ub'] = df['mixing_time_ub'] / df['solution_density']

def get_min_mixing_time_df(df: pd.DataFrame, lb_type: str):
    return df.loc[df.groupby(['identifier'])[lb_type].idxmin()]

def scatter_mixing_times(simple_df: pd.DataFrame, reduced_df: pd.DataFrame):
    plt.scatter(simple_df['num_solutions'], simple_df['exp_mixing_time_lb'], label='simple LB')
    plt.scatter(reduced_df['num_solutions'], reduced_df['mixing_time_lb'], label='reduced LB')
    plt.xlabel('number of solutions')
    plt.ylabel('expected mixing time')
    plt.yscale('log')
    plt.legend()
    plt.show()

def scatter_mixing_time_ratios(simple_df: pd.DataFrame, reduced_df: pd.DataFrame):
    # find common identifiers
    common_ids = set(simple_df['identifier']).intersection(set(reduced_df['identifier']))
    simple_df = simple_df.loc[simple_df['identifier'].isin(common_ids)].copy().reset_index(drop=True)
    reduced_df = reduced_df.loc[reduced_df['identifier'].isin(common_ids)].copy().reset_index(drop=True)
    # sort both by identifier
    simple_df.sort_values(by=['identifier'], inplace=True)
    reduced_df.sort_values(by=['identifier'], inplace=True)
    # scatter num_solutions by ratio of mixing times
    plt.scatter(simple_df['num_solutions'], simple_df['exp_mixing_time_lb'] / reduced_df['mixing_time_lb'])
    plt.xlabel('number of solutions')
    plt.ylabel('ratio of expected mixing times')
    plt.yscale('log')
    plt.show()

def connectivity_analysis(reduced_df: pd.DataFrame):
    print('Fraction connected', reduced_df['is_connected'].value_counts()[True] / len(reduced_df))

def plot_simple_column(simple_df: pd.DataFrame, column: str):
    unique_ids = simple_df['identifier'].unique()
    for ide in unique_ids:
        df = simple_df.loc[simple_df['identifier'] == ide].copy()
        df.sort_values(by=['gamma'], inplace=True)
        plt.plot(df['gamma'], df[column], color='b', marker='o')
    plt.xlabel(r'$\gamma$')
    plt.ylabel(column)
    plt.yscale('log')
    plt.show()

if __name__ == '__main__':
    parser_builder.parse_args()
    print(parser_builder.args)
    args = parser_builder.args

    simple_df, reduced_df = load_results(args.mcmc_output_dir)
    add_exp_mixing_time(simple_df)
    opt_simple_df = get_min_mixing_time_df(simple_df, 'exp_mixing_time_lb')
    opt_reduced_df = get_min_mixing_time_df(reduced_df, 'mixing_time_lb')
    # scatter_mixing_times(opt_simple_df, opt_reduced_df)
    scatter_mixing_time_ratios(opt_simple_df, opt_reduced_df)
    connectivity_analysis(reduced_df)
    # plot_simple_column(simple_df, 'solution_density')
    # plot_simple_column(simple_df, 'exp_mixing_time_lb')

    # get counts of each identifier
    print(simple_df['identifier'].value_counts())
