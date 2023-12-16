import sys
import os
import numpy as np
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

AXIS_LABELS = {
        'solution_density': 'Solution density',
        'mixing_time': 'Mixing time',
        'num_sols': 'Number of solutions',
        'mixing_time_lb': 'Lower bound on mixing time',
        'mixing_time_ub': 'Upper bound on mixing time',
        'num_elements': 'Number of items in each solution',
        }

def extract_params(all_results, pattern, t=float):
    # returns a dictionary mapping parameter values to filenames
    params = {}
    for k in all_results:
        m = re.match(pattern, k)
        if m is not None:
            params[t(m.group(1))] = k
    return params

def load_results(results_dir: str):
    # max_num_solutions = 100
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
                # results['mixing_time_lb'] = results['mixing_time_bounds'][0]
                # results['mixing_time_ub'] = results['mixing_time_bounds'][1]
                # del results['mixing_time_bounds']
                simple_results.append(results)
        elif m2 is not None:
            with open(os.path.join(results_dir, file), 'rb') as f:
                results = pickle.load(f)
                results['identifier'] = m2.group(1)
                results['k'] = int(m2.group(2))
                # results['mixing_time_lb'] = results['mixing_time_bounds'][0]
                # results['mixing_time_ub'] = results['mixing_time_bounds'][1]
                # del results['mixing_time_bounds']
                reduced_results.append(results)
    # for results in simple_results + reduced_results:
        # if 'mixing_time_tolerance' not in results:
            # results['mixing_time_tolerance'] = 0.0
    simple_df = pd.DataFrame(simple_results)
    reduced_df = pd.DataFrame(reduced_results)
    # simple_df = simple_df.loc[simple_df['num_solutions'] <= max_num_solutions]
    # reduced_df = reduced_df.loc[reduced_df['num_solutions'] <= max_num_solutions]
    return simple_df, reduced_df

def add_mixing_time(df: pd.DataFrame, eps=1/(2*np.exp(1))):
    df['spectral_gap_lb'] = df['spectral_gap'] - df['spectral_gap_tolerance']
    df['spectral_gap_ub'] = df['spectral_gap'] + df['spectral_gap_tolerance']
    df['mixing_time_lb'] = np.floor((1/df['spectral_gap_ub'] - 1) * np.log(1/(2*eps)))
    df['mixing_time_ub'] = np.ceil(1/df['spectral_gap_lb'] * np.log(df['num_solutions']/eps))
    df.loc[df['spectral_gap_lb'] <= 0, 'mixing_time_ub'] = np.inf

def add_exp_mixing_time(df: pd.DataFrame):
    df['exp_mixing_time_lb'] = df['mixing_time_lb'] / df['solution_density']
    df['exp_mixing_time_ub'] = df['mixing_time_ub'] / df['solution_density']

def get_min_mixing_time_df(df: pd.DataFrame, lb_type: str):
    return df.loc[df.groupby(['identifier'])[lb_type].idxmin()]

def scatter_mixing_times(simple_df: pd.DataFrame, reduced_df: pd.DataFrame, x_axis:str):
    plt.scatter(simple_df[x_axis], simple_df['exp_mixing_time_lb'], label='simple LB')
    plt.scatter(reduced_df[x_axis], reduced_df['mixing_time_lb'], label='reduced LB')
    plt.xlabel(x_axis)
    plt.ylabel('expected mixing time LB')
    plt.yscale('log')
    plt.legend()
    plt.show()

def scatter_only_reduced(reduced_df: pd.DataFrame, x_axis:str, column: str):
    plt.scatter(reduced_df[x_axis], reduced_df[column])
    # only show integer x ticks
    # plt.xticks(np.arange(min(reduced_df[x_axis]), max(reduced_df[x_axis])+1, 1.0))
    plt.xlabel(AXIS_LABELS[x_axis] if x_axis in AXIS_LABELS else x_axis)
    plt.ylabel(AXIS_LABELS[column] if column in AXIS_LABELS else column)
    plt.yscale('log')
    plt.savefig('img/reduced_{}_{}.png'.format(x_axis, column), dpi=300)
    plt.show()

def scatter_mixing_time_ratios(simple_df: pd.DataFrame, reduced_df: pd.DataFrame, bound='lower'):
    # find common identifiers
    common_ids = set(simple_df['identifier']).intersection(set(reduced_df['identifier']))
    simple_df = simple_df.loc[simple_df['identifier'].isin(common_ids)].copy().reset_index(drop=True)
    reduced_df = reduced_df.loc[reduced_df['identifier'].isin(common_ids)].copy().reset_index(drop=True)
    # sort both by identifier
    simple_df.sort_values(by=['identifier'], inplace=True)
    reduced_df.sort_values(by=['identifier'], inplace=True)
    # scatter num_solutions by ratio of mixing times
    if bound == 'lower':
        plt.scatter(simple_df['num_states'], simple_df['exp_mixing_time_lb'] / reduced_df['mixing_time_lb'])
        plt.ylabel('ratio of expected mixing times LBs')
    else:
        plt.scatter(simple_df['num_states'], simple_df['exp_mixing_time_lb'] / reduced_df['mixing_time_ub'])
        plt.ylabel('min ratio of expected mixing times')
    plt.xlabel('number of states')
    plt.yscale('log')
    plt.show()

def scatter_solutions_vs_states(simple_df: pd.DataFrame):
    plt.scatter(simple_df['num_solutions'], simple_df['num_states'])
    plt.xlabel('number of solutions')
    plt.ylabel('number of states')
    plt.yscale('log')
    plt.show()

def connectivity_analysis(reduced_df: pd.DataFrame):
    for k in sorted(reduced_df['k'].unique()):
        print(f'Fraction connected for k={k}', reduced_df.loc[reduced_df['k'] == k]['is_connected'].value_counts()[True] / len(reduced_df.loc[reduced_df['k'] == k]))
    # Find the identifier whre is_connected is False
    print('Disconnected ids', reduced_df.loc[reduced_df['is_connected'] == False]['identifier'].unique())

def max_mixing_time(reduced_df: pd.DataFrame):
    # get identifier for the row with max mixing_time_lb
    ind = reduced_df['mixing_time_lb'].idxmax()
    max_lb = reduced_df.loc[ind]['identifier']
    max_tol = reduced_df.loc[ind]['spectral_gap_tolerance']
    max_tol = reduced_df.loc[ind]['spectral_gap']
    max_time = max(reduced_df['mixing_time_lb'])
    print('Max mixing time', max_lb, max_time, max_tol)

def plot_column(df: pd.DataFrame, column: str, x_axis: str):
    if 'is_connected' in df.columns:
        full_df = df[df['is_connected']]
    else:
        full_df = df
    unique_ids = full_df['identifier'].unique()
    for ide in unique_ids:
        df = full_df.loc[full_df['identifier'] == ide].copy()
        df.sort_values(by=[x_axis], inplace=True)
        if df['spectral_gap_tolerance'].unique().size > 1:
            plt.plot(df[x_axis], df[column], color='r', marker='o')
        else:
            plt.plot(df[x_axis], df[column], color='b', marker='o')
    if x_axis == 'gamma':
        plt.xlabel(r'$\gamma$')
    else:
        # only show integer x ticks
        # plt.xticks(np.arange(min(df[x_axis]), max(df[x_axis])+1, 1.0))
        plt.xlabel('$k$')
    plt.ylabel(column)
    plt.yscale('log')
    plt.show()

def hist_column(opt_simple_df: pd.DataFrame, column: str, log=False):
    if log:
        plt.hist(np.log(opt_simple_df[column]), bins=20)
        plt.xlabel(f'log {column}')
    else:
        plt.hist(opt_simple_df[column], bins=20)
        plt.xlabel(column)
    plt.ylabel('count')
    plt.show()

if __name__ == '__main__':
    parser_builder.parse_args()
    print(parser_builder.args)
    args = parser_builder.args

    common_num_sols = 100

    simple_df, reduced_df = load_results(args.mcmc_output_dir)

    add_mixing_time(simple_df)
    add_exp_mixing_time(simple_df)
    add_mixing_time(reduced_df)

    full_reduced_df = reduced_df.copy()

    simple_df = simple_df.loc[simple_df['num_solutions'] <= common_num_sols]
    reduced_df = reduced_df.loc[reduced_df['num_solutions'] <= common_num_sols]

    opt_simple_df = get_min_mixing_time_df(simple_df, 'exp_mixing_time_lb')
    opt_reduced_df = get_min_mixing_time_df(reduced_df, 'mixing_time_lb')
    opt_full_reduced_df_lb = get_min_mixing_time_df(full_reduced_df, 'mixing_time_lb')
    opt_full_reduced_df_ub = get_min_mixing_time_df(full_reduced_df, 'mixing_time_ub')

    # scatter_solutions_vs_states(opt_simple_df)
    # scatter_mixing_times(opt_simple_df, opt_reduced_df, 'num_solutions')
    scatter_mixing_times(opt_simple_df, opt_reduced_df, 'num_elements')
    # scatter_mixing_time_ratios(opt_simple_df, opt_reduced_df, bound='upper')
    # scatter_only_reduced(opt_full_reduced_df_ub, 'num_solutions', 'mixing_time_ub')
    scatter_only_reduced(opt_full_reduced_df_ub, 'num_elements', 'mixing_time_ub')
    connectivity_analysis(full_reduced_df)
    # max_mixing_time(full_reduced_df)
    plot_column(simple_df, 'solution_density', 'gamma')
    plot_column(simple_df, 'exp_mixing_time_lb', 'gamma')
    # plot_column(simple_df, 'exp_mixing_time_ub', 'gamma')
    plot_column(full_reduced_df, 'mixing_time_lb', 'k')
    hist_column(opt_simple_df, 'exp_mixing_time_lb', log=True)
    hist_column(opt_reduced_df, 'mixing_time_ub')

    # TODO verify that gamma = 1 is sufficient
    # looks like there's at least one block where we need larger gamma
