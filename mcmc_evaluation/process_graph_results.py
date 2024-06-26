import sys
from collections import Counter
import os
import numpy as np
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
from matplotlib.legend_handler import HandlerTuple
sys.path.append('../')
from syn_census.utils.config2 import ParserBuilder

SHOW = True

simple_re = re.compile(r'(\d+-\d+-\d+)_(\d+(.\d+)?)_gibbs_results.pkl')
reduced_re = re.compile(r'(\d+-\d+-\d+)_(\d+(.\d+)?)_reduced_results.pkl')

parser_builder = ParserBuilder(
        {'state': True,
         'micro_file': True,
         'block_clean_file': True,
         'mcmc_output_dir': True,
         'num_sols': False,
         'no_show': False,
         })

AXIS_LABELS = {
        'solution_density': r'Solution density ($p_{\gamma}$)',
        'mixing_time': 'Mixing time',
        'num_solutions': r'$|\mathcal{X}|$',
        'mixing_time_lb': r'$\underline{N}_{k}$',
        'exp_mixing_time_lb': r'$\underline{N}_{\gamma}$',
        'mixing_time_ub': 'Upper bound on mixing time',
        'num_elements': '$m$',
        'gamma': r'$\gamma$',
        'k': '$k$',
        }

def save_if_file(save_file: str):
    if save_file:
        print(f'Saving to {save_file}')
        plt.savefig(save_file, dpi=300)

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

def add_mixing_time_reduced(df: pd.DataFrame, eps=1/(2*np.exp(1))):
    df['spectral_gap_lb'] = df['spectral_gap'] - df['spectral_gap_tolerance']
    df['spectral_gap_ub'] = df['spectral_gap'] + df['spectral_gap_tolerance']
    df['mixing_time_lb'] = np.floor((1/df['spectral_gap_ub'] - 1) * np.log(1/(2*eps)))
    df.loc[df['num_solutions'] == 1, 'mixing_time_lb'] = 1
    df['mixing_time_ub'] = np.ceil(1/df['spectral_gap_lb'] * np.log(df['num_solutions']/eps))
    df.loc[df['is_connected'] == False, 'mixing_time_ub'] = np.inf
    df.loc[df['is_connected'] == False, 'mixing_time_lb'] = np.inf
    assert all(df['mixing_time_ub'] >= df['mixing_time_lb'])

def add_mixing_time_simple(df: pd.DataFrame, eps=1/(2*np.exp(1))):
    eps_times_sol_density = df['solution_density'] * eps
    df['spectral_gap_lb'] = df['spectral_gap'] - df['spectral_gap_tolerance']
    df['spectral_gap_ub'] = df['spectral_gap'] + df['spectral_gap_tolerance']
    df['mixing_time_lb'] = np.floor((1/df['spectral_gap_ub'] - 1) * np.log(3/(4*eps_times_sol_density)))
    df['mixing_time_ub'] = np.ceil(1/df['spectral_gap_lb'] * np.log(3*df['num_solutions']/(2*eps_times_sol_density)))
    df.loc[df['spectral_gap_lb'] <= 0, 'mixing_time_ub'] = np.inf

def add_exp_mixing_time(df: pd.DataFrame, eps=1/(2*np.exp(1))):
    df['exp_mixing_time_lb'] = df['mixing_time_lb'] / (df['solution_density'] * (1 + 2*eps/3))
    df['exp_mixing_time_ub'] = df['mixing_time_ub'] / (df['solution_density'] * (1 - 2*eps/3))

def get_min_mixing_time_df(df: pd.DataFrame, lb_type: str):
    return df.loc[df.groupby(['identifier'])[lb_type].idxmin()]

def scatter_mixing_times(simple_df: pd.DataFrame,
                         reduced_df: pd.DataFrame,
                         x_axis:str,
                         simple_fail_df: pd.DataFrame = None,
                         reduced_fail_df: pd.DataFrame = None,
                         save_file=''):
    plt.scatter(simple_df[x_axis], simple_df['exp_mixing_time_lb'], label=r'$\underline{N}_{\gamma^*}$ (simple LB)')
    plt.scatter(reduced_df[x_axis], reduced_df['mixing_time_lb'], marker='^', label=r'$\underline{N}_{k=3}$ (reduced LB)', zorder=10)
    plt.scatter(reduced_df[x_axis], reduced_df['mixing_time_ub'], marker='v', label=r'$\overline{N}_{k=3}$ (reduced UB)', zorder=10)
    # for i, row in reduced_df.iterrows():
        # plt.plot([row[x_axis], row[x_axis]], [row['mixing_time_lb'], row['mixing_time_ub']], color='gray', linestyle='--')
    if simple_fail_df is not None or reduced_fail_df is not None:
        plt.gca().set_prop_cycle(None)
        vals = [df[x_axis] for df in [simple_fail_df, reduced_fail_df] if df is not None]
        labels = [label for label, df in zip(['simple failures', 'reduced failures'],
                                              [simple_fail_df, reduced_fail_df])
                  if df is not None]
        plt.hist(vals,
                 label=labels,
                 bins=reduced_df['num_elements'].max(),
                 color='red',
                 hatch='//',
                 zorder=0)
    if x_axis in AXIS_LABELS:
        plt.xlabel(AXIS_LABELS[x_axis])
    else:
        plt.xlabel(x_axis)
    plt.yscale('log')
    plt.legend()
    save_if_file(save_file)
    if SHOW:
        plt.show()
    else:
        plt.close()

def scatter_only_reduced(reduced_df: pd.DataFrame,
                         x_axis:str,
                         column: str,
                         reduced_fail_df: pd.DataFrame = None,
                         save_file=''):
    plt.scatter(reduced_df[x_axis], reduced_df[column], label='reduced')
    if reduced_fail_df is not None:
        plt.gca().set_prop_cycle(None)
        plt.hist(reduced_fail_df[x_axis],
                 label='failures',
                 bins=reduced_df['num_elements'].max())
    # only show integer x ticks
    # plt.xticks(np.arange(min(reduced_df[x_axis]), max(reduced_df[x_axis])+1, 1.0))
    plt.xlabel(AXIS_LABELS[x_axis] if x_axis in AXIS_LABELS else x_axis)
    # plt.ylabel(AXIS_LABELS[column] if column in AXIS_LABELS else column)
    plt.ylabel(r'$\underline{N}_{k=3}$' if column == 'mixing_time_lb' else r'$\overline{N}_{k=3}$')
    plt.yscale('log')
    if reduced_fail_df is not None:
        plt.legend()
    # fname = f'img/{args.state}_reduced_{x_axis}_{column}.png'
    # print('Saving to', fname)
    # plt.savefig(fname, dpi=300)
    save_if_file(save_file)
    if SHOW:
        plt.show()
    else:
        plt.close()

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
    if SHOW:
        plt.show()
    else:
        plt.close()

def scatter_solutions_vs_states(simple_df: pd.DataFrame):
    plt.scatter(simple_df['num_solutions'], simple_df['num_states'])
    plt.xlabel('number of solutions')
    plt.ylabel('number of states')
    plt.yscale('log')
    if SHOW:
        plt.show()
    else:
        plt.close()

def connectivity_analysis(reduced_df: pd.DataFrame, output_file: str):
    for k in sorted(reduced_df['k'].unique()):
        k_df = reduced_df.loc[reduced_df['k'] == k]
        print(f'Fraction connected for k={k}', k_df['is_connected'].value_counts()[True] / len(k_df))
    # Find the identifier whre is_connected is False
    ids = reduced_df.loc[reduced_df['is_connected'] == False]['identifier']
    ks = reduced_df.loc[reduced_df['is_connected'] == False]['k']
    print('Disconnected ids', ids.values, ks.values)
    with open(output_file, 'w') as f:
        print('Writing disconnected ids to', output_file)
        for ide, k in zip(ids, ks):
            f.write(f'{ide},{k}\n')

def max_mixing_time(reduced_df: pd.DataFrame):
    # get identifier for the row with max mixing_time_lb
    ind = reduced_df['mixing_time_lb'].idxmax()
    max_lb = reduced_df.loc[ind]['identifier']
    max_tol = reduced_df.loc[ind]['spectral_gap_tolerance']
    max_tol = reduced_df.loc[ind]['spectral_gap']
    max_time = max(reduced_df['mixing_time_lb'])
    print('Max mixing time', max_lb, max_time, max_tol)

def plot_column(df: pd.DataFrame, column: str, x_axis: str, save_file='', title=''):
    df = df.copy()
    # df.loc[df['spectral_gap_tolerance'] > 0, column] = np.NaN
    if 'is_connected' in df.columns:
        full_df = df[df['is_connected']]
    else:
        full_df = df
    unique_ids = full_df['identifier'].unique()
    unique_params = full_df[x_axis].unique()
    disconnected = tuple()
    incomplete = tuple()
    complete = tuple()
    low_precision = tuple()
    for ide in unique_ids:
        df = full_df.loc[full_df['identifier'] == ide].copy()
        df.sort_values(by=[x_axis], inplace=True)
        if df[x_axis].min() != full_df[x_axis].min():
            disconnected = tuple(plt.plot(df[x_axis], df[column], color='xkcd:light purple', marker='P', zorder=4))
        elif len(df) != len(unique_params):
            incomplete = tuple(plt.plot(df[x_axis], df[column], color='y', marker='d', zorder=2))
        else:
            complete = tuple(plt.plot(df[x_axis], df[column], color='b', marker='o', zorder=1))
    tol_df = full_df[full_df['spectral_gap_tolerance'] > 0]
    if len(tol_df) > 0:
        low_precision = plt.scatter(tol_df[x_axis], tol_df[column], color='r', marker='^', zorder=3, label='low precision')
    if x_axis == 'gamma':
        plt.xlabel(r'$\gamma$')
    else:
        # only show integer x ticks
        plt.xticks(np.arange(min(df[x_axis]), max(df[x_axis])+1, 1.0))
        plt.xlabel('$k$')
    plt.ylabel(AXIS_LABELS[column] if column in AXIS_LABELS else column)
    plt.yscale('log')
    handles, labels = zip(*[(h, l) for h, l in zip((complete, incomplete, disconnected, low_precision), ('complete', 'incomplete', 'disconnected', 'low precision')) if h])
    plt.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None)})
    # handles, _ = plt.gca().get_legend_handles_labels()
    # if len(handles) > 1:
        # plt.legend()
    plt.title(title)
    save_if_file(save_file)
    if SHOW:
        plt.show()
    else:
        plt.close()

def hist_column(opt_simple_df: pd.DataFrame, column: str, log=False):
    if log:
        plt.hist(np.log(opt_simple_df[column]), bins=20)
        plt.xlabel(f'log {column}')
    else:
        plt.hist(opt_simple_df[column], bins=20)
        plt.xlabel(column)
    plt.ylabel('count')
    if SHOW:
        plt.show()
    else:
        plt.close()

def opt_param(opt_df: pd.DataFrame, param_col: str, save_file=''):
    xy = [(x, y) for x, y in zip(opt_df['num_elements'], opt_df[param_col])]
    c = Counter(xy)
    xs, ys, ss = zip(*[(x, y, 10*c[(x, y)]) for x, y in c])
    plt.scatter(xs, ys, s=ss)
    plt.xlabel('$m$')
    if param_col == 'k':
        plt.ylabel('$k^*$')
    elif param_col == 'gamma':
        plt.ylabel(r'$\gamma^*$')
    else:
        plt.ylabel('Optimal ' + AXIS_LABELS[param_col] if param_col in AXIS_LABELS else param_col)
    if opt_df[param_col].dtype == 'int':
        # only show integer y ticks
        plt.yticks(np.arange(min(opt_df[param_col]), max(opt_df[param_col])+1, 1.0))
    save_if_file(save_file)
    if SHOW:
        plt.show()
    else:
        plt.close()

def get_opt_global_param(df: pd.DataFrame, param_col: str, bound_col: str):
    df = df.copy()
    df[param_col] = df[param_col].astype('category')
    # df.loc[df['spectral_gap_tolerance'] > 0, bound_col] = np.inf

    # Find the value of k that minimizes the maximum count
    # print( df.groupby([param_col])[bound_col].mean())
    result = df.groupby([param_col])[bound_col].mean().idxmin()
    return result

def get_df_from_failures(failures: set, df: pd.DataFrame):
    df = df.loc[df['identifier'].isin(failures)].copy()
    df.rename(columns={'num_sols': 'num_solutions'}, inplace=True)
    return df

def get_complete_failures(simple_df, reduced_df, blocks_out_file: str, jobs_file: str):
    results_df = pd.read_csv(blocks_out_file)
    simple_jobs = []
    reduced_jobs = []
    simple_failures = set()
    reduced_failures = set()
    with open(jobs_file) as f:
        for line in f:
            identifier, job_type = line.strip().split(',')
            if job_type == 'gibbs':
                simple_jobs.append(identifier)
            else:
                reduced_jobs.append(identifier)
    print(len(simple_jobs), len(reduced_jobs))
    for identifier in simple_jobs:
        if identifier not in simple_df['identifier'].values:
            simple_failures.add(identifier)
    for identifier in reduced_jobs:
        if identifier not in reduced_df['identifier'].values:
            reduced_failures.add(identifier)
    simple_fail_df = get_df_from_failures(simple_failures, results_df)
    reduced_fail_df = get_df_from_failures(reduced_failures, results_df)
    return simple_fail_df, reduced_fail_df


if __name__ == '__main__':
    parser_builder.parse_args()
    print(parser_builder.args)
    args = parser_builder.args
    plt.rcParams['font.size'] = 14

    SHOW = not args.no_show

    blocks_out_file = os.path.join(args.mcmc_output_dir, 'all_blocks.csv')
    jobs_file = os.path.join(args.mcmc_output_dir, 'sampled_block_ids.txt')

    simple_df, reduced_df = load_results(args.mcmc_output_dir)
    simple_fail_df, reduced_fail_df = get_complete_failures(simple_df, reduced_df, blocks_out_file, jobs_file)
    print(len(simple_fail_df), 'simple failures')
    if len(simple_fail_df) == 0:
        simple_fail_df = None
    if len(reduced_fail_df) == 0:
        reduced_fail_df = None

    add_mixing_time_simple(simple_df)
    add_exp_mixing_time(simple_df)
    add_mixing_time_reduced(reduced_df)

    print(get_opt_global_param(simple_df, 'gamma', 'exp_mixing_time_lb'))
    print(get_opt_global_param(reduced_df, 'k', 'mixing_time_lb'))

    full_reduced_df = reduced_df.copy()

    opt_simple_df = get_min_mixing_time_df(simple_df, 'exp_mixing_time_lb')
    opt_reduced_df = get_min_mixing_time_df(reduced_df, 'mixing_time_lb')
    opt_full_reduced_df_lb = get_min_mixing_time_df(full_reduced_df, 'mixing_time_lb')
    opt_full_reduced_df_ub = get_min_mixing_time_df(full_reduced_df, 'mixing_time_ub')

    opt_param(
            opt_simple_df,
            'gamma',
            save_file=f'./img/{args.state}_opt_param_gamma.png')
    opt_param(opt_reduced_df, 'k', save_file=f'./img/{args.state}_opt_param_k.png')

    # scatter_solutions_vs_states(opt_simple_df)
    # scatter_mixing_times(opt_simple_df, opt_reduced_df, 'num_solutions')
    reduced_df_3 = reduced_df.loc[reduced_df['k'] == 3]
    scatter_mixing_times(opt_simple_df,
                         # opt_reduced_df,
                         reduced_df_3,
                         'num_elements',
                         simple_fail_df,
                         reduced_fail_df,
                         save_file=f'./img/{args.state}_all_num_elements_mixing_time.png')
    # scatter_mixing_time_ratios(opt_simple_df, opt_reduced_df, bound='upper')
    # scatter_only_reduced(opt_full_reduced_df_ub, 'num_solutions', 'mixing_time_ub')
    # scatter_only_reduced(opt_full_reduced_df_lb, 'num_elements', 'mixing_time_lb')
    full_reduced_df_3 = full_reduced_df.loc[full_reduced_df['k'] == 3]
    # scatter_only_reduced(opt_full_reduced_df_lb,
    scatter_only_reduced(full_reduced_df_3,
                         'num_solutions',
                         'mixing_time_lb',
                         reduced_fail_df,
                         save_file=f'./img/{args.state}_num_solutions_mtlb.png')
    connectivity_analysis(full_reduced_df, os.path.join(args.mcmc_output_dir, 'disconnected_graphs.txt'))

    # max_mixing_time(full_reduced_df)
    # plot_column(simple_df, 'solution_density', 'gamma')

    plot_column(
            simple_df,
            'exp_mixing_time_lb',
            'gamma',
            save_file=f'./img/{args.state}_gamma_exp_mixing_time_lb.png',
            )
    # plot_column(simple_df, 'exp_mixing_time_ub', 'gamma')
    plot_column(
            full_reduced_df,
            'mixing_time_lb',
            'k',
            save_file=f'./img/{args.state}_k_mixing_time_lb.png',
            )
    # hist_column(opt_simple_df, 'exp_mixing_time_lb', log=True)
    # hist_column(opt_reduced_df, 'mixing_time_ub')
