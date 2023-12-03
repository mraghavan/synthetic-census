import sys
import os
import pickle
import re
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
sys.path.append('../')
from syn_census.utils.config2 import ParserBuilder
from get_graph_stats import is_simple, SIMPLE_PATTERN, K_PATTERN

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


if __name__ == '__main__':
    parser_builder.parse_args()
    print(parser_builder.args)
    args = parser_builder.args
    
    fname = '{}_graph_results.pkl'
    # get all files matching the patterns
    formats = [fname.format(SIMPLE_PATTERN), fname.format(K_PATTERN)]
    matching_files = []
    for form in formats:
        matching_files += [f for f in os.listdir(args.mcmc_output_dir) if re.match(form, f)]
    print(matching_files)
    all_results = {}
    
    for file in matching_files:
        with open(os.path.join(args.mcmc_output_dir, file), 'rb') as f:
            print(f'Loading {file}')
            all_results[file] = pickle.load(f)
    print(all_results)

    plot_solution_density_and_mixing_time(all_results)
