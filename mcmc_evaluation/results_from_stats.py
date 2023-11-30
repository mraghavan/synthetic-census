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
    mixing_times = [simple_results[gammas[g]]['mixing_time'] for g in gamma_list]
    print(mixing_times)
    densities = [simple_results[gammas[g]]['solution_density'] for g in gamma_list]
    print(densities)
    ax1: Axes = None #type: ignore
    _, ax1 = plt.subplots() #type: ignore
    line1, = ax1.plot(gamma_list, [1/d for d in densities], label='1/solution density')
    ax1.set_yscale('log') #type: ignore
    ax1.set_xlabel(r'$\gamma$')
    ax1.set_ylabel('1/solution density')
    ax2 = ax1.twinx()
    line2, = ax2.plot(gamma_list, mixing_times, label='mixing time', color='r')
    ax2.set_yscale('log')
    ax2.set_ylabel('mixing time')
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper center')
    plt.tight_layout()
    plt.savefig('img/mixing_time_vs_solution_density.png')
    plt.show()

    k_results = {k: v for k, v in all_results.items() if not is_simple(k)}
    ks = extract_params(k_results, K_PATTERN, int)
    k_list = sorted(ks.keys())

    expected_times = [mt * 1/d for mt, d in zip(mixing_times, densities)]
    k_mixing_times = [k_results[ks[k]]['mixing_time'] for k in k_list]
    _, ax1 = plt.subplots() #type: ignore
    line1, = ax1.plot(k_list, k_mixing_times, label='complex')
    ax1.set_yscale('log') #type: ignore
    ax1.set_xlabel('$k$')
    ax1.set_ylabel('expected #iterations')
    ax2 = ax1.twiny()
    line2, = ax2.plot(gamma_list, expected_times, label='simple', color='r')
    ax2.set_xlabel(r'$\gamma$')
    lines = [line1, line2]
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
