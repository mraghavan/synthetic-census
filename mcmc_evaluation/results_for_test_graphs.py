import sys
import os
import pickle
import re
import matplotlib.pyplot as plt
# from matplotlib.axes import Axes
sys.path.append('../')

def is_simple(k):
    return type(k[2]) == float

def exp_mixing_time_lb(all_results, c, d, gamma):
    return  all_results[(c, d, gamma)]['mixing_time'][0] / all_results[(c, d, gamma)]['solution_density']

def exp_mixing_time_ub(all_results, c, d, gamma):
    return  all_results[(c, d, gamma)]['mixing_time'][1] / all_results[(c, d, gamma)]['solution_density']

def plot_solution_density_and_mixing_time(all_results):
    simple_results = {k: v for k, v in all_results.items() if is_simple(k)}
    gammas = [k[2] for k in simple_results.keys()]
    c_d_pairs = set((k[0], k[1]) for k in all_results.keys())
    gamma_list = sorted(set(gammas))
    print(gamma_list)
    for c, d in c_d_pairs:
        # mixing_times_lbs = [all_results[(c, d, g)]['mixing_time'][0] for g in gamma_list]
        mixing_times_ubs = [all_results[(c, d, g)]['mixing_time'][1] for g in gamma_list]
        # plt.plot(gamma_list, mixing_times_lbs, label='LB for c={}, d={}'.format(c, d), marker='o')
        plt.plot(gamma_list, mixing_times_ubs, label='UB for c={}, d={}'.format(c, d), marker='o')
    plt.xlabel(r'$\gamma$')
    plt.ylabel('mixing time')
    plt.yscale('log')
    plt.legend()
    plt.show()

    for c, d in c_d_pairs:
        densities = [all_results[(c, d, g)]['solution_density'] for g in gamma_list]
        plt.plot(gamma_list, densities, label='solution density for c={}, d={}'.format(c, d), marker='o')
    plt.xlabel(r'$\gamma$')
    plt.ylabel('solution density')
    plt.yscale('log')
    plt.legend()
    plt.show()

    for c, d in c_d_pairs:
        exp_mixing_times = [exp_mixing_time_lb(all_results, c, d, g) for g in gamma_list]
        plt.plot(gamma_list, exp_mixing_times, label='expected #iterations LB for c={}, d={}'.format(c, d), marker='o')
    plt.xlabel(r'$\gamma$')
    plt.ylabel('expected #iterations')
    plt.yscale('log')
    plt.legend()
    plt.show()

    d = 4
    all_cs = sorted(set(k[0] for k in all_results.keys() if k[1] == d))
    print(all_cs)
    for c in all_cs:
        plt.plot(gamma_list, [exp_mixing_time_lb(all_results, c, d, g) for g in gamma_list], label='expected #iterations for LB c={}, d={}'.format(c, d), marker='o')
    plt.xlabel(r'$\gamma$')
    plt.ylabel('expected #iterations LB')
    plt.yscale('log')
    plt.legend()
    plt.show()

    for c in all_cs:
        plt.plot(gamma_list, [exp_mixing_time_ub(all_results, c, d, g) for g in gamma_list], label='expected #iterations for UB c={}, d={}'.format(c, d), marker='o')
    plt.xlabel(r'$\gamma$')
    plt.ylabel('expected #iterations UB')
    plt.yscale('log')
    plt.legend()
    plt.show()
    # densities = [simple_results[gammas[g]]['solution_density'] for g in gamma_list]
    # ax1: Axes = None #type: ignore
    # _, ax1 = plt.subplots() #type: ignore
    # # color_cycle = ax1._get_lines.color_cycle
    # line1, = ax1.plot(gamma_list, [1/d for d in densities], label='1/solution density', marker='o')
    # ax1.set_yscale('log') #type: ignore
    # ax1.set_xlabel(r'$\gamma$')
    # ax1.set_ylabel('1/solution density')
    # ax2 = ax1.twinx()
    # line2, = ax2.plot(gamma_list, mixing_times_lbs, label='mixing time LB', color='r', marker='o')
    # line3, = ax2.plot(gamma_list, mixing_times_ubs, label='mixing time UB', color='g', marker='o')
    # ax2.set_yscale('log')
    # ax2.set_ylabel('mixing time')
    # lines = [line1, line2, line3]
    # labels = [line.get_label() for line in lines]
    # ax1.legend(lines, labels, loc='upper center')
    # plt.tight_layout()
    # plt.savefig('img/mixing_time_vs_solution_density.png')
    # plt.show()

    # conductances = [simple_results[gammas[g]]['conductance_ub'] for g in gamma_list]
    # conductance_mixings = [(1/(2*c) - 1) for c in conductances]
    # conductance_lbs = [mt*1/d for mt, d in zip(conductance_mixings, densities)]

    # k_results = {k: v for k, v in all_results.items() if not is_simple(k)}
    # ks = extract_params(k_results, K_PATTERN, int)
    # k_list = sorted(ks.keys())

    # expected_time_lbs = [mt * 1/d for mt, d in zip(mixing_times_lbs, densities)]
    # expected_time_ubs = [mt * 1/d for mt, d in zip(mixing_times_ubs, densities)]
    # k_mixing_time_lbs = [k_results[ks[k]]['mixing_time'][0] for k in k_list]
    # k_mixing_time_ubs = [k_results[ks[k]]['mixing_time'][1] for k in k_list]
    # _, ax1 = plt.subplots() #type: ignore
    # lines = []
    # l, = ax1.plot(k_list, k_mixing_time_lbs, label='complex LB', marker='o')
    # lines.append(l)
    # l, = ax1.plot(k_list, k_mixing_time_ubs, label='complex UB', marker='o')
    # lines.append(l)
    # ax1.set_yscale('log') #type: ignore
    # ax1.set_xlabel('$k$')
    # ax1.set_ylabel('expected #iterations')
    # ax2 = ax1.twiny()
    # l, = ax2.plot(gamma_list, expected_time_lbs, label='simple LB', color='r', marker='o')
    # lines.append(l)
    # l, = ax2.plot(gamma_list, expected_time_ubs, label='simple UB', color='g', marker='o')
    # lines.append(l)
    # l, = ax2.plot(gamma_list, conductance_lbs, label='conductance LB', color='y', marker='o')
    # lines.append(l)
    # ax2.set_xlabel(r'$\gamma$')
    # labels = [line.get_label() for line in lines]
    # plt.title("Expected iterations to generate a valid solution")
    # ax1.legend(lines, labels)
    # plt.tight_layout()
    # ax1.grid(axis='y')
    # plt.savefig('img/expected_time_k_gamma.png')
    # plt.show()


def load_all_results(open_dir):
    all_results = {}
    pattern = r'test_mcmc_results_(\d+)_(\d+)_(\d+(.\d+)?).pkl'
    for filename in os.listdir(open_dir):
        m = re.match(pattern, filename)
        if m:
            with open(os.path.join(open_dir, filename), 'rb') as f:
                print('Loading', filename)
                c = int(m.group(1))
                d = int(m.group(2))
                param = m.group(3)
                if '.' in param:
                    param = float(param)
                else:
                    param = int(param)
                results = pickle.load(f)
                all_results[(c, d, param)] = results
    return all_results

if __name__ == '__main__':
    open_dir = 'test_graphs/'
    all_results = load_all_results(open_dir)
    plot_solution_density_and_mixing_time(all_results)

