import pickle
import matplotlib.pyplot as plt
from config2 import ParserBuilder
import re
import os

parser_builder = ParserBuilder(
        {'state': True,
         'micro_file': True,
         'block_clean_file': True,
         'synthetic_output_dir': False,
         })


if __name__ == '__main__':
    parser_builder.parse_args()
    print(parser_builder.args)
    parser_builder.verify_required_args()
    args = parser_builder.args

    results = None
    # open all the files
    random_tvds = None
    random_missing_masses = None
    sample_size = None
    with open(args.synthetic_output_dir + 'random_dists.pkl', 'rb') as f:
        random_dists = pickle.load(f)
        print(random_dists.keys())
        random_tvds = random_dists['random_tvds']
        random_missing_masses = random_dists['random_missing_masses']
        sample_size = random_dists['sample_size']

    all_tvds = {}
    all_missing_masses = {}
    for f in os.listdir(args.synthetic_output_dir):
        # check if f matches mcmc_[0-9]+_[0-9]+.pkl
        if re.match('mcmc_[0-9]+_[0-9]+.pkl', f):
            with open(args.synthetic_output_dir + f, 'rb') as f:
                results = pickle.load(f)
                tvds = results['all_tvds']
                missing_masses = results['all_missing_masses']
                for k, v in tvds.items():
                    all_tvds[k] = v
                for k, v in missing_masses.items():
                    all_missing_masses[k] = v
    print(all_tvds)
    print(all_missing_masses)

    # num_random_samples = results['num_random_samples']
    # print('Number of samples per test', num_random_samples)
    
    # random_tvds = random_dists['random_tvds']
    # random_missing_masses = results['random_missing_masses']
    # all_tvds = results['all_tvds']
    # all_missing_masses = results['all_missing_masses']

    # get 95% upper bound on random tvds
    random_tvds = sorted(random_tvds)
    tvd_upper = [random_tvds[int(len(random_tvds) * 0.95)]]

    # get width of 95% confidence interval on random tvds
    tvd_interval = random_tvds[int(len(random_tvds) * 0.975)] - random_tvds[int(len(random_tvds) * 0.025)]

    # get 95% upper bound on random missing masses
    random_missing_masses = sorted(random_missing_masses)
    missing_mass_upper = [random_missing_masses[int(len(random_missing_masses) * 0.95)]]

    # get width of 95% confidence interval on random missing masses
    missing_mass_interval = random_missing_masses[int(len(random_missing_masses) * 0.975)] - random_missing_masses[int(len(random_missing_masses) * 0.025)]

    # make tvd plot: one line for each k, x axis is num_iterations
    all_ks = sorted(list(set([k for _, k in all_tvds])))
    all_num_iterations = sorted(list(set([num_iterations for num_iterations, _ in all_tvds])))
    for k in all_ks:
        plt.plot(all_num_iterations, [all_tvds[(num, k)] for num in all_num_iterations], label='k={}'.format(k))
    plt.legend()
    plt.xlabel('Number of MCMC iterations')
    plt.ylabel('Total variation distance')
    # log scale x axis
    plt.xscale('log')
    # make horizontal lines at tvd_interval
    plt.plot([min(all_num_iterations), max(all_num_iterations)], [tvd_upper, tvd_upper], color='black', linestyle='--')
    plt.title('TVD, {} samples'.format(sample_size))
    plt.savefig('tvd_plot.png')
    plt.show()

    # make missing mass plot: one line for each k, x axis is num_iterations
    for k in all_ks:
        plt.plot(all_num_iterations, [all_missing_masses[(num, k)] for num in all_num_iterations], label='k={}'.format(k))
    plt.legend()
    plt.xlabel('Number of MCMC iterations')
    plt.ylabel('Missing probability mass')
    # log scale x axis
    plt.xscale('log')
    # make horizontal lines at missing_mass_interval
    plt.plot([min(all_num_iterations), max(all_num_iterations)], [missing_mass_upper, missing_mass_upper], color='black', linestyle='--')
    plt.title('Missing probability mass, {} samples'.format(sample_size))
    plt.savefig('missing_mass_plot.png')
    plt.show()
