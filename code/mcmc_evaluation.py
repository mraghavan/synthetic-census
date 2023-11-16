import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    results = None
    with open('./mcmc_results.pkl', 'rb') as f:
        results = pickle.load(f)
    print(results.keys())

    # num_random_samples = results['num_random_samples']
    # print('Number of samples per test', num_random_samples)
    
    random_tvds = results['random_tvds']
    random_missing_masses = results['random_missing_masses']
    all_tvds = results['all_tvds']
    all_missing_masses = results['all_missing_masses']

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
        plt.errorbar([num_iterations for num_iterations, _ in all_tvds if k == _], [all_tvds[(num_iterations, k)] for num_iterations, _ in all_tvds if k == _], label='k={}'.format(k), yerr=tvd_interval)
    plt.legend()
    plt.xlabel('Number of MCMC iterations')
    plt.ylabel('Total variation distance')
    # log scale x axis
    plt.xscale('log')
    # make horizontal lines at tvd_interval
    plt.plot([min(all_num_iterations), max(all_num_iterations)], [tvd_upper, tvd_upper], color='black', linestyle='--')
    plt.title('TVD, {} samples'.format(results['num_random_samples']))
    plt.savefig('tvd_plot.png')
    plt.show()

    # make missing mass plot: one line for each k, x axis is num_iterations
    for k in all_ks:
        plt.errorbar([num_iterations for num_iterations, _ in all_missing_masses if k == _], [all_missing_masses[(num_iterations, k)] for num_iterations, _ in all_missing_masses if k == _], label='k={}'.format(k), yerr=missing_mass_interval)
    plt.legend()
    plt.xlabel('Number of MCMC iterations')
    plt.ylabel('Missing probability mass')
    # log scale x axis
    plt.xscale('log')
    # make horizontal lines at missing_mass_interval
    plt.plot([min(all_num_iterations), max(all_num_iterations)], [missing_mass_upper, missing_mass_upper], color='black', linestyle='--')
    plt.title('Missing probability mass, {} samples'.format(results['num_random_samples']))
    plt.savefig('missing_mass_plot.png')
    plt.show()
