from knapsack_utils import *
from ip_distribution import ip_solve
from math import log
from scipy.special import comb
from collections import Counter
from itertools import combinations
from functools import lru_cache
import random
import multiprocessing as mp

MAX_SOLUTIONS = 1000

def get_log_prob(sol, dist):
    if len(sol) == 0:
        return 0
    return sum(log(dist[hh]) for hh in sol) + log(perms_to_combs(sol))

def sols_equal(sol1, sol2):
    c = Counter()
    for hh in sol1:
        c[hh] += 1
    for hh in sol2:
        c[hh] -= 1
    return all(v == 0 for v in c.values())

class MCMCSampler:
    # num_transitions = 0
    def __init__(self, dist, num_iterations=1000, k=5):
        self.dist = dist
        self.num_iterations = num_iterations
        self.k = k

    @lru_cache(maxsize=None)
    def ip_solve_cached(self, counts):
        # print(counts)
        solutions = ip_solve(counts, self.dist, num_solutions=MAX_SOLUTIONS)
        if len(solutions) == 0:
            raise Exception('No solutions')
        elif len(solutions) >= MAX_SOLUTIONS:
            raise Exception('Too many solutions')
        return solutions

    def get_g_x_xprime_and_g_xprime_x(self, x, xprime):
        # Currently unused
        # print('x', x)
        # print('xprime', xprime)
        all_ys, all_removed = self.find_all_feasible_removals(x, xprime)
        counter_x = Counter(x)
        counter_xprime = Counter(xprime)
        g_xprime_given_x = 0
        g_x_given_xprime = 0
        for y, removed in zip(all_ys, all_removed):
            pr_y_given_x = self.get_y_given_x(removed, counter_x)
            # print('y given x', pr_y_given_x)
            counter_removed_prime = counter_minus(Counter(xprime), y)
            removed_prime = tuple(counter_removed_prime.elements())
            pr_y_given_xprime = self.get_y_given_x(removed_prime, counter_xprime)
            # print('y given xprime', pr_y_given_xprime)
            num_sols = len(self.ip_solve_cached(tup_sum(removed)))
            if num_sols >= MAX_SOLUTIONS:
                raise Exception('Too many solutions')
            g_x_given_xprime += pr_y_given_xprime / num_sols
            g_xprime_given_x += pr_y_given_x / num_sols
        return g_x_given_xprime, g_xprime_given_x

    def get_y_given_x(self, removed, counter_x):
        removed_counter = Counter(removed)
        return prod([comb(counter_x[a], removed_counter[a]) for a in removed_counter]) / comb(sum(counter_x.values()), len(removed))

    def find_all_feasible_removals(self, x, xprime):
        counter_xprime = Counter(xprime)
        must_remove = []
        for hh in x:
            if counter_xprime[hh] == 0:
                must_remove.append(hh)
            else:
                counter_xprime[hh] -= 1
        if len(must_remove) == self.k:
            return [counter_minus(Counter(x), Counter(must_remove))], [tuple(must_remove)]
        # print('Must remove', must_remove)
        remaining_k = self.k - len(must_remove)
        counter_must_remove = Counter(must_remove)
        remaining = []
        for hh in x:
            if counter_must_remove[hh] == 0:
                remaining.append(hh)
            else:
                counter_must_remove[hh] -= 1
        assert len(remaining) + len(must_remove) == len(x)
        #TODO do we actually want handle duplicates this way?
        all_removed = []
        all_ys = []
        remaining = sorted(remaining)
        used = set()
        for c in combinations(remaining, remaining_k):
            to_remove = tuple(must_remove + list(c))
            # if to_remove in used:
                # # print('here')
                # continue
            used.add(to_remove)
            all_removed.append(to_remove)
            all_ys.append(counter_minus(Counter(x), Counter(to_remove)))
        return all_ys, all_removed

    def get_sampling_num(self, x_counter, removal_counter):
        return prod([comb(x_counter[a], removal_counter[a]) for a in removal_counter])

    def generate_random_removal(self, x: tuple[tuple[int]]):
        # remove k elements at random
        remove_indices = np.random.choice(len(x), self.k, replace=False)
        # y = tuple(a for i, a in enumerate(x) if i not in remove_indices)
        removed = tuple(x[i] for i in remove_indices)
        if np.random.random() < 1/self.get_sampling_num(Counter(x), Counter(removed)):
            return removed
        else:
            return None

    def mcmc_solve(self, counts: tuple[int]):
        # get initial solution using ip_solve
        current_solution = ip_solve(counts, self.dist, num_solutions=1)[0]
        # print('Initial solution:', current_solution)
        assert(len(current_solution) > self.k)
        for _ in range(self.num_iterations):
            # TODO make this use get_next_state
            # print('Iteration', i, current_solution)
            # Make the chain lazy
            if np.random.random() < 0.5:
                continue
            # randomly remove k items from the solution
            removed = self.generate_random_removal(current_solution)
            if removed is None:
                continue
            counter_removed = Counter(removed)
            xprime = []
            for hh in current_solution:
                if counter_removed[hh] == 0:
                    xprime.append(hh)
                else:
                    counter_removed[hh] -= 1
            # print('Removing', removed)
            # use ip_solve to solve the new subproblem
            all_solutions = self.ip_solve_cached(tup_sum(removed))
            if len(all_solutions) >= MAX_SOLUTIONS:
                raise Exception('Too many solutions')
            # randomly choose one of the solutions
            xprime += all_solutions[np.random.choice(len(all_solutions))]
            xprime = tuple(xprime)
            if sols_equal(xprime, current_solution):
                continue
            x_prob = get_log_prob(current_solution, self.dist)
            xprime_prob = get_log_prob(xprime, self.dist)
            ratio = np.exp(xprime_prob - x_prob)
            # print('ratio', ratio)

            # Removing this because it's now symmetric
            # g_x_given_xprime, g_xprime_given_x = self.get_g_x_xprime_and_g_xprime_x(current_solution, xprime)

            # print('g_x_given_xprime', g_x_given_xprime)
            # print('g_xprime_given_x', g_xprime_given_x)
            # Chaning this because it's now symmetric
            # A = min(1, ratio * g_x_given_xprime / g_xprime_given_x)

            A = min(1, ratio)
            # print('A', A)
            if np.random.uniform() < A:
                # print(i, 'Accepting', xprime)
                current_solution = xprime
                # MCMCSampler.num_transitions += 1
        return current_solution

    def get_next_state(self, counts: tuple[int], current_state: tuple[tuple[int]]):
        if np.random.random() < 0.5:
            return current_state
        removed = self.generate_random_removal(current_state)
        if removed is None:
            return current_state
        counter_removed = Counter(removed)
        xprime = []
        for hh in current_state:
            if counter_removed[hh] == 0:
                xprime.append(hh)
            else:
                counter_removed[hh] -= 1
        # print('Removing', removed)
        # use ip_solve to solve the new subproblem
        all_solutions = self.ip_solve_cached(tup_sum(removed))
        if len(all_solutions) >= MAX_SOLUTIONS:
            raise Exception('Too many solutions')
        # randomly choose one of the solutions
        xprime += all_solutions[np.random.choice(len(all_solutions))]
        xprime = tuple(sorted(xprime))
        if sols_equal(xprime, current_state):
            return current_state
        x_prob = get_log_prob(current_state, self.dist)
        xprime_prob = get_log_prob(xprime, self.dist)
        ratio = np.exp(xprime_prob - x_prob)

        A = min(1, ratio)
        if np.random.uniform() < A:
            return xprime
        else:
            return current_state

def counter_to_tuple(counter: Counter[tuple]):
    return tuple(sorted(list(counter.elements())))

class SimpleMCMCSampler:
    def __init__(self, dist: dict[tuple, int], gamma=1.0):
        self.dist = dist
        self.gamma = gamma

    def get_next_state(self, counts: tuple[int], x: tuple[tuple[int]], dist=None):
        # Make the chain lazy
        x_counter = Counter(x)
        if np.random.random() < 0.5:
            return x
        # sample a random item from self.dict
        if dist is None:
            dist = self.dist
        random_item = random.choice(list(dist.keys()))
        d_max = 0
        cur_item = random_item
        while is_eligible(cur_item, counts):
            d_max += 1
            cur_item = tup_plus(cur_item, random_item)
        if d_max == 0:
            return x
        d = np.random.randint(0, d_max + 1)
        xprime_counter = x_counter.copy()
        xprime_counter[random_item] = d
        if sum(xprime_counter.values()) == 0:
            # empty solution
            return counter_to_tuple(xprime_counter)
        if not is_feasible(list(xprime_counter.elements()), counts):
            return x
        return counter_to_tuple(xprime_counter)

    def mcmc_solve(self, counts: tuple[int], num_iterations=1000):
        current_solution = Counter() # type: Counter[tuple]
        dist = {item: prob for item, prob in self.dist.items() if is_eligible(item, counts)}
        for i in range(num_iterations):
            x_counter = current_solution
            x = counter_to_tuple(x_counter)
            xprime_counter = Counter(self.get_next_state(counts, x, dist))
            if x_counter == xprime_counter:
                continue

            x_prob_diff_sum = self.get_prob_diff_sum(counts, x_counter)
            x_prob = get_log_prob(tuple(current_solution.elements()), self.dist) - self.gamma * x_prob_diff_sum
            xprime_prob_diff_sum = self.get_prob_diff_sum(counts, xprime_counter)
            xprime_prob = get_log_prob(tuple(xprime_counter.elements()), self.dist) - self.gamma * xprime_prob_diff_sum
            ratio = np.exp(xprime_prob - x_prob)
            # print('ratio', ratio)
            # print('g_x_given_xprime', g_x_given_xprime)
            # print('g_xprime_given_x', g_xprime_given_x)
            A = min(1, ratio)
            # print('A', A)
            if np.random.uniform() < A:
                # print(i, 'Accepting', xprime)
                current_solution = xprime_counter
        return current_solution

    def mcmc_sample(self, counts, burn_in=10000, num_samples=10000):
        current_solution = Counter()
        sol_counter = Counter()
        for i in range(burn_in + num_samples):
            if i>= burn_in:
                sol_counter[tuple(sorted(current_solution.elements()))] += 1
            x_counter = current_solution
            xprime_counter = Counter(self.get_next_state(counts, x_counter))
            if x_counter == xprime_counter:
                continue

            x_prob_diff_sum = self.get_prob_diff_sum(counts, x_counter)
            x_prob = get_log_prob(tuple(current_solution.elements()), self.dist) - self.gamma * x_prob_diff_sum
            xprime_prob_diff_sum = self.get_prob_diff_sum(counts, xprime_counter)
            xprime_prob = get_log_prob(tuple(xprime_counter.elements()), self.dist) - self.gamma * xprime_prob_diff_sum
            ratio = np.exp(xprime_prob - x_prob)
            # print('ratio', ratio)
            # print('g_x_given_xprime', g_x_given_xprime)
            # print('g_xprime_given_x', g_xprime_given_x)
            A = min(1, ratio)
            # print('A', A)
            if np.random.uniform() < A:
                # print(i, 'Accepting', xprime)
                current_solution = xprime_counter
        return normalize(sol_counter)

    def get_prob_diff_sum(self, counts, x_counter):
        if sum(x_counter.values()) == 0:
            return sum(counts)
        prob_diff = tup_minus(counts, tup_sum(tuple(x_counter.elements())))
        assert all(x >= 0 for x in prob_diff)
        return sum(prob_diff)

def get_sol_dist(counts, dist, num_samples=400):
    sampler = MCMCSampler(dist, k=3)
    d = Counter()
    for _ in range(num_samples):
        sol = sampler.mcmc_solve(counts)
        d[tuple(sorted(sol))] += 1
    return d

def main():
    dist = {
            (1, 0, 1): 1,
            (0, 1, 1): 1,
            (1, 1, 1): 1,
            (2, 0, 1): 1,
            }
    dist = normalize(dist)
    counts = (5, 1, 5)

    # sampler = SimpleMCMCSampler(dist, gamma=2)

    # sol_counter = sampler.mcmc_sample(counts, burn_in=1000000, num_samples=1000000)
    # print(sol_counter)
    # exact_sols = sum(v for k, v in sol_counter.items() if sampler.get_prob_diff_sum(counts, Counter(k)) == 0)
    # print(exact_sols)
    # actual_sol_dist = normalize({k: v for k, v in sol_counter.items() if sampler.get_prob_diff_sum(counts, Counter(k)) == 0})
    # print(actual_sol_dist)

    # sol_counter = Counter()
    # for _ in range(1000):
        # sol = sampler.mcmc_solve(counts, num_iterations=10000)
        # sol_counter[tuple(sorted(sol.elements()))] += 1
    # sol_counter = normalize(sol_counter)
    # print(sol_counter)
    # exact_sols = sum(v for k, v in sol_counter.items() if sampler.get_prob_diff_sum(counts, Counter(k)) == 0)
    # print(exact_sols)
    # actual_sol_dist = normalize({k: v for k, v in sol_counter.items() if sampler.get_prob_diff_sum(counts, Counter(k)) == 0})
    # print(actual_sol_dist)

    # print(tup_minus(counts, tup_sum(list(sol.elements()))))

    # sampler = MCMCSampler(dist, k=4)
    d = Counter()
    # num_samples = 1000
    # num_iterations = 1000
    pool = mp.Pool(mp.cpu_count())
    dists = pool.starmap(get_sol_dist, [(counts, dist)] * mp.cpu_count())
    d = sum(dists, Counter())
    print('Total samples', sum(d.values()))
    print('Approx error', 1/np.sqrt(sum(d.values())))
    # for _ in range(num_samples):
        # sol = sampler.mcmc_solve((5, 1, 5), num_iterations=num_iterations)
        # d[tuple(sorted(sol))] += 1
    print(normalize(d))
    # print(sampler.ip_solve_cached.cache_info())
    # print('Transition fraction', MCMCSampler.num_transitions / (num_samples * num_iterations))

if __name__ == '__main__':
    main()
