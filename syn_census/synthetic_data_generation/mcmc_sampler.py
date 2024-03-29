from math import log
from scipy.special import comb
from collections import Counter
from itertools import combinations
from functools import lru_cache
import random
import numpy as np
import multiprocessing as mp
from ..utils.knapsack_utils import perms_to_combs, counter_minus, tup_sum, prod, is_eligible, tup_minus, tup_plus, counter_minus, is_feasible, normalize, exp_normalize, scipy_multinomial, exp_noramlize_list
from ..utils.ip_distribution import ip_solve

def get_log_prob(sol, dist):
    if len(sol) == 0:
        return 0
    return sum(log(dist[hh]) for hh in sol) + log(perms_to_combs(sol))

def get_log_prob_vector(sol: np.ndarray, dist: dict):
    if len(sol) == 0:
        return 0
    nonzero = np.nonzero(sol)[0]
    s = 0
    for i in nonzero:
        s += log(dist[i]) * sol[i]
    s += log(scipy_multinomial(sol[nonzero]))
    return s

def sols_equal(sol1, sol2):
    c = Counter()
    for hh in sol1:
        c[hh] += 1
    for hh in sol2:
        c[hh] -= 1
    return all(v == 0 for v in c.values())

class MCMCSampler:
    # num_transitions = 0
    def __init__(self, dist, num_iterations=1000, k=5, max_solutions=1000):
        self.dist = dist
        self.num_iterations = num_iterations
        self.k = k
        self.all_hhs = sorted(self.dist.keys())
        self.V = np.array([hh for hh in self.all_hhs], dtype=np.int64).T
        self.hh_map = {hh: i for i, hh in enumerate(self.all_hhs)}
        self.index_dist = {i: self.dist[hh] for i, hh in enumerate(self.all_hhs)}
        self.max_solutions = max_solutions

    @lru_cache(maxsize=None)
    def ip_solve_cached(self, counts):
        #TODO use ip_enumerate
        # print(counts)
        solutions = ip_solve(counts, self.dist, num_solutions=self.max_solutions)
        if len(solutions) == 0:
            raise Exception('No solutions')
        elif len(solutions) >= self.max_solutions:
            raise Exception('Too many solutions. This may mean that k is too large.')
        return solutions

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

    def get_sampling_num_vector(self, x: np.ndarray, removal: np.ndarray):
        removal_nonzero = np.nonzero(removal)[0]
        p = 1
        for a in removal_nonzero:
            assert x[a] >= removal[a]
            p *= comb(x[a], removal[a])
        return p

    def generate_random_removal(self, x: tuple):
        # remove k elements at random
        remove_indices = np.random.choice(len(x), self.k, replace=False)
        # y = tuple(a for i, a in enumerate(x) if i not in remove_indices)
        removed = tuple(sorted(x[i] for i in remove_indices))
        if np.random.random() < 1/self.get_sampling_num(Counter(x), Counter(removed)):
            return removed
        else:
            return None
    
    def generate_random_removal_vector(self, x: np.ndarray):
        # remove k elements at random
        nonzeros = np.nonzero(x)[0]
        nonzeros_with_multiplicity = np.repeat(nonzeros, x[nonzeros])
        remove_indices = np.random.choice(nonzeros_with_multiplicity, self.k, replace=False)
        removal = np.bincount(remove_indices, minlength=len(x))
        if np.random.random() < 1/self.get_sampling_num_vector(x, removal):
            return removal
        else:
            return None

    def mcmc_solve(self, counts: tuple, initial_solution: tuple=None):
        if initial_solution is None:
            current_solution = ip_solve(counts, self.dist, num_solutions=1)[0]
        else:
            current_solution = initial_solution
        current_solution_array = np.zeros(len(self.dist), dtype=np.int64)
        for hh in current_solution:
            current_solution_array[self.hh_map[hh]] += 1
        assert(len(current_solution) > self.k)
        for _ in range(self.num_iterations):
            # Make the chain lazy
            if np.random.random() < 0.5:
                continue
            # randomly remove k items from the solution
            removed = self.generate_random_removal_vector(current_solution_array)
            if removed is None:
                continue
            xprime = current_solution_array - removed
            # use ip_solve to solve the new subproblem
            # TODO change this to ip_enumerate?
            all_solutions = self.ip_solve_cached(tuple(self.V.dot(removed)))
            if len(all_solutions) >= self.max_solutions:
                # this means our choice of k was too large
                raise Exception('Too many solutions; k is too large')
            if len(all_solutions) == 1:
                continue
            candidates = []
            log_probs = []
            for solution in all_solutions:
                solution_array = np.zeros(len(self.dist), dtype=np.int64)
                for hh in solution:
                    solution_array[self.hh_map[hh]] += 1
                candidate = xprime + solution_array
                candidates.append(candidate)
                log_probs.append(get_log_prob_vector(candidate, self.index_dist))
            normalized_probs = exp_noramlize_list(log_probs)
            current_solution = random.choices(candidates, weights=normalized_probs)[0]
        solution_nonzero = np.nonzero(current_solution)[0]
        solution_tuple = tuple()
        for i in solution_nonzero:
            solution_tuple += (self.all_hhs[i],) * current_solution[i]
        return solution_tuple

    def get_next_state(self, counts: tuple, current_state: tuple):
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
        if len(all_solutions) >= self.max_solutions:
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

def counter_to_tuple(counter: Counter):
    return tuple(sorted(list(counter.elements())))

class SimpleMCMCSampler:
    def __init__(self, dist: dict, gamma=1.0):
        self.dist = dist
        self.gamma = gamma
        self.all_hhs = sorted(self.dist.keys())
        self.V = np.array([hh for hh in self.all_hhs], dtype=np.int64).T
        self.hh_map = {hh: i for i, hh in enumerate(self.all_hhs)}
        self.index_dist = {i: self.dist[hh] for i, hh in enumerate(self.all_hhs)}

    @lru_cache(maxsize=None)
    def get_eligible_indices(self, counts: tuple):
        eligible = [i for i in range(len(self.all_hhs)) if is_eligible(self.all_hhs[i], counts)]

        return eligible

    def get_next_state_vector(self, counts: tuple, x: np.ndarray, cache: dict = {}):
        if np.random.random() < 0.5:
            return x
        index = np.random.choice(self.get_eligible_indices(counts))
        x[index] = 0
        tup_x_index = (tuple(x), index)
        array_counts = np.array(counts)
        if tup_x_index not in cache:
            prev_prob = 1
            # neighbor_dist = {}
            nonzero = np.nonzero(x)[0]
            if len(nonzero) > 0:
                s_norm_without_i = sum(x[nonzero])
            else:
                s_norm_without_i = 0
            v_i_norm = sum(self.all_hhs[index])
            j = 0
            neighbor_list = []
            prob_list = []
            remainder = array_counts - self.V.dot(x)
            while(np.all(j * self.V[:, index] <= remainder)):
                # update probs
                if j == 0:
                    new_prob = 1
                else:
                    new_prob = prev_prob * ((s_norm_without_i + j) / j) * self.index_dist[index] * np.exp(self.gamma*v_i_norm)
                neighbor_list.append(tuple(x))
                prob_list.append(new_prob)
                prev_prob = new_prob
                x[index] += 1
                j += 1
            total = sum(prob_list)
            normalized_prob_list = [p/total for p in prob_list]
            cache[tup_x_index] = neighbor_list, normalized_prob_list
        else:
            neighbor_list, normalized_prob_list = cache[tup_x_index]
        next_x = np.array(random.choices(neighbor_list, normalized_prob_list), dtype=np.int64).flatten()
        return next_x

    def get_next_state(self, counts: tuple, x: tuple, dist=None):
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
        cur_amt = x_counter[random_item]
        d = cur_amt
        # This is a bit of a hack
        while d == cur_amt:
            d = np.random.randint(0, d_max + 1)
        xprime_counter = x_counter.copy()
        xprime_counter[random_item] = d
        if sum(xprime_counter.values()) == 0:
            # empty solution
            return counter_to_tuple(xprime_counter)
        if not is_feasible(list(xprime_counter.elements()), counts):
            return x
        return counter_to_tuple(xprime_counter)

    def mcmc_iterate(self, counts: tuple, num_iterations: int, cache: dict={}):
        x = np.zeros(len(self.all_hhs), dtype=np.int64)
        for _ in range(num_iterations):
            x = self.get_next_state_vector(counts, x, cache)
        return x

    def mcmc_solve(self, counts: tuple, num_iterations=1000):
        cache = {}
        # TODO num_iterations consistent
        x = np.zeros(len(self.all_hhs), dtype=np.int64)
        while np.any(self.V.dot(x) != counts):
            x = self.mcmc_iterate(counts, num_iterations, cache)
        solution_nonzero = np.nonzero(x)[0]
        solution_tuple = tuple()
        for i in solution_nonzero:
            solution_tuple += (self.all_hhs[i],) * x[i]
        return solution_tuple

    # def mcmc_sample(self, counts, burn_in=10000, num_samples=10000):
        # current_solution = Counter()
        # sol_counter = Counter()
        # for i in range(burn_in + num_samples):
            # if i>= burn_in:
                # sol_counter[tuple(sorted(current_solution.elements()))] += 1
            # x_counter = current_solution
            # x = counter_to_tuple(x_counter)
            # xprime_counter = Counter(self.get_next_state(counts, x))
            # if x_counter == xprime_counter:
                # continue

            # x_prob_diff_sum = self.get_prob_diff_sum(counts, x_counter)
            # x_prob = get_log_prob(tuple(current_solution.elements()), self.dist) - self.gamma * x_prob_diff_sum
            # xprime_prob_diff_sum = self.get_prob_diff_sum(counts, xprime_counter)
            # xprime_prob = get_log_prob(tuple(xprime_counter.elements()), self.dist) - self.gamma * xprime_prob_diff_sum
            # ratio = np.exp(xprime_prob - x_prob)
            # # print('ratio', ratio)
            # # print('g_x_given_xprime', g_x_given_xprime)
            # # print('g_xprime_given_x', g_xprime_given_x)
            # A = min(1, ratio)
            # # print('A', A)
            # if np.random.uniform() < A:
                # # print(i, 'Accepting', xprime)
                # current_solution = xprime_counter
        # return normalize(sol_counter)

    def get_prob_diff_sum(self, counts, x_counter):
        if sum(x_counter.values()) == 0:
            return sum(counts)
        prob_diff = tup_minus(counts, tup_sum(tuple(x_counter.elements())))
        assert all(x >= 0 for x in prob_diff)
        return sum(prob_diff)

def get_prob_diff_sum(counts, x_counter):
    if sum(x_counter.values()) == 0:
        return sum(counts)
    prob_diff = tup_minus(counts, tup_sum(tuple(x_counter.elements())))
    assert all(x >= 0 for x in prob_diff)
    return sum(prob_diff)

def get_sol_dist_reduced(counts, dist, num_samples=400):
    sampler = MCMCSampler(dist, k=3)
    d = Counter()
    for _ in range(num_samples):
        sol = sampler.mcmc_solve(counts)
        d[tuple(sorted(sol))] += 1
    return d

def get_sol_dist_simple(counts, dist, num_samples=25):
    sampler = SimpleMCMCSampler(dist, gamma=2)
    d = Counter()
    for _ in range(num_samples):
        sol = sampler.mcmc_solve(counts, num_iterations=20000)
        d[tuple(sorted(sol))] += 1
    return d

def run_test():
    dist = {
            (1, 0, 1): 1,
            (0, 1, 1): 1,
            (1, 1, 1): 1,
            (2, 0, 1): 1,
            }
    dist = normalize(dist)
    counts = (5, 1, 5)

    print('Testing simple')
    d = Counter()
    num_threads = mp.cpu_count()
    pool = mp.Pool(num_threads)
    dists = pool.starmap(get_sol_dist_simple, [(counts, dist)] * num_threads)
    d = sum(dists, Counter())
    print('Total samples', sum(d.values()))
    print('Approx error', 1/np.sqrt(sum(d.values())))
    print(normalize(d))
    all_solutions = ip_solve(counts, dist)
    solution_dist = {solution: get_log_prob(solution, dist) for solution in all_solutions}
    print(exp_normalize(solution_dist))

    print('Testing reduced')
    d = Counter()
    num_threads = mp.cpu_count()
    pool = mp.Pool(num_threads)
    dists = pool.starmap(get_sol_dist_reduced, [(counts, dist)] * num_threads)
    d = sum(dists, Counter())
    print('Total samples', sum(d.values()))
    print('Approx error', 1/np.sqrt(sum(d.values())))
    print(normalize(d))
    all_solutions = ip_solve(counts, dist)
    solution_dist = {solution: get_log_prob(solution, dist) for solution in all_solutions}
    print(exp_normalize(solution_dist))
