import numpy as np
from functools import cache
from math import log, factorial
# import psutil
from ..utils.knapsack_utils import get_ordering, is_eligible, tup_minus, tup_is_zero, exp_normalize, logsumexp, tup_times, normalize

def knapsack_solve(counts: tuple[int, ...], dist: dict[tuple[int, ...], float]):

    dist = {k: v for k, v in dist.items() if is_eligible(k, counts)}

    ordering = get_ordering(dist)

    print('Approximate number of subproblems', len(ordering) * np.prod([c+1 for c in counts if c > 0]))

    @cache
    def inner_get_log_probs(hh_count, new_counts):
        # print('sol_so_far, new_counts', sol_so_far, len(sol_so_far), '\n', new_counts)
        # if np.random.rand() < .00001:
            # print('Current memory usage', psutil.Process().memory_info().rss / (1024 * 1024), 'MB')
            # print(inner_get_log_probs.cache_info())

        if tup_is_zero(new_counts):
            return (), 0 # log(1)
        if hh_count >= len(ordering):
            return (), float('-inf')
        next_hh = ordering[hh_count]
        if not is_eligible(next_hh, new_counts):
            return (1,), inner_get_log_probs(hh_count+1, new_counts)[1]
        else:
            # figure out how many of next_hh we can add
            n = 0
            dummy_counts = new_counts
            while is_eligible(next_hh, dummy_counts):
                dummy_counts = tup_minus(dummy_counts, next_hh)
                n += 1
            all_log_probs = []
            for i in range(0, n+1):
                next_counts = tup_minus(new_counts, tuple(i * j for j in next_hh))
                _, log_prob = inner_get_log_probs(hh_count+1, next_counts)
                next_log_prob = log(dist[next_hh]**i) + log_prob - log(factorial(i))
                all_log_probs.append(next_log_prob)
            # tuple where element i is exp(all_log_probs[i]), normalized
            dict_to_normalize = {i: p for i, p in enumerate(all_log_probs) if p > float('-inf')}
            if len(dict_to_normalize) == 0:
                return (), float('-inf')
            # print('to normalize', dict_to_normalize)
            normalized = exp_normalize(dict_to_normalize)
            # print('normalized', normalized)
            sample_probs = tuple(normalized[i] if i in normalized else 0 for i in range(len(all_log_probs)))
            return sample_probs, logsumexp(all_log_probs)

    sol_so_far = ()
    # print(counts)
    # print(type(counts))
    inner_get_log_probs(0, counts)
    new_counts = ()
    print(inner_get_log_probs.cache_info())
    # print('Ordering', ordering)
    # print('iglp', inner_get_log_probs((1, 3), (2, 0, 1)))
    while not tup_is_zero(counts) and len(sol_so_far) < len(ordering):
        sample_probs, _ = inner_get_log_probs(len(sol_so_far), counts)
        # print('sample_probs, sol_so_far, counts', sample_probs, sol_so_far, counts)
        if sample_probs == ():
            sol_so_far += (0,)
        else:
            # print('sample_probs', sample_probs)
            sample = np.random.choice(len(sample_probs), p=sample_probs)
            # print('sample', sample)
            new_counts = tup_minus(counts, tup_times(ordering[len(sol_so_far)], sample))
            counts = new_counts
            sol_so_far += (sample,)
    overall_solution = ()
    for i, hh_count in enumerate(sol_so_far):
        overall_solution += (ordering[i],) * hh_count
    return overall_solution

if __name__ == '__main__':
    dist = {
            (1, 0, 1): 1,
            (0, 1, 1): 1,
            (1, 1, 1): 1,
            (2, 0, 1): 1,
            }
    dist = normalize(dist)
    sol = knapsack_solve((5, 1, 5), dist)
    print('Overall solution:', sol)
