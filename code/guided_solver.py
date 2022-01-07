from build_micro_dist import *
from knapsack_utils import *
import multiprocessing
import time
from ip_distribution import ip_solve
from encoding import *
from math import log

class SolverParams():
    def __init__(self, num_sols):
        self.num_sols = num_sols

SOLVER_PARAMS = SolverParams(100)

class SolverResults():
    OK = 'Ok'
    BAD_DIST = 'Insufficient distribution'
    FALLBACK = 'Used fallback solver'
    FALLBACK_INCOMPLETE = 'Used fallback solver, not all solutions found'
    BAD_COUNTS = 'Counts don\'t match'
    INCOMPLETE = 'Not all solutions found'

    def __init__(self):
        self.status = SolverResults.OK

SOLVER_RESULTS = SolverResults()

def filter_dist(dist, counts):
    return normalize({k: v for k, v in dist.items() if is_eligible(k, counts)})

def recompute_probs(sol, dist):
    new_sol = {}
    for seq in sol:
        new_sol[seq] = np.exp(sum(log(dist[hh]) for hh in seq) + log(perms_to_combs(seq)))
    return normalize(new_sol)

def solve(row, all_dists, fallback_dist):
    hhs = row_to_hhs(row)
    if len(hhs) == 0:
        SOLVER_RESULTS.status = SolverResults.BAD_DIST
        print('Returning early')
        return {}
    hh_ordering = list(sorted(hhs.keys()))
    counts = encode_row(row)
    full_counts = counts + tuple(hhs[hh] for hh in hh_ordering)
    full_dist = {}
    n = len(hh_ordering)
    for i, hh in enumerate(hh_ordering):
        try:
            dist = all_dists[hh]
        except KeyError as e:
            print('Missing key', e)
            SOLVER_RESULTS.status = SolverResults.BAD_DIST
            return {}
        i_tup = [0] * n
        i_tup[i] = 1
        i_tup = tuple(i_tup)
        for hh_counts, prob in dist.items():
            hh_encoded = encode_hh(hh_counts, hh[0]) + i_tup
            full_dist[hh_encoded] = prob

    print('counts', counts)
    print('full_counts', full_counts)
    print('hhs', hhs)
    solution = recompute_probs(ip_solve(full_counts, full_dist, num_solutions=SOLVER_PARAMS.num_sols), full_dist)
    if len(solution) == 0:
        SOLVER_RESULTS.status = SolverResults.BAD_COUNTS
        return solve_race_counts_only(row, fallback_dist)
    if len(solution) >= SOLVER_PARAMS.num_sols:
        SOLVER_RESULTS.status = SolverResults.INCOMPLETE
    else:
        SOLVER_RESULTS.status = SolverResults.OK
    solution = normalize(simplify_solution(solution))
    return solution

def solve_race_counts_only(row, fallback_dist):
    # encode as (race_counts) + (total_num_over_18)
    print('Falling back to second solver')
    counts = encode_row_fallback(row)
    print('counts', counts)
    solution = recompute_probs(ip_solve(counts, fallback_dist, num_solutions=SOLVER_PARAMS.num_sols), fallback_dist)
    if len(solution) == 0:
        SOLVER_RESULTS.status = SolverResults.BAD_COUNTS
        return {}
    if len(solution) >= SOLVER_PARAMS.num_sols:
        SOLVER_RESULTS.status = SolverResults.FALLBACK_INCOMPLETE
    else:
        SOLVER_RESULTS.status = SolverResults.FALLBACK
    solution = normalize(simplify_solution(solution))
    return solution

def simplify_solution(solution):
    simplified = Counter()
    for hhs, prob in solution.items():
        simp_hhs = tuple(decode_hh(coded) for coded in hhs)
        simplified[simp_hhs] += prob
    return simplified

if __name__ == '__main__':
    dist = read_microdata(get_micro_file())
    print(len(dist))
