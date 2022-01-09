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
        self.level = 1
        self.use_age = True

SOLVER_RESULTS = SolverResults()

def filter_dist(dist, counts):
    return normalize({k: v for k, v in dist.items() if is_eligible(k, counts)})

def recompute_probs(sol, dist):
    if len(sol) == 0:
        return {}
    new_sol = {}
    for seq in sol:
        new_sol[seq] = sum(log(dist[hh]) for hh in seq) + log(perms_to_combs(seq))
    return exp_normalize(new_sol)

def solve(row, all_dists, fallback_dist):
    use_age = has_valid_age_data(row)
    SOLVER_RESULTS.level = 1
    SOLVER_RESULTS.use_age = use_age
    hhs = row_to_hhs(row)
    if len(hhs) == 0:
        return solve_fallback(row, fallback_dist, level=3)
    elif sum(hhs.values()) == 1 and use_age:
        SOLVER_RESULTS.status = SolverResults.OK
        # Need to wrap in 2 tuples
        # Outer one is a list of solutions
        # Inner is the list of households within that solution
        return {(get_race_counts(row) + (get_over_18_total(row),),): 1.0}
    hh_ordering = list(sorted(hhs.keys()))
    counts = get_race_counts(row)
    if use_age:
        counts += get_over_18_counts(row)
    full_counts = counts + tuple(hhs[hh] for hh in hh_ordering)
    full_dist = {}
    n = len(hh_ordering)
    for i, hh in enumerate(hh_ordering):
        try:
            dist = all_dists[hh]
        except KeyError as e:
            print('Missing key', e)
            return solve_fallback(row, fallback_dist)
        i_tup = make_one_hot(i, n)
        for hh_counts, prob in dist.items():
            hh_encoded = hh_counts[:-1]
            if use_age:
                over_18_tup = [0] * len(Race)
                over_18_tup[hh[0].value - 1] = hh_counts[-1]
                hh_encoded += tuple(over_18_tup)
            hh_encoded += i_tup
            full_dist[hh_encoded] = prob

    print('full_counts', full_counts)
    print('hhs', hhs)
    solution = ip_solve(full_counts, full_dist, num_solutions=SOLVER_PARAMS.num_sols)
    if len(solution) >= SOLVER_PARAMS.num_sols:
        SOLVER_RESULTS.status = SolverResults.INCOMPLETE
    else:
        SOLVER_RESULTS.status = SolverResults.OK
    solution = recompute_probs(solution, full_dist)

    if len(solution) == 0:
        return solve_fallback(row, fallback_dist)

    if use_age:
        solution = normalize(decode_solution(solution, decode_1))
    else:
        solution = normalize(decode_solution(solution, decode_1_no_age))
    return solution

def solve_fallback(row, fallback_dist, level=2):
    assert level in (2, 3)
    print('Falling back to level', level)
    use_age = has_valid_age_data(row)
    SOLVER_RESULTS.level = level
    SOLVER_RESULTS.use_age = use_age
    counts = get_race_counts(row)
    if use_age:
        counts += (get_over_18_total(row),)
    if level == 2:
        counts += (get_num_hhs(row),)
    if level == 2:
        if use_age:
            encode = encode_2
            decode = decode_2
        else:
            encode = encode_2_no_age
            decode = decode_2_no_age
    else:
        if use_age:
            encode = encode_3
        else:
            encode = encode_3_no_age
        decode = decode_3

    dist = {encode(hh): prob for hh, prob in fallback_dist.items() if is_eligible(hh, counts)}

    solution = ip_solve(counts, dist, num_solutions=SOLVER_PARAMS.num_sols)
    if len(solution) >= SOLVER_PARAMS.num_sols:
        SOLVER_RESULTS.status = SolverResults.INCOMPLETE
    else:
        SOLVER_RESULTS.status = SolverResults.OK
    solution = recompute_probs(solution, dist)
    if len(solution) == 0:
        if level == 2:
            return solve_fallback(row, fallback_dist, level=3)
        else:
            SOLVER_RESULTS.status = SolverResults.BAD_COUNTS
            return {}
    solution = normalize(decode_solution(solution, decode))
    return solution

def decode_solution(solution, decoder):
    simplified = Counter()
    for hhs, prob in solution.items():
        simp_hhs = tuple(decoder(coded) for coded in hhs)
        simplified[simp_hhs] += prob
    return simplified

if __name__ == '__main__':
    dist = read_microdata(get_micro_file())
    print(len(dist))
