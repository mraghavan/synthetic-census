from math import log
from collections import OrderedDict, Counter
from ..utils.census_utils import has_valid_age_data
from ..utils.knapsack_utils import normalize, exp_normalize, perms_to_combs, tup_sum, tup_minus
from ..utils.ip_distribution import ip_solve
from ..utils.encoding import encode_row, get_num_hhs, MAX_LEVEL

class SolverParams():
    def __init__(self, num_sols):
        self.num_sols = num_sols

SOLVER_PARAMS = SolverParams(100)

class SolverResults():
    OK = 'Ok'
    UNSOLVED = 'Unsolved'
    INCOMPLETE = 'Not all solutions found'

    def __init__(self):
        self.status = SolverResults.OK
        self.level = 1
        self.use_age = True

SOLVER_RESULTS = SolverResults()

def recompute_probs(sol, dist):
    if len(sol) == 0:
        return OrderedDict()
    new_sol = OrderedDict()
    for seq in sol:
        new_sol[seq] = sum(log(dist[hh]) for hh in seq) + log(perms_to_combs(seq))
    return exp_normalize(new_sol)

def recompute_probs_gamma(sol, dist, counts, gamma):
    if len(sol) == 0:
        return OrderedDict()
    new_sol = OrderedDict()
    for seq in sol:
        if seq:
            new_sol[seq] = sum(log(dist[hh]) for hh in seq) + log(perms_to_combs(seq)) - gamma*sum(tup_minus(counts, tup_sum(seq)))
        else:
            new_sol[seq] = -gamma * sum(counts)
    return exp_normalize(new_sol)

def reduce_dist(dist, level, use_age):
    c = Counter()
    for k, v in dist.items():
        c[k.reduce(level, use_age)] += v
    return normalize(c)

def solve(row, dist, level=1):
    SOLVER_RESULTS.level = level
    if level > MAX_LEVEL:
        SOLVER_RESULTS.status = SolverResults.UNSOLVED
        return OrderedDict()
    solve_dist = dist
    use_age = has_valid_age_data(row)
    SOLVER_RESULTS.use_age = use_age
    counts = encode_row(row)
    if get_num_hhs(row) == 1 and use_age:
        SOLVER_RESULTS.status = SolverResults.OK
        sol = OrderedDict({(counts,): 1.0})
        return sol
    if level > 1:
        solve_dist = reduce_dist(dist, level, use_age)
        counts = counts.reduce(level, use_age)
    print(counts)
    sol = ip_solve(counts, solve_dist, num_solutions=SOLVER_PARAMS.num_sols)
    if len(sol) == 0:
        return solve(row, dist, level + 1)
    if len(sol) == SOLVER_PARAMS.num_sols:
        SOLVER_RESULTS.status = SolverResults.INCOMPLETE
    else:
        SOLVER_RESULTS.status = SolverResults.OK
    sol = recompute_probs(sol, solve_dist)
    return sol
