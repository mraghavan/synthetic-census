from build_micro_dist import *
from census_utils import Race
from encoding import *
from knapsack_utils import *
from ip_distribution import ip_solve
from encoding import *
from math import log

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
        return {}
    new_sol = {}
    for seq in sol:
        new_sol[seq] = sum(log(dist[hh]) for hh in seq) + log(perms_to_combs(seq))
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
        return {}
    solve_dist = dist
    use_age = has_valid_age_data(row)
    SOLVER_RESULTS.use_age = use_age
    counts = encode_row(row)
    if get_num_hhs(row) == 1 and use_age:
        SOLVER_RESULTS.status = SolverResults.OK
        sol = {(counts,): 1.0}
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
