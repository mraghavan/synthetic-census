import gurobipy as gp
import numpy as np
from gurobipy import GRB
from math import log
from .knapsack_utils import get_ordering, normalize
from functools import lru_cache

@lru_cache(maxsize=1000)
def ip_enumerate(counts: tuple, elements: tuple, num_solutions=50):
    """
    Enumerate all solutions to the knapsack problem with the given counts and elements.
    Faster than ip_solve because it can be cached.
    """
    return ip_solve(counts, {e: 1 for e in elements}, num_solutions=num_solutions)

def ip_solve(counts: tuple, dist: dict, num_solutions=50):
    ordering = get_ordering(dist)
    constraint_mat = np.array(ordering).T
    # print(constraint_mat)

    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = gp.Model('IP solver', env=env)
    m.Params.LogToConsole = 0
    m.Params.PoolSearchMode = 2
    m.Params.PoolSolutions = num_solutions
    # var_dict = {i: m.addVar(vtype=GRB.INTEGER, name="x" + str(i)) for i, house in enumerate(ordering) if is_eligible(house, counts)}
    # var_list = np.array([var_dict[i] for i in range(len(var_dict))])
    x = m.addMVar(len(ordering), vtype=GRB.INTEGER, lb=0)
    nl_probs = np.array([-log(dist[i]) for i in ordering])
    # prob of sequence + approx multinomial correction
    # m.setObjective(nl_probs @ x + np.ones((len(ordering),)) @ x)
    m.setObjective(nl_probs @ x, GRB.MINIMIZE)
    m.addConstr(constraint_mat @ x == np.array(counts))
    m.optimize()
    # for v in m.getVars():
        # print('%s %g' % (v.varName, v.x))

    # print('Obj: %g' % m.objVal)
    nSolutions = m.SolCount
    # print('Number of solutions found: ' + str(nSolutions))
    # for sol in range(nSolutions):
        # m.setParam(GRB.Param.SolutionNumber, sol)
        # print(m.PoolObjVal)
        # values = m.Xn
        # for v, h in zip(values, ordering):
            # if v > 0:
                # print(h, ':', int(v))

    sols = []
    for sol in range(nSolutions):
        m.setParam(GRB.Param.SolutionNumber, sol)
        values = m.Xn
        current_sol = []
        for v, h in zip(values, ordering):
            if v > 0:
                current_sol += [h] * round(v)
        current_sol = tuple(current_sol)
        sols.append(current_sol)
    env.dispose()
    return sols

if __name__ == '__main__':
    dist = {
            (1, 0, 1): 1,
            (0, 1, 1): 1,
            (1, 1, 1): 1,
            (2, 0, 1): 1,
            }
    dist = normalize(dist)
    sol = ip_solve((5, 1, 5), dist, num_solutions=3)
    print(sol)
