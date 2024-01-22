from collections import Counter, OrderedDict
from functools import reduce
from scipy.special import comb
import numpy as np
import operator

def prod(seq):
    return reduce(operator.mul, seq, 1)

def normalize(dist):
    tot = sum(dist.values())
    return OrderedDict({k: v/tot for k, v in dist.items()})

def exp_normalize(dist: dict):
    m = max(dist.values())
    dist = OrderedDict({k: np.exp(v - m) for k, v in dist.items()})
    return normalize(dist)

def exp_noramlize_list(l: list):
    m = max(l)
    new_l = [np.exp(i - m) for i in l]
    s = sum(new_l)
    return [i / s for i in new_l]

def get_ordering(dist: dict):
    return sorted(list(dist.keys()))

def is_eligible(house, counts):
    return all(i <= j for i, j in zip(house, counts))

def tup_minus(a, b):
    return tuple(i - j for i, j in zip(a, b))

def tup_plus(a, b):
    return tuple(i + j for i, j in zip(a, b))

def tup_sum(l):
    assert len(l) > 0
    s = l[0]
    for i in l[1:]:
        s = tup_plus(s, i)
    return s

def counter_minus(c1, c2):
    # return c1 - c2 where c1 and c2 are Counters
    diff = Counter({k: v - c2[k] for k, v in c1.items()})
    assert all(v >= 0 for v in diff.values())
    return diff

def is_feasible(x, counts):
    return all(i <= j for i, j in zip(tup_sum(x), counts))

def tup_times(a: tuple, b: int):
    return tuple(i * b for i in a)

def tup_is_zero(tup):
    return all(i == 0 for i in tup)

def tup_is_nonneg(tup):
    return all(i >= 0 for i in tup)

def perms_to_combs(seq):
    c = Counter(seq)
    t = tuple(c.values())
    return scipy_multinomial(t)

# @lru_cache(maxsize=None)
def scipy_multinomial(params):
    if len(params) == 1:
        return 1
    coeff = (comb(sum(params), params[-1], exact=True) *
            scipy_multinomial(params[:-1]))
    return coeff

def make_one_hot(i: int, n: int):
    l = [0] * n
    l[i] = 1
    return tuple(l)

def make_one_hot_np(i: int, n: int):
    return np.array(make_one_hot(i, n), dtype=int)

def logsumexp(seq):
    m = max(seq)
    return m + np.log(sum(np.exp(i - m) for i in seq))
