from collections import Counter
from functools import lru_cache, reduce
import scipy
from scipy.special import comb
import numpy as np
import operator

def prod(seq):
    return reduce(operator.mul, seq, 1)

def normalize(dist):
    tot = sum(dist.values())
    return {k: v/tot for k, v in dist.items()}

def exp_normalize(dist):
    m = max(dist.values())
    dist = {k: np.exp(v - m) for k, v in dist.items()}
    return normalize(dist)

def get_ordering(dist):
    return sorted(list(dist.keys()))

def is_eligible(house, counts):
    return all(i <= j for i, j in zip(house, counts))

def tup_minus(a, b):
    return tuple(i - j for i, j in zip(a, b))

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

def make_one_hot(i, n):
    l = [0] * n
    l[i] = 1
    return tuple(l)

def make_one_hot_np(i, n):
    return np.array(make_one_hot(i, n), dtype=int)
