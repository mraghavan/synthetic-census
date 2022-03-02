import pandas as pd
from collections import Counter
import re
from census_utils import *
from knapsack_utils import *
from collections import namedtuple

def rh_to_str(rh):
    r, h = rh
    return r_to_str(r) + '_' + str(h)

def r_to_str(r):
    return str(r).lstrip('Race.')

def t_to_str(t):
    r, f, s = t
    if type(r) == Race:
        return r_to_str(r) + '_' + str(f) + '_' + str(s)
    else:
        return 'H' + '_' + str(f) + '_' + str(s)

HH_tup = namedtuple('HH_tup',
        [rh_to_str(rh) for rh in RACE_HIS_ENUM] + ['n_18', 'holder_race', 'holder_eth', 'is_family', 'size'])

class Household():
    def __init__(self, is_family, n_under_18):
        self.holder = None
        self.is_family = is_family
        self.people = []
        self.n_under_18 = n_under_18

    @property
    def size(self):
        return len(self.people)

    @property
    def rh_counts(self):
        c = Counter()
        for person in self.people:
            c[(person.race, person.eth)] += 1
        return tuple(c[rh] for rh in RACE_HIS_ENUM)

    @property
    def race_counts(self):
        c = Counter()
        for person in self.people:
            c[person.race] += 1
        return tuple(c[r] for r in Race)

    @property
    def eth_count(self):
        c = 0
        for person in self.people:
            c += person.eth
        return c

    def fix_family(self):
        if self.size == 1:
            self.is_family = False

    @property
    def to_tuple(self):
        c = Counter()
        for person in self.people:
            c[(person.race, person.eth)] += 1
        t = self.rh_counts + (self.n_over_18,) + (self.holder.race, self.holder.eth, self.is_family, min(self.size, 7))
        return HH_tup(*t)

    @property
    def race_type(self):
        return (self.holder.race, self.is_family, min(self.size, 7))

    @property
    def eth_type(self):
        return (self.holder.eth, self.is_family, min(self.size, 7))

    @property
    def n_over_18(self):
        assert len(self.people) >= self.n_under_18
        return len(self.people) - self.n_under_18

    def __hash__(self):
        return hash(self.to_tuple)

    def __eq__(self, other):
        return self.to_tuple == other.to_tuple

    def __repr__(self):
        return str(self.to_tuple)

class Person():
    def __init__(self, race, eth):
        self.race = race
        self.eth = eth

def read_microdata(fname):
    dist = Counter()
    with open(fname) as f:
        hh_data = None
        weight = None
        for i, line in enumerate(f):
            if re.match('^P', line):
                race = get_race_from_p_record(line)
                eth = get_eth_from_p_record(line)
                if hh_data.holder == None:
                    hh_data.holder = Person(race, eth)
                hh_data.people.append(Person(race, eth))
            else:
                if hh_data is not None and hh_data.holder is not None:
                    hh_data.fix_family()
                    dist[hh_data] += weight
                    # if hh_data.hh_info not in all_dists:
                        # all_dists[hh_data.hh_info] = Counter()
                    # all_dists[hh_data.hh_info][hh_data.to_tuple] += weight
                hh_data = Household(get_is_family_from_h_record(line), get_n_under_18_from_h_record(line))
                weight = get_weight_from_h_record(line)
                # TODO: figure out if weights are meaningful
                weight = 1
        if hh_data is not None and hh_data.holder is not None:
            hh_data.fix_family()
            dist[hh_data] += weight
            # if hh_data.hh_info not in all_dists:
                # all_dists[hh_data.hh_info] = Counter()
            # all_dists[hh_data.hh_info][hh_data.to_tuple] += 1
        # fallback_dist = Counter()
        # for hh_data, dist in all_dists.items():
            # for k, v in dist.items():
                # fallback_dist[k] += v
        # fallback_dist = Counter(normalize(fallback_dist))
        # all_dists = {info: normalize(dist) for info, dist in all_dists.items()}
        return Counter(normalize(dist))

if __name__ == '__main__':
    print('Testing microdata build')
    dist = read_microdata(get_micro_file())
    print(len(dist), 'unique HHs')
    # print(list(all_dists.keys()))
    print(dist.most_common(10))
