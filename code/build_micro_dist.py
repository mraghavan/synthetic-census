import pandas as pd
from collections import Counter
import re
from census_utils import *
from knapsack_utils import *
from collections import namedtuple
import numpy as np

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
            # Sometimes there are discrepancies in the data
            # If there's a household with just one 'adult' but they're under 18,
            # make them 18 for consistency.
            if self.people[0].age < 18 and self.n_over_18 == 1:
                self.people[0].age = 18
        # If this breaks, we'll need to do more data consistency investigation
        assert sum(p.age >= 18 for p in self.people) == self.n_over_18

    @property
    def to_tuple(self):
        t = self.rh_counts + (self.n_over_18,) + (self.holder.race, self.holder.eth, self.is_family, min(self.size, 7))
        return HH_tup(*t)

    @property
    def to_tuple_granular(self):
        c = []
        for person in self.people:
            c += [(person.race, person.eth, person.age >= 18)]
        return tuple(sorted(c))

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
    def __init__(self, race, eth, age):
        self.race = race
        self.eth = eth
        self.age = age

    def __str__(self):
        return '%s %d %d' % (self.race, self.eth, self.age)

def read_microdata(fname):
    dist = Counter()
    with open(fname) as f:
        hh_data = None
        weight = None
        for i, line in enumerate(f):
            if re.match('^P', line):
                race = get_race_from_p_record(line)
                eth = get_eth_from_p_record(line)
                age = get_age_from_p_record(line)
                if hh_data.holder == None:
                    hh_data.holder = Person(race, eth, age)
                hh_data.people.append(Person(race, eth, age))
            else:
                if hh_data is not None and hh_data.holder is not None:
                    hh_data.fix_family()
                    dist[hh_data] += weight
                hh_data = Household(get_is_family_from_h_record(line), get_n_under_18_from_h_record(line))
                weight = get_weight_from_h_record(line)
                # TODO: figure out if weights are meaningful
                weight = 1
        if hh_data is not None and hh_data.holder is not None:
            hh_data.fix_family()
            dist[hh_data] += weight
        return Counter(normalize(dist))

def read_microdata_granular(fname):
    with open(fname) as f:
        hh_data = None
        weight = None
        dist_map = {}
        for i, line in enumerate(f):
            if re.match('^P', line):
                race = get_race_from_p_record(line)
                eth = get_eth_from_p_record(line)
                age = get_age_from_p_record(line)
                if hh_data.holder == None:
                    hh_data.holder = Person(race, eth, age)
                hh_data.people.append(Person(race, eth, age))
            else:
                if hh_data is not None and hh_data.holder is not None:
                    hh_data.fix_family()
                    key = hh_data.race_counts + (hh_data.eth_count,) + (hh_data.n_over_18,)
                    if key not in dist_map:
                        dist_map[key] = Counter()
                    dist_map[key][hh_data.to_tuple_granular] += 1
                hh_data = Household(get_is_family_from_h_record(line), get_n_under_18_from_h_record(line))
        if hh_data is not None and hh_data.holder is not None:
            hh_data.fix_family()
            key = hh_data.race_counts + (hh_data.eth_count,) + (hh_data.n_over_18,)
            if key not in dist_map:
                dist_map[key] = Counter()
            dist_map[key][hh_data.to_tuple_granular] += 1
        new_dist_map = {k: normalize(v) for k, v in dist_map.items()}
        return new_dist_map

if __name__ == '__main__':
    print('Testing microdata build')
    dist = read_microdata(get_micro_file())
    print(len(dist), 'unique HHs')
    # print(list(all_dists.keys()))
    print(dist.most_common(10))
