import pandas as pd
from collections import Counter
import re
from census_utils import *
from knapsack_utils import *

class Household():
    def __init__(self, is_family, n_under_18):
        self.holder_race = None
        self.is_family = is_family
        self.people = []
        self.n_under_18 = n_under_18

    @property
    def size(self):
        return len(self.people)

    @property
    def hh_info(self):
        assert len(self.people) >= self.n_under_18
        # Format: (holder race, is family, size (capped at 7))
        return (self.holder_race, self.is_family, min(self.size, 7))

    @property
    def to_tuple(self):
        assert len(self.people) >= self.n_under_18
        c = Counter()
        for person in self.people:
            c[person.race] += 1
        return tuple(c[race] for race in Race) + (len(self.people) - self.n_under_18,)

class Person():
    def __init__(self, race):
        self.race = race

def read_microdata(fname):
    # will map from (holder race X is_family X size) to a distribution over HHs
    all_dists = {}
    with open(fname) as f:
        hh_data = None
        weight = None
        for i, line in enumerate(f):
            if re.match('^P', line):
                race = get_race_from_p_record(line)
                if hh_data.holder_race == None:
                    hh_data.holder_race = race
                hh_data.people.append(Person(race))
            else:
                if hh_data is not None and hh_data.holder_race is not None:
                    if hh_data.hh_info not in all_dists:
                        all_dists[hh_data.hh_info] = Counter()
                    all_dists[hh_data.hh_info][hh_data.to_tuple] += weight
                hh_data = Household(get_is_family_from_h_record(line), get_n_under_18_from_h_record(line))
                weight = get_weight_from_h_record(line)
                # TODO: figure out if weights are meaningful
                weight = 1
        if hh_data is not None and hh_data.holder_race is not None:
            if hh_data.hh_info not in all_dists:
                all_dists[hh_data.hh_info] = Counter()
            all_dists[hh_data.hh_info][hh_data.to_tuple] += 1
        fallback_dist = Counter()
        for hh_data, dist in all_dists.items():
            for k, v in dist.items():
                fallback_dist[k] += v
        fallback_dist = Counter(normalize(fallback_dist))
        all_dists = {info: normalize(dist) for info, dist in all_dists.items()}
        return all_dists, fallback_dist

if __name__ == '__main__':
    print('Testing microdata build')
    all_dists, fallback_dist = read_microdata(get_micro_file())
    print(len(all_dists), 'unique (HH race, is_family, size) combinations')
    print(list(all_dists.keys()))
    print(fallback_dist.most_common(20))
