# Used to encode both households and block rows for knapsack solving

from census_utils import *
import numpy as np
import pandas as pd

def encode_hh(hh, holder_race):
    assert len(hh) == len(Race) + 1
    race_counts = hh[:-1]
    n_over_18 = hh[-1]
    over_18_tup = np.zeros(len(Race), dtype=int)
    over_18_tup[holder_race.value - 1] = n_over_18
    over_18_tup = tuple(over_18_tup)
    return race_counts + over_18_tup

def decode_hh(coded):
    return coded[:7] + (sum(coded[7:14]),)

def encode_row(row):
    race_counts = get_race_counts(row)
    age_counts = get_over_18_counts(row)
    return race_counts + age_counts

def encode_row_fallback(row):
    race_counts = get_race_counts(row)
    tot_over_18 = get_over_18_total(row)
    return race_counts + (tot_over_18,)

if __name__ == '__main__':
    print(encode_hh((3, 1, 0, 0, 0, 0, 0, 2), Race.WHITE))
    df = pd.read_csv(get_block_out_file())
    for ind, row in df.iterrows():
        print(encode_row(row))
        1/0
