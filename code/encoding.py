# Used to encode both households and block rows for knapsack solving

from census_utils import *
import numpy as np
import pandas as pd

def decode_1(coded):
    return coded[:7] + (sum(coded[7:14]),)

def decode_1_no_age(coded):
    return coded[:len(Race)]

def encode_2(hh):
    return hh + (1,)

def encode_2_no_age(hh):
    return hh[:-1] + (1,)

def decode_2(coded):
    return coded[:len(Race) + 1]

def decode_2_no_age(coded):
    return coded[:len(Race)]

def encode_3(hh):
    return hh

def encode_3_no_age(hh):
    return hh[:-1]

def decode_3(coded):
    return coded

if __name__ == '__main__':
    print(encode_hh((3, 1, 0, 0, 0, 0, 0, 2), Race.WHITE))
    df = pd.read_csv(get_block_out_file())
    for ind, row in df.iterrows():
        print(encode_row(row))
        1/0
