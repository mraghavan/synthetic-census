from census_utils import *
from knapsack_utils import *
import pandas as pd
from build_micro_dist import read_microdata
from collections import Counter

NON_BLOCK_COLS_L = [
        'TOTAL',
        'W',
        'B',
        'AI_AN',
        'AS',
        'H_PI',
        'OTH',
        'TWO_OR_MORE',
        '18_PLUS',
        'HH_NUM',
        ]

HH_COLS = NON_BLOCK_COLS_L[1:-1]

NON_BLOCK_COLS = set(NON_BLOCK_COLS_L)

def tvd(d1, d2):
    all_keys = set(d1.keys())
    all_keys = all_keys.union(set(d2.keys()))
    c1 = Counter(normalize(d1))
    c2 = Counter(normalize(d2))
    diff = 0
    for key in all_keys:
        curr_diff = abs(c1[key] - c2[key])
        diff += curr_diff
    return diff/2

def load_synthetic():
    return pd.read_csv(get_synthetic_out_file())

def get_block_df(df):
    new_cols = [col for col in df.columns if col not in NON_BLOCK_COLS]
    df = df[new_cols]
    block_df = df.groupby('identifier').first().reset_index()
    return block_df

def test_representativeness(df):
    _, fallback_dist = read_microdata(get_micro_file())
    hh_dist = Counter()
    hh_df = df[HH_COLS]
    counts = hh_df.groupby(list(hh_df.columns)).size().reset_index(name='counts')
    for ind, row in counts.iterrows():
        hh_counts = tuple(row[HH_COLS].tolist())
        hh_dist[hh_counts] = row[-1]
    return tvd(hh_dist, fallback_dist)

if __name__ == '__main__':
    df = load_synthetic()
    # print(df.head())
    print('Number of households:', len(df))
    print('Total population:', sum(df['TOTAL']))
    block_df = get_block_df(df)
    print('Number of blocks:', len(block_df))
    print('Blocks with accurate age data: %.2f%%' % (sum(block_df['AGE_ACCURACY']) / len(block_df) * 100))
    print('Blocks solved at accuracy = 1: %.2f%%' % (sum(block_df['ACCURACY'] == 1) / len(block_df) * 100))
    print('Blocks solved at accuracy = 2: %.2f%%' % (sum(block_df['ACCURACY'] == 2) / len(block_df) * 100))
    print('Blocks solved at accuracy = 3: %.2f%%' % (sum(block_df['ACCURACY'] == 3) / len(block_df) * 100))
    tot_var_dist = test_representativeness(df)
    print('TVD between PUMS and synthetic HH distributions:', tot_var_dist)
