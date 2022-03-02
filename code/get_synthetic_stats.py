from census_utils import *
from knapsack_utils import *
import pandas as pd
from build_micro_dist import read_microdata
from collections import Counter
import sys

NON_BLOCK_COLS_L = [
        'TOTAL',
        'W',
        'B',
        'AI_AN',
        'AS',
        'H_PI',
        'OTH',
        'TWO_OR_MORE',
        'NUM_HISP',
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

def load_synthetic(name=''):
    return pd.read_csv(get_synthetic_out_file(name))

def get_block_df(df):
    new_cols = [col for col in df.columns if col not in NON_BLOCK_COLS]
    df = df[new_cols]
    block_df = df.groupby('identifier').first().reset_index()
    return block_df

def process_dist(dist):
    new_dist = Counter()
    for k, v in dist.items():
        new_dist[k.race_counts + (k.eth_count,) + (k.n_over_18,)] += v
    return new_dist

def hh_size(tup):
    return sum(tup[:len(Race)])

def get_size_dist(dist):
    size_dist = Counter()
    for hh, prob in dist.items():
        size_dist[hh_size(hh)] += prob
    return size_dist

def size_filter(dist, size):
    filtered = Counter()
    for hh, prob in dist.items():
        if hh_size(hh) == size:
            filtered[hh] += prob
    if len(filtered) == 0:
        return {}
    else:
        return normalize(filtered)

def size_adjusted_tvd(d1, d2):
    # Get size distribution from d1
    size_dist = get_size_dist(d1)
    distances = {}
    for s in size_dist:
        d1_filtered = size_filter(d1, s)
        d2_filtered = size_filter(d2, s)
        if not d1_filtered or not d2_filtered:
            distances[s] = 1
        else:
            distances[s] = tvd(d1_filtered, d2_filtered)
    tot = 0
    for s in size_dist:
        tot += size_dist[s] * distances[s]
    return tot

def test_representativeness(df):
    fallback_dist = process_dist(read_microdata(get_micro_file()))
    hh_dist = Counter()
    hh_df = df[HH_COLS]
    counts = hh_df.groupby(list(hh_df.columns)).size().reset_index(name='counts')
    for ind, row in counts.iterrows():
        hh_counts = tuple(row[HH_COLS].tolist())
        hh_dist[hh_counts] += row[-1]
    hh_dist = normalize(hh_dist)
    print(Counter(hh_dist).most_common(10))
    print(Counter(fallback_dist).most_common(10))
    print(get_size_dist(fallback_dist))
    return tvd(hh_dist, fallback_dist), size_adjusted_tvd(fallback_dist, hh_dist)

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        task_name = sys.argv[1] + '_'
    else:
        task_name = ''
    df = load_synthetic(task_name)
    # print(df.head())
    print('Number of households:', len(df))
    print('Total population:', sum(df['TOTAL']))
    block_df = get_block_df(df)
    print('Number of blocks:', len(block_df))
    print('Blocks with accurate age data: %.2f%%' % (sum(block_df['AGE_ACCURACY']) / len(block_df) * 100))
    print('Blocks solved at accuracy = 1: %.2f%%' % (sum(block_df['ACCURACY'] == 1) / len(block_df) * 100))
    print('Blocks solved at accuracy = 2: %.2f%%' % (sum(block_df['ACCURACY'] == 2) / len(block_df) * 100))
    print('Blocks solved at accuracy = 3: %.2f%%' % (sum(block_df['ACCURACY'] == 3) / len(block_df) * 100))
    tot_var_dist, size_adjusted = test_representativeness(df)
    print('TVD between PUMS and synthetic HH distributions:', tot_var_dist)
    print('Size-adjusted TVD between PUMS and synthetic HH distributions:', size_adjusted)

    tot_var_dist, size_adjusted = test_representativeness(df[df['ACCURACY'] == 1])
    print('TVD between PUMS and synthetic HH distributions for ACCURACY == 1:', tot_var_dist)
    print('Size-adjusted TVD between PUMS and synthetic HH distributions for ACCURACY == 1:', size_adjusted)
