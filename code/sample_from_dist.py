from census_utils import *
from build_block_df import *
from build_micro_dist import read_microdata
import os
import pickle
import numpy as np
from collections import Counter
from knapsack_utils import normalize

POP_COLS = {
        'H7X002',
        'H7X003',
        'H7X004',
        'H7X005',
        'H7X006',
        'H7X007',
        'H7X008',
        }
SHORT_RN = {
        'H7X001': 'TOTAL',
        'H7X002': 'W',
        'H7X003': 'B',
        'H7X004': 'AI_AN',
        'H7X005': 'AS',
        'H7X006': 'H_PI',
        'H7X007': 'OTH',
        'H7X008': 'TWO_OR_MORE',
        'H8A003': 'BLOCK_18_PLUS',
        }

OUTPUT_COLS = [
        'YEAR',
        'STATE',
        'STATEA',
        'COUNTY',
        'COUNTYA',
        'COUSUBA',
        'TRACTA',
        'BLKGRPA',
        'BLOCKA',
        'NAME',
        'BLOCK_TOTAL',
        'H8A003',
        'H7X001',
        'H7X002',
        'H7X003',
        'H7X004',
        'H7X005',
        'H7X006',
        'H7X007',
        'H7X008',
        'NUM_HISP',
        '18_PLUS',
        'HH_NUM',
        'ACCURACY',
        'AGE_ACCURACY',
        ]

RECORD_LENGTH = len(Race) + 2

CARRYOVER = set(OUTPUT_COLS) - POP_COLS

def process_dist(dist):
    new_dist = Counter()
    for tup, prob in dist.items():
        new_dist[tup.race_counts + (tup.eth_count, tup.n_over_18)] += prob
    return normalize(new_dist)

def load_sample_and_accs():
    dist = read_microdata(get_micro_file())
    dist = process_dist(dist)
    d = get_dist_dir()
    sample = {}
    accs = {}
    errors = []
    for fname in os.listdir(d):
        if re.match('[0-9]+_[0-9]+.pkl', fname):
            print('Reading from', d+fname)
            with open(d + fname, 'rb') as f:
                result_list = pickle.load(f)
                for results in result_list:
                    keys, probs = zip(*results['sol'].items())
                    keys = list(keys)
                    if len(keys) > 1:
                        try:
                            breakdown = keys[np.random.choice(range(len(keys)), p=probs)]
                        except:
                            print('Error', results['id'])
                            errors.append(int(results['id']))
                            continue
                    else:
                        breakdown = keys[0]
                    # Add age if missing
                    if len(breakdown[0]) == RECORD_LENGTH - 1:
                        breakdown = add_age(breakdown, dist)
                    sample[int(results['id'])] = breakdown
                    accs[int(results['id'])] = (results['level'], results['age'])
    return sample, accs, errors

def add_age(hh_list, dist):
    out_list = []
    for hh in hh_list:
        eligible = [full_hh for full_hh in dist if hh == full_hh[:RECORD_LENGTH-1]]
        probs = np.array([dist[full_hh] for full_hh in eligible])
        probs = probs / np.sum(probs)
        out_list.append(eligible[np.random.choice(range(len(eligible)), p=probs)])
    return tuple(sorted(out_list))

if __name__ == '__main__':
    df = pd.read_csv(get_block_out_file())
    print(df.head())
    out_df = pd.DataFrame(columns=OUTPUT_COLS)
    print(out_df.head())
    sample, accs, errors = load_sample_and_accs()
    for ind, row in df.iterrows():
        if row['identifier'] in errors:
            print('Error index', ind)
            continue
        try:
            breakdown = sample[row['identifier']]
        except:
            continue
        r_list = [row[x] for x in df.columns if x in CARRYOVER]
        try:
            block_df = pd.DataFrame((r_list + [sum(b[:-2])] + list(b) + [i] + list(accs[row['identifier']]) for i, b in enumerate(breakdown)), columns=out_df.columns)
        except Exception as e:
            print('Error:', ind)
            print(breakdown)
            print(e)
            continue
        out_df = pd.concat([out_df, block_df], ignore_index=True)
    out_df.rename(columns=SHORT_RN, inplace=True)
    make_identifier_non_unique(out_df)
    print(out_df.head())
    with open(get_synthetic_out_file(), 'w') as f:
        out_df.to_csv(f, index=False)
