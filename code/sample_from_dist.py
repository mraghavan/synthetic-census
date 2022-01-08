from census_utils import *
from build_block_df import *
import os
import pickle
import numpy as np

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
        '18_PLUS',
        'HH_NUM',
        'ACCURACY',
        'AGE_ACCURACY',
        ]

CARRYOVER = set(OUTPUT_COLS) - POP_COLS


def load_sample_and_accs():
    d = get_dist_dir()
    sample = {}
    accs = {}
    errors = []
    for fname in os.listdir(d):
        if fname.endswith('pkl'):
            with open(d + fname, 'rb') as f:
                results = pickle.load(f)
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
                sample[int(results['id'])] = breakdown
                accs[int(results['id'])] = (results['level'], results['age'])
    return sample, accs, errors

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
        block_df = pd.DataFrame((r_list + [sum(b[:-1])] + list(b) + [i] + list(accs[row['identifier']]) for i, b in enumerate(breakdown)), columns=out_df.columns)
        out_df = pd.concat([out_df, block_df], ignore_index=True)
    out_df.rename(columns=SHORT_RN, inplace=True)
    make_identifier_non_unique(out_df)
    print(out_df.head())
    with open(get_synthetic_out_file(), 'w') as f:
        out_df.to_csv(f, index=False)
