from census_utils import *
from build_micro_dist import read_microdata_granular
import pandas as pd
import numpy as np
import sys
from sample_from_dist import DEMO_COLS, RACE_MAP
import random

def load_data(task_name):
    return pd.read_csv(get_synthetic_out_file(task_name))

def make_td_identifier(df):
    ID_COLS = [
            'STATEA',
            'COUNTYA',
            'TRACTA',
            'BLOCKA',
            ]
    id_lens = [2, 3, 6, 4]
    str_cols = [col + '_str' for col in ID_COLS]
    for col, l, col_s in zip(ID_COLS, id_lens, str_cols):
        assert max(num_digits(s) for s in df[col].unique()) <= l
        df[col_s] = df[col].astype(str).str.zfill(l)
    df['td_identifier'] = df[str_cols].astype(str).agg(''.join, axis=1)
    for col_s in str_cols:
        del df[col_s]

def sample_people(options):
    if len(options) == 1:
        return list(options.keys())[0]
    return random.choices(list(options.keys()), weights=list(options.values()))[0]

def extract_hh_tuple(row):
    return tuple(row[DEMO_COLS])

def get_people_from_row(row : pd.Series, people):
    l = []
    for p in people:
        pers = row.to_numpy(copy=True)
        p_race, p_eth, p_age = p
        for r in Race:
            pers[get_people_from_row.INDS['TOTAL']] = 1
            pers[get_people_from_row.INDS[RACE_MAP[r]]] = 0
            pers[get_people_from_row.INDS[RACE_MAP[p_race]]] = 1
            pers[get_people_from_row.INDS['NUM_HISP']] = p_eth
            if p_age:
                pers[get_people_from_row.INDS['18_PLUS']] = 1
            else:
                pers[get_people_from_row.INDS['18_PLUS']] = 0
        l.append(pers)
    return l

def extract_tuple(key):
    r = None
    for a, b in zip(key[:len(Race)], Race):
        if a == 1:
            r = b
    if key[len(Race)] == 1:
        eth = 1
    else:
        eth = 0
    if key[len(Race)+1] == 1:
        age = 1
    else:
        age = 0
    return (r, eth, age)

def make_list_with_k_1s(k, total):
    l = [1] * k + [0] * (total-k)
    return l

def sample_people_fallback(key):
    if sum(key[:len(Race)]) == 1:
        return [extract_tuple(key)]
    races = []
    for k, r in zip(key[:len(Race)], Race):
        races += [r] * k
    eths = make_list_with_k_1s(key[len(Race)], len(races))
    ages = make_list_with_k_1s(key[len(Race)+1], len(races))
    random.shuffle(eths)
    random.shuffle(ages)
    return list(zip(races, eths, ages))

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        task_name = sys.argv[1] + '_'
    else:
        task_name = ''
    print('Loading data...')
    df = load_data(task_name)
    num_rows = len(df)
    print(num_rows, 'rows')
    print('Reading microdata...')
    microdata = read_microdata_granular(get_micro_file())

    new_rows = []
    print('Adding identifier...')
    make_td_identifier(df)
    print(df.head())
    print(df.columns)
    print(DEMO_COLS)
    get_people_from_row.INDS = {col: df.columns.get_loc(col) for col in ['TOTAL'] + DEMO_COLS}
    for i, row in df.iterrows():
        if i % 10000 == 0:
            print('%d / %d' % (i, len(df)))
        if row['TOTAL'] == 1:
            new_rows.append(row.to_numpy())
            continue
        key = extract_hh_tuple(row)
        if key in microdata:
            people = sample_people(microdata[key])
        else:
            people = sample_people_fallback(key)
        new_rows += get_people_from_row(row, people)
    people_df = pd.DataFrame(new_rows, columns=df.columns)
    for col in DEMO_COLS:
        people_df[col] = people_df[col].astype(int)
    print(people_df.head())
    print(len(people_df), 'rows')
    for col in ['TOTAL'] + DEMO_COLS:
        assert df[col].sum() == people_df[col].sum()
    if WRITE:
        with open(get_person_micro_file(task_name), 'w') as f:
            people_df.to_csv(f, index=False)
