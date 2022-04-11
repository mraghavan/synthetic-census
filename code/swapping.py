from census_utils import *
import pandas as pd
import geopandas as gpd
from random import random
from math import sqrt
from scipy.spatial import KDTree
import numpy as np
from codetiming import Timer
import json

@Timer('Loading data')
def load_data():
    return pd.read_csv(get_synthetic_out_file())

@Timer()
def make_identifier_synth(df):
    ID_COLS = ['TRACTA', 'COUNTYA', 'BLOCKA']
    id_lens = [6, 3, 4]
    str_cols = [col + '_str' for col in ID_COLS]
    for col, l, col_s in zip(ID_COLS, id_lens, str_cols):
        assert max(num_digits(s) for s in df[col].unique()) <= l
        df[col_s] = df[col].astype(str).str.zfill(l)
    df['id'] = df[str_cols].astype(str).agg('-'.join, axis=1)
    for col_s in str_cols:
        del df[col_s]

@Timer()
def load_shape_data(area):
    block_map = gpd.read_file(get_shape_file(area))
    return block_map.to_crs("EPSG:3395")

@Timer()
def make_identifier_synth_geo(df):
    ID_COLS = ['TRACTCE10', 'COUNTYFP10', 'BLOCKCE10']
    id_lens = [6, 3, 4]
    str_cols = [col + '_str' for col in ID_COLS]
    for col, l, col_s in zip(ID_COLS, id_lens, str_cols):
        assert max(num_digits(s) for s in df[col].unique()) <= l
        df[col_s] = df[col].astype(str).str.zfill(l)
    df['id'] = df[str_cols].astype(str).agg('-'.join, axis=1)
    for col_s in str_cols:
        del df[col_s]

@Timer()
def build_trees_and_inds(df):
    trees = {}
    indices = {}
    for t, a in all_num_age_pairs:
        matches = df[(df['TOTAL'] == t) & (df['18_PLUS'] == a)]
        pts = np.array([matches['INTPTLAT10'], matches['INTPTLON10']]).T
        indices[(t, a)] = {i: index for i, (index, row) in enumerate(matches.iterrows())}
        trees[(t, a)] = KDTree(pts)
    return trees, indices


def block_distance(row):
    lat1 = row['INTPTLAT10']
    lat2 = row['other_lat']
    lon1 = row['INTPTLON10']
    lon2 = row['other_lon']
    return sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)

def find_k_closest(row, df, k):
    t = row['TOTAL']
    a = row['18_PLUS']
    tree = trees[(t, a)]
    inds = indices[(t, a)]
    if len(inds) <= k:
        return df[(df['TOTAL'] == t) & (df['18_PLUS'] == a)]
    lat = row['INTPTLAT10']
    lon = row['INTPTLON10']
    num_to_query = k+1
    while num_to_query <= len(inds):
        dists, candidates = tree.query((lat, lon), num_to_query)
#         print(candidates)
        first_non_zero = 0
        while first_non_zero < len(dists) and dists[first_non_zero] == 0:
            first_non_zero += 1
        dists = dists[first_non_zero:]
        candidates = candidates[first_non_zero:]
        if len(dists) < k:
            num_to_query *= 2
            continue
#         print(dists)
#         print(candidates)
        cand_inds = [inds[c] for c in candidates]
#         print(cand_inds)
        cand_rows = df.loc[cand_inds].copy()
        cand_rows['distance'] = dists
        cand_rows = cand_rows[cand_rows['swapped'] == 0]
        if len(cand_rows) < k:
            num_to_query *= 2
            continue
        return cand_rows.head(k)
    cand_rows = df[(df['TOTAL'] == t) & (df['18_PLUS'] == a) & (df['swapped'] == 0) & (df['blockid'] != row['blockid'])].head(k).copy()
    cand_rows['distance'] = cand_rows.apply(block_distance, axis=1)
    return cand_rows

def flag_risk(df):
    dist_u = params['risk_dist']
    dist_n = {k: v*num_rows for k, v in dist_u.items()}
    flagging = ['W', 'B', 'AI_AN', 'AS', 'H_PI', 'OTH', 'TWO_OR_MORE', 'NUM_HISP']
    counts = df[flagging].groupby(flagging).size().reset_index()
    merged = df.merge(counts,
             how='left',
             on=flagging,
             validate='many_to_one',
    ).rename({0: 'frequency'}, axis=1)
    merged.sort_values(by=['BLOCK_TOTAL', 'frequency'], axis=0, inplace=True)
    vec_n = [[i] * int(dist_n[i]) for i in (4, 3, 2, 1)]
    l = []
    for v in vec_n:
        l += v
    if len(l) < num_rows:
        l += [1] * (num_rows - len(l))
    merged['U'] = l
    merged['prob'] = merged['U'].replace(params['swap_probs'])
    return merged

def get_swap_partners(df):
    hh_1s = []
    hh_2s = []
    dists = []
    df['swapped'] = 0
    num_matches = params['num_matches']
    s = params['swap_rate']
    print('Total number of swaps', int(s*num_rows)//2)
    print('Beginning swapping...')
    with Timer():
        for i, row in df.iterrows():
            j = df['swapped'].sum()
            if j % 5000 == 0:
                print(j, '/', int(s*num_rows))
            if j >= num_rows*s:
                break
            if df.loc[i, 'swapped'] == 1:
                continue
            do_swap = random() < row['prob']
            if not do_swap:
                continue
            matches = find_k_closest(row, df, num_matches)
            m = matches.sample()
            partner_index = m.index[0]
            m = m.reset_index().iloc[0]
            hh_1s.extend([row['household.id'], m['household.id']])
            hh_2s.extend([m['household.id'], row['household.id']])
            dists.extend([m['distance'], m['distance']])
            if i == partner_index:
                print(i, partner_index)
            if df.loc[i, 'swapped'] == 1:
                print(i)
                print(df.loc[i])
                print(row)
            if df.loc[partner_index, 'swapped'] == 1:
                print(i, partner_index)
            assert i != partner_index
            assert df.loc[i, 'swapped'] == 0
            assert df.loc[partner_index, 'swapped'] == 0
            df.loc[[i, partner_index], 'swapped'] = 1
    partners = pd.DataFrame({'hh_1': hh_1s, 'hh_2': hh_2s, 'distance': dists})
    return partners

def finish_swap(df, pairs):
    swapped_df = df.merge(
        pairs,
        left_on = 'household.id',
        right_on = 'hh_1',
        how = 'left',
        validate = 'one_to_one',
    )
    swapped_df.drop(columns=['hh_1', 'INTPTLAT10', 'INTPTLON10', 'COUNTY', 'NAME', 'COUSUBA', 'BLKGRPA', 'ACCURACY', 'AGE_ACCURACY'], inplace=True)
    swapped_df.head()

    swap_subset = swapped_df['swapped'] == 1
    expanded = swapped_df.loc[swap_subset, 'hh_2'].str.split('-', expand=True)
    swapped_df.loc[swap_subset, 'COUNTYA'] = pd.to_numeric(expanded[1])
    swapped_df.loc[swap_subset, 'TRACTA'] = pd.to_numeric(expanded[0])
    swapped_df.loc[swap_subset, 'BLOCKA'] = pd.to_numeric(expanded[2])
    swapped_df.loc[swap_subset, 'household.id'] = swapped_df.loc[swap_subset, 'hh_2']
    swapped_df.rename({'id': 'blockid'}, inplace=True, axis=1)
    del swapped_df['hh_2']
    return swapped_df

if __name__ == '__main__':
    print('Loading data...')
    df = load_data()
    num_rows = len(df)
    print(num_rows, 'rows')
    params = None
    with open('swapping_params.json') as f:
        params = json.load(f)
    params['risk_dist'] = {int(k): v for k, v in params['risk_dist'].items()}
    params['swap_probs'] = {int(k): v for k, v in params['swap_probs'].items()}
    merged = flag_risk(df)
    del merged['identifier']
    print('Adding identifier...')
    make_identifier_synth(merged)

    merged['hh_str'] = merged['HH_NUM'].astype(str).str.zfill(4)
    merged['household.id'] = merged[['id', 'hh_str']].astype(str).agg('-'.join, axis=1)
    del merged['hh_str']

    print('Loading shape data...')
    block_geo = load_shape_data('BLOCK')

    print('Adding geo identifier...')
    make_identifier_synth_geo(block_geo)

    merged = merged.merge(
        block_geo[['INTPTLAT10', 'INTPTLON10', 'id']],
        on='id',
        how='left',
        validate='many_to_one',
    )

    merged['INTPTLAT10'] = pd.to_numeric(merged['INTPTLAT10'])
    merged['INTPTLON10'] = pd.to_numeric(merged['INTPTLON10'])

    all_num_age_pairs = set(zip(merged['TOTAL'], merged['18_PLUS']))

    print('Building trees...')
    trees, indices = build_trees_and_inds(merged)
    partners = get_swap_partners(merged)

    just_pairs = partners[['hh_1', 'hh_2']]
    print(len(just_pairs), 'pairs')
    print(merged['swapped'].sum(), 'total swapped')
    assert len(just_pairs) == merged['swapped'].sum()

    swapped_df = finish_swap(merged, just_pairs)
    if WRITE:
        with open(get_swapped_file(), 'w') as f:
            swapped_df.to_csv(f)
