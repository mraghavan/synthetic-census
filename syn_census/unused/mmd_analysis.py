import pandas as pd
import geopandas as gpd
import numpy as np
from census_utils import *
from config import *
import matplotlib.pyplot as plt

DEMO_COLS = ['TOTAL', 'W', 'B', 'AI_AN', 'AS', 'H_PI', 'OTH', 'TWO_OR_MORE', 'NUM_HISP', 'X18_PLUS']
DEMO_COLS_ORIG = ['H7X001', 'H7X002', 'H7X003', 'H7X004', 'H7X005', 'H7X006', 'H7X007', 'H7X008', 'H7Z010', 'H8A003']
ID_COLS = ['TRACTA', 'COUNTYA', 'BLOCKA']

def load_full_data():
    df = pd.read_csv(get_block_file())
    # return df[df['H7X001'] > 0]
    return df

def load_shape_data():
    block_map = gpd.read_file(get_shape_file())
    block_grp_map = gpd.read_file(get_grp_shape_file())
    return block_map.to_crs("EPSG:3395"), block_grp_map.to_crs("EPSG:3395")

def load_swapped_data():
    df = pd.read_csv(get_swapped_file())
    df.rename(columns={'county': 'COUNTYA', 'tract': 'TRACTA'}, inplace=True)
    return df

def make_identifier_swapped(df : pd.DataFrame):
    id_lens = [6, 3, 4]
    str_cols = [col + '_str' for col in ID_COLS]
    for col, l, col_s in zip(ID_COLS, id_lens, str_cols):
        assert max(num_digits(s) for s in df[col].unique()) <= l
        df[col_s] = df[col].astype(str).str.zfill(l)
    df['blockid'] = df[str_cols].astype(str).agg('-'.join, axis=1)
    for col_s in str_cols:
        del df[col_s]

def aggregate_df(df : pd.DataFrame):
    print('Aggregating')
    agg_dict = {col : np.sum for col in DEMO_COLS}
    agg_dict.update({'swapped': np.sum})
    new_df = df.copy()
    # make_identifier_swapped(new_df)
    # household.id has the old location; blockid has the new location
    # new_df['blockid'] = df['household.id'].str.replace('-[0-9]+$', '')
    agg_df = new_df.groupby('blockid').agg(agg_dict)
    return agg_df

def look_for_r_eth_diffs(df : pd.DataFrame):
    # diffs = df[DEMO_COLS] != df[DEMO_COLS_ORIG]
    comps = [df[c] != df[co] for c, co in zip(DEMO_COLS, DEMO_COLS_ORIG)]
    diffs = comps[0]
    for comp in comps[1:]:
        diffs = diffs | comp
    print(sum(diffs), 'blocks with different counts')

def pct_white_change(row):
    before_pct = row['H7X002'] / row['TOTAL']
    after_pct = row['W'] / row['TOTAL']
    return after_pct - before_pct

def mmd_change(row):
    before_pct = row['H7X002'] / row['TOTAL']
    after_pct = row['W'] / row['TOTAL']
    if before_pct > .5 and after_pct < .5:
        return -1
    elif before_pct < .5 and after_pct > .5:
        return 1
    return 0

def district_level(df : pd.DataFrame, col):
    agg_dict = {col : np.sum for col in DEMO_COLS + DEMO_COLS_ORIG}
    agg_dict.update({'swapped': np.sum})
    agg_df = df.groupby(col).agg(agg_dict)
    print(len(agg_df), col, 'districts')
    pct_change = agg_df.apply(pct_white_change, axis=1)
    print(pct_change)
    plt.hist(pct_change)
    plt.title(col)
    plt.show()
    mmd = agg_df.apply(mmd_change, axis=1)
    print(mmd)
    plt.hist(mmd)
    plt.title(col)
    plt.show()

def make_before_and_after_df(combined_df, block_map):
    df = block_map.merge(combined_df,
            how='inner',
            on='GISJOIN',
            validate='one_to_one')
    before_cols = {col: new_col + '_BEFORE' for (col, new_col) in zip(DEMO_COLS_ORIG, DEMO_COLS)}
    after_cols = {col: col + '_AFTER' for col in DEMO_COLS}
    df.rename(columns=before_cols, inplace=True)
    df.rename(columns=after_cols, inplace=True)
    return df

def plot_race(df, block_grp_map):
    # df.plot.scatter(x='pct_white', y='swapped')
    # plt.show()
    # df['pct_white'] = df['W_BEFORE']/df['TOTAL_BEFORE']
    # df['log_total'] = np.log(df['TOTAL_AFTER'])
    fig, ax = plt.subplots(figsize=(6, 6))
    df.plot(ax=ax,
            column='swapped',
            cmap='Reds',
            legend=True,
            missing_kwds={"color": "grey"})
    block_grp_map.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=.2)
    # fig.patch.set_visible(False)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    full_df = load_full_data()
    make_identifier_swapped(full_df)
    block_map, block_grp_map = load_shape_data()
    print(full_df.head())
    print(len(full_df))
    swapped_df = load_swapped_data()
    print(swapped_df.head())
    print(swapped_df.tail())
    print('Total number of households:', len(swapped_df))
    print('Total number swapped:', swapped_df['swapped'].sum())
    agg_df = aggregate_df(swapped_df)
    combined_df = full_df.join(agg_df, on='blockid')
    # print('Combined')
    # print(combined_df.head())
    before_after_df = make_before_and_after_df(combined_df, block_map)
    print(before_after_df.head())
    look_for_r_eth_diffs(combined_df)
    plot_race(before_after_df, block_grp_map)
    # district_level(combined_df, 'SLDUA')
    # district_level(combined_df, 'SLDLA')
