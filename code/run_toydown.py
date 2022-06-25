# Based on code from mggg/census-diff-privacy
import numpy as np
import pandas as pd
from toydown import GeoUnit, ToyDown
from dask.distributed import Client
import argparse
import multiprocessing
from census_utils import *
from sample_from_dist import DEMO_COLS
from build_block_df import USEFUL_COLS
from hh_to_person_microdata import make_td_identifier
from partition_blocks import read_block_data

ARRAY_ORDER = ['TOTAL', 'NUM_HISP', 'W', 'B', 'AI_AN', 'AS', 'H_PI', 'OTH', 'TWO_OR_MORE']

def load_data(task_name):
    return pd.read_csv(get_person_micro_file(task_name))

def make_non_hispanic(df):
    df.loc[df['NUM_HISP'] > 0, ['W', 'B', 'AI_AN', 'AS', 'H_PI', 'OTH', 'TWO_OR_MORE']] = 0

def make_arrays(df):
    tot_df = df[['TOTAL'] + DEMO_COLS + ['td_identifier']].groupby('td_identifier').sum().reset_index()
    vap_df = df[df['18_PLUS'] > 0][['TOTAL'] + DEMO_COLS + ['td_identifier']].groupby('td_identifier').sum().reset_index()
    print(tot_df.head())
    print(vap_df.head())
    leaves = {}
    for (i, tot_row), (j, vap_row) in zip(tot_df.iterrows(), vap_df.iterrows()):
        assert tot_row['td_identifier'] == vap_row['td_identifier']
        tot_array = tot_row[ARRAY_ORDER].values
        vap_array = vap_row[ARRAY_ORDER].values
        leaves[str(tot_row['td_identifier'])] = {'TOTPOP': tot_array, 'VAP': vap_array}
    return leaves

def create_tree_from_leaves(leaf_dict):
    """ Given a dictionary, where the keys are the names of leaf nodes (labeled by their path)
        and the corresponding value is the associated attribute counts, this function returns
        the list of GeoUnits that defines the corresponding tree.
    """
    nodes = leaf_dict.copy()
    h = len(list(leaf_dict.keys())[0])
    counts = ["TOTPOP", "VAP"]
    n = len(list(leaf_dict.values())[0][counts[0]])
    level_offsets = [3, 4, 10]
    
    for offset in level_offsets:
        level_names = list(set(list(map(lambda s: s[:-offset], leaf_dict.keys()))))
        for node in level_names:
            nodes[node] = {c: np.array([v[c] for k, v in leaf_dict.items() if k.startswith(node)]).sum(axis=0) for c in counts}
    return [GeoUnit(k, k[:-3], v) if len(k) == 15 else GeoUnit(k, k[:-1], v) if len(k) == 12 else GeoUnit(k, k[:-6], v) if len(k) == 11 else GeoUnit(k, k[:-3], v) for k, v in nodes.items()]

def build_df_from_dict(d, block_df):
    tot_rows = []
    vap_rows = []
    cols_with_age = ARRAY_ORDER + ['18_PLUS'] + ['td_identifier']
    cols = ARRAY_ORDER + ['td_identifier']
    for k, v in d.items():
        if len(k) < 15:
            continue
        tot_rows.append(v['TOTPOP'].tolist() + [v['VAP'][0], k])
        vap_rows.append(v['VAP'].tolist() + [k])
    tot_df = pd.DataFrame(tot_rows, columns=cols_with_age)
    vap_df = pd.DataFrame(vap_rows, columns=cols)
    block_df['td_identifier'] = block_df['td_identifier'].astype(str)
    tot_df = join_with_block_info(tot_df, block_df)
    vap_df = join_with_block_info(vap_df, block_df)
    return tot_df, vap_df

def join_with_block_info(df, block_df):
    return df.merge(block_df[list(USEFUL_COLS) + ['identifier', 'td_identifier']],
            on='td_identifier',
            how='inner',
            validate='one_to_one',
            )

if __name__ == '__main__':
    multiprocessing.freeze_support()
    ## Set up args
    parser = argparse.ArgumentParser(description="ToyDownMultiAttribute noise", 
                                     prog="run_toydown.py")
    parser.add_argument('name', nargs='?', default='')
    parser.add_argument("w", metavar="num_workers", type=int,
                        help="How many cores to use")
    parser.add_argument("n", metavar="num_runs", type=int,
                        help="How many runs to perform")
    parser.add_argument("eps", metavar="epsilon", type=float,
                        help="total epsilon budget")
    parser.add_argument("eps_split", metavar="epsilon_split", type=str,
                        choices=["equal", "top_heavy", "mid_heavy", "bottom_heavy"],
                        help="how to budget epsilon across levels")
    args = parser.parse_args()


    ## Set up data
    print("Loading data...")
    if args.name:
        task_name = args.name + '_'
    else:
        task_name = ''
    df = load_data(task_name)
    block_df = read_block_data()
    print("Making identifier...")
    make_td_identifier(block_df)

    make_non_hispanic(df)

    leaves = make_arrays(df)

    geounits = create_tree_from_leaves(leaves)
    geounits.reverse()
    state_data = {
            'TOTPOP': df[ARRAY_ORDER].sum(axis=0).values,
            'VAP': df[df['18_PLUS'] > 0][ARRAY_ORDER].sum(axis=0).values
            }
    print(state_data)
    state_geo = GeoUnit(str(df['STATEA'][0]), None, state_data)

    # tx = GeoUnit("48", None, tx_data)
    # county_geounits = [GeoUnit(geoid, "48", attr) for geoid, attr in counties.items()]
    geounits.insert(0, state_geo)

    split_dict = {"equal": [1/5, 1/5, 1/5, 1/5, 1/5], "top_heavy": [1/2, 1/4, 1/12, 1/12, 1/12], 
                  "mid_heavy": [1/12, 1/6, 1/2, 1/6, 1/12], "bottom_heavy": [1/12, 1/12, 1/12, 1/4, 1/2]}

    eps = args.eps
    eps_split_name = args.eps_split
    eps_split = split_dict[eps_split_name]
    n_samps = args.n
    height = len(eps_split)
    n_workers = args.w

    def run_model(i):
        return model_all.noise_and_adjust(verbose=False, bounds='non-negative')

    client = Client(processes=True, threads_per_worker=1, n_workers=n_workers)
    print(client, flush=True)

    model_all = ToyDown(geounits, height, eps, eps_split, gurobi=True)

    print("Starting {} runs with eps {} and {} split".format(n_samps*n_workers, eps, eps_split_name), flush=True)
    client.scatter(model_all, broadcast=True)

    for j in range(n_samps):
        adjusteds = client.map(run_model, range(n_workers))
        d = client.gather(adjusteds)[0]
        tot_df, vap_df = build_df_from_dict(d, block_df)
        # to_sav = np.array((client.gather(adjusteds)))
        if WRITE:
            tot_df.to_csv(get_dp_tot_file(task_name), index=False)
            vap_df.to_csv(get_dp_vap_file(task_name), index=False)
        del adjusteds
        
    del model_all
