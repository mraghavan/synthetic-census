import pandas as pd
import pickle
from collections import Counter
from ..utils.census_utils import approx_equal, Race, RACES, hh_to_race_eth_age_tup
from ..utils.knapsack_utils import normalize
from ..preprocessing.build_micro_dist import read_microdata
from ..utils.config2 import ParserBuilder

parser_builder = ParserBuilder({
    'state': True,
    'micro_file': True,
    'synthetic_data': True,
    })

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

TEX_VARS = {}

HH_COLS = NON_BLOCK_COLS_L[1:-1]

NON_BLOCK_COLS = set(NON_BLOCK_COLS_L)

STATE = ''

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

def load_synthetic(name: str):
    return pd.read_csv(name)

def get_block_df(df):
    new_cols = [col for col in df.columns if col not in NON_BLOCK_COLS]
    df = df[new_cols]
    block_df = df.groupby('identifier').first().reset_index()
    return block_df

def process_dist(dist):
    new_dist = Counter()
    for k, v in dist.items():
        new_dist[hh_to_race_eth_age_tup(k)] += v
    return new_dist

def hh_size(tup):
    return sum(tup[:len(Race)])

def get_size_dist(dist):
    size_dist = Counter()
    for hh, prob in dist.items():
        size_dist[hh_size(hh)] += prob
    return size_dist

def get_size_dist_max_7(dist):
    size_dist = Counter()
    for hh, prob in dist.items():
        size_dist[min(hh_size(hh), 7)] += prob
    return size_dist

def size_filter_max_7(dist, size):
    filtered = Counter()
    for hh, prob in dist.items():
        if min(hh_size(hh), 7) == size:
            filtered[hh] += prob
    if len(filtered) == 0:
        return {}
    else:
        return normalize(filtered)

def size_race_adjusted_tvd(d1, d2):
    # Get size distribution from d1
    size_dist = get_size_dist_max_7(d1)
    size_dist2 = get_size_dist_max_7(d2)
    add_tex_var('SynSizeOne', size_dist[1])
    add_tex_var('PUMSSizeOne', size_dist2[1])
    distances = {}
    for s in size_dist:
        d1_filtered = size_filter_max_7(d1, s)
        d2_filtered = size_filter_max_7(d2, s)
        if s == 1:
            d2_filtered = adjust_ethnicity(d1_filtered, d2_filtered)
        if not d1_filtered or not d2_filtered:
            distances[s] = 1
        else:
            distances[s] = tvd(d1_filtered, d2_filtered)
    tot = 0
    for s in size_dist:
        tot += size_dist[s] * distances[s]
    return tot

def adjust_race(d1, d2):
    r1_mass = get_r_mass(d1)
    r2_mass = get_r_mass(d2)
    new_d2 = {}
    for hh, prob in d2.items():
        hh_race = get_race_size_one_hh(hh)
        new_d2[hh] = prob * r1_mass[hh_race] / r2_mass[hh_race]
    assert approx_equal(sum(new_d2.values()), 1)
    return new_d2

def adjust_ethnicity(d1, d2):
    h1_mass = get_h_mass(d1)
    h2_mass = get_h_mass(d2)
    new_d2 = {}
    for hh, prob in d2.items():
        hh_eth = get_eth_size_one_hh(hh)
        new_d2[hh] = prob * h1_mass[hh_eth] / h2_mass[hh_eth]
    assert approx_equal(sum(new_d2.values()), 1)
    return new_d2

def get_r_mass(dist):
    r_mass = Counter()
    for hh, prob in dist.items():
        hh_race = get_race_size_one_hh(hh)
        r_mass[hh_race] += prob
    return r_mass

def get_h_mass(dist):
    h_mass = Counter()
    for hh, prob in dist.items():
        hh_race = get_eth_size_one_hh(hh)
        h_mass[hh_race] += prob
    return h_mass

def get_race_size_one_hh(hh):
    return next(i for i, x in enumerate(hh) if x != 0)

def get_eth_size_one_hh(hh):
    return hh[-2]

def test_representativeness(df, fallback_dist):
    hh_dist = Counter()
    hh_df = df[HH_COLS]
    counts = hh_df.groupby(list(hh_df.columns)).size().reset_index(name='counts')
    for _, row in counts.iterrows():
        hh_counts = tuple(row[HH_COLS].tolist())
        hh_dist[hh_counts] += row[-1]
    hh_dist = normalize(hh_dist)
    test_mixed_race(hh_dist, fallback_dist)
    return tvd(hh_dist, fallback_dist), size_race_adjusted_tvd(hh_dist, fallback_dist)

def test_mixed_race(hh_dist, fallback_dist):
    hh_mr = dist_to_mr(hh_dist)
    fb_mr = dist_to_mr(fallback_dist)
    add_tex_var('SynMRPct', hh_mr[True] * 100)
    add_tex_var('PUMSMRPct', fb_mr[True] * 100)

def dist_to_mr(dist):
    mr_dist = Counter()
    for hh, prob in dist.items():
        if is_mixed_race(hh):
            mr_dist[True] += prob
        else:
            mr_dist[False] += prob
    return mr_dist

def is_mixed_race(hh):
    r_tup = hh[:len(RACES)]
    num_non_zero = sum(1 for x in r_tup if x != 0)
    return num_non_zero > 1

def add_tex_var(name, value, precision=2):
    if name in TEX_VARS:
        return
    TEX_VARS[name] = (value, precision)

def print_all_tex_vars(state):
    for name, (value, precision) in TEX_VARS.items():
        if type(value) == str:
            print('\\newcommand{\\%s%s}{%s}' % (state, name, value))
        elif type(value) == int:
            print('\\newcommand{\\%s%s}{%s}' % (state, name, f"{value:,}"))
        # if value is sufficiently small, use scientific notation
        elif abs(value) < 0.001:
            print(('\\newcommand{\\%s%s}{%.' + str(precision) + 'e}') % (state, name, value))
        else:
            print(('\\newcommand{\\%s%s}{%.' + str(precision) + 'f}') % (state, name, value))

def print_results(
        state: str,
        synthetic_data: str,
        micro_file: str,
        tag: str = '',
        ):
    df = load_synthetic(synthetic_data)
    add_tex_var('TotalHH', len(df))
    add_tex_var('TotalPop', sum(df['TOTAL']))
    block_df = get_block_df(df)
    add_tex_var('NumBlocks', len(block_df))
    add_tex_var('AgeAccurate', sum(block_df['AGE_ACCURACY']) / len(block_df) * 100)
    add_tex_var('AccOne', sum(block_df['ACCURACY'] == 1) / len(block_df) * 100)
    add_tex_var('AccTwo', sum(block_df['ACCURACY'] == 2) / len(block_df) * 100)
    add_tex_var('AccThree', sum(block_df['ACCURACY'] == 3) / len(block_df) * 100)

    fallback_dist = process_dist(read_microdata(micro_file))

    tot_var_dist, size_adjusted = test_representativeness(df, fallback_dist)
    add_tex_var('TVDUnadjustedAll', tot_var_dist, precision=3)
    add_tex_var('TVDAdjustedAll', size_adjusted, precision=3)

    tot_var_dist, size_adjusted = test_representativeness(df[df['ACCURACY'] == 1], fallback_dist)
    add_tex_var('TVDUnadjustedAccOne', tot_var_dist, precision=3)
    add_tex_var('TVDAdjustedAccOne', size_adjusted, precision=3)

    print_all_tex_vars(state + tag)

def get_adjustments(df, fallback_dist, smoothing=0.001, mr_mult=2.0):
    hh_dist = Counter()
    hh_df = df[HH_COLS]
    counts = hh_df.groupby(list(hh_df.columns)).size().reset_index(name='counts')
    for _, row in counts.iterrows():
        hh_counts = tuple(row[HH_COLS].tolist())
        hh_dist[hh_counts] += row[-1]
    hh_dist = normalize(hh_dist)
    all_keys = set(hh_dist.keys())
    all_keys = all_keys.union(set(fallback_dist.keys()))
    c1 = Counter(normalize(hh_dist))
    c2 = Counter(normalize(fallback_dist))
    ratios = {}
    for key in all_keys:
        c1[key] += smoothing
        c2[key] += smoothing
        if c1[key] == 0:
            continue
        ratios[key] = c2[key] / c1[key]
        if is_mixed_race(key):
            ratios[key] *= mr_mult
    return ratios

def write_dist_adjustment(
        state: str,
        synthetic_data: str,
        micro_file: str,
        adjustment_file: str,
        smoothing: float = 0.001,
        ):
    df = load_synthetic(synthetic_data)
    fallback_dist = process_dist(read_microdata(micro_file))
    ratios = get_adjustments(df, fallback_dist, smoothing=smoothing)
    print('Ratio range', min(ratios.values()), max(ratios.values()))
    ratios['smoothing'] = smoothing
    with open(adjustment_file, 'wb') as f:
        pickle.dump(ratios, f)
