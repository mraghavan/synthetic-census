import pandas as pd
from ..utils.census_utils import num_digits

USEFUL_COLS = {
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
        }

ID_COLS = [
        'COUNTYA',
        'TRACTA',
        'BLOCKA',
        ]

DATA_PREFIXES = {
        'H7',
        'H8',
        'H9',
        'IA',
        }

def block_col_select(col):
    return col in USEFUL_COLS or col[:2] in DATA_PREFIXES

def make_identifier(df):
    make_identifier_non_unique(df)
    assert len(df) == len(df.identifier.unique())

def make_identifier_non_unique(df):
    id_lens = [3, 6, 4]
    str_cols = [col + '_str' for col in ID_COLS]
    for col, l, col_s in zip(ID_COLS, id_lens, str_cols):
        assert max(num_digits(s) for s in df[col].unique()) <= l
        df[col_s] = df[col].astype(str).str.zfill(l)
    df['identifier'] = df[str_cols].astype(str).agg('-'.join, axis=1)
    for col_s in str_cols:
        del df[col_s]

def get_clean_block_df(fname: str):
    df = pd.read_csv(fname)
    df = df[[c for c in df if block_col_select(c)]]
    df = df[df['H7X001'] > 0]
    make_identifier(df)
    print('Block')
    print(df.head())
    print(list(df.columns))
    print(len(df), 'non-empty blocks')
    return df
