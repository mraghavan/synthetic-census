import pandas as pd
from census_utils import *
from functools import reduce
import operator

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
    df['identifier'] = reduce(operator.add, [df[col].astype(str) for col in ID_COLS])
    # Make sure identifiers are unique, otherwise need to add more columns to ID_COLS
    assert len(df) == len(df.identifier.unique())

def get_clean_block_df():
    df = pd.read_csv(get_block_file())
    df = df[[c for c in df if block_col_select(c)]]
    df = df[df['H7X001'] > 0]
    make_identifier(df)
    print('Block')
    print(df.head())
    print(list(df.columns))
    print(len(df), 'non-empty blocks')
    return df


if __name__ == '__main__':
    df = get_clean_block_df()
    with open(get_block_out_file(), 'w') as f:
        df.to_csv(f, index=False)
