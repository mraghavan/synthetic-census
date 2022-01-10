from census_utils import *
import pandas as pd

NON_BLOCK_COLS = {
        'TOTAL',
        'W',
        'B',
        'AI_AN',
        'AS',
        'H_PI',
        'OTH',
        'TWO_OR_MORE',
        '18_PLUS',
        'HH_NUM',
        }

def load_synthetic():
    return pd.read_csv(get_synthetic_out_file())

def get_block_df(df):
    new_cols = [col for col in df.columns if col not in NON_BLOCK_COLS]
    df = df[new_cols]
    block_df = df.groupby('identifier').first().reset_index()
    return block_df

if __name__ == '__main__':
    df = load_synthetic()
    # print(df.head())
    print('Number of households:', len(df))
    print('Total population:', sum(df['TOTAL']))
    block_df = get_block_df(df)
    print('Number of blocks:', len(block_df))
    print('Blocks with accurate age data: %.2f%%' % (sum(block_df['AGE_ACCURACY']) / len(block_df) * 100))
    print('Blocks solved at accuracy = 1: %.2f%%' % (sum(block_df['ACCURACY'] == 1) / len(block_df) * 100))
    print('Blocks solved at accuracy = 2: %.2f%%' % (sum(block_df['ACCURACY'] == 2) / len(block_df) * 100))
    print('Blocks solved at accuracy = 3: %.2f%%' % (sum(block_df['ACCURACY'] == 3) / len(block_df) * 100))
