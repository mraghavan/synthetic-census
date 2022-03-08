from census_utils import *
import pandas as pd
import numpy as np

def aggregate_to_block(df):
    demo_cols = ['TOTAL', 'W', 'B', 'AI_AN', 'AS', 'H_PI', 'OTH', 'TWO_OR_MORE', 'NUM_HISP', '18_PLUS']
    agg_dict = {col : np.sum for col in demo_cols}
    agg_dict['AGE_ACCURACY'] = np.mean
    agg_dict['ACCURACY'] = np.mean
    agg_df = df.groupby('identifier').agg(agg_dict)
    return agg_df

def combine_dfs(block_df, agg_df):
    df = block_df.merge(agg_df,
            how='inner',
            on='identifier',
            validate='one_to_one')
    return df

def simplify_df(df):
    cols = df.columns
    to_drop = [c for c in cols if c.startswith('IA')]
    to_drop += [c for c in cols if c.startswith('H9')]
    to_drop += [c for c in cols if c.startswith('H8M')]
    df.drop(columns=to_drop, inplace=True)

def check_age(df):
    df_age = df[df['AGE_ACCURACY'] == 1]
    df_no_age = df[df['AGE_ACCURACY'] != 1]
    print(df_age['H8A003'].sum(), df_age['18_PLUS'].sum())
    print(df_no_age['H8A003'].sum(), df_no_age['18_PLUS'].sum())
    print('Accurate age adult fraction. Original:', df_age['H8A003'].sum()/df_age['H7X001'].sum(), 'Synthetic:', df_age['18_PLUS'].sum()/df_age['TOTAL'].sum())
    print('Bad age adult fraction. Original:', df_no_age['H8A003'].sum()/df_no_age['H7X001'].sum(), 'Synthetic:', df_no_age['18_PLUS'].sum()/df_no_age['TOTAL'].sum())

if __name__ == '__main__':
    # load original
    block_df = pd.read_csv(get_block_out_file())
    # load synthetic
    synth_df = pd.read_csv(get_synthetic_out_file())
    print(synth_df.head())
    agg_df = aggregate_to_block(synth_df)
    print(agg_df.head())
    # combine
    df = combine_dfs(block_df, agg_df)
    print(df.head())
    # compare
    compare_counts(df)
