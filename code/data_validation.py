from census_utils import *
import pandas as pd

def validate_row(ind, row):
    err = 0
    P3_tot = row['H7X001']
    P16_tot = row['H8A001']
    # if P3_tot != P16_tot:
        # err = 1
    if not has_valid_age_data(row):
        print(ind, 'invalid age data')
        print(ind, ': P3 and P16 disagree by ', P3_tot - P16_tot)
        err = 1
    P28_tot = row['H8M001']
    if P28_tot == 0:
        print(ind, 'P28 has no households')
        err = 1
    return err

if __name__ == '__main__':
    df = pd.read_csv(get_block_out_file())
    errs = 0
    for ind, row in df.iterrows():
        errs += validate_row(ind, row)
    print('Total errors:', errs, errs/len(df))
