from ..utils.census_utils import has_valid_age_data
from ..utils.config2 import ParserBuilder
import pandas as pd

parser_builder = ParserBuilder({
    'block_clean_file': True,
    })

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
    parser_builder.parse_args()
    print(parser_builder.args)
    df = pd.read_csv(parser_builder.args.block_clean_file)
    errs = 0
    for ind, row in df.iterrows():
        errs += validate_row(ind, row)
    print('Total errors:', errs, errs/len(df))
