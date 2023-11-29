from .preprocessing.build_block_df import get_clean_block_df
from .utils.config2 import ParserBuilder

parser_builder = ParserBuilder({
    'block_file': True,
    'block_clean_file': False,
    })

if __name__ == '__main__':
    parser_builder.parse_args()
    print(parser_builder.args)
    df = get_clean_block_df(parser_builder.args.block_file)
    if parser_builder.args.block_clean_file:
        print('Writing to', parser_builder.args.block_clean_file)
        with open(parser_builder.args.block_clean_file, 'w') as f:
            df.to_csv(f, index=False)
