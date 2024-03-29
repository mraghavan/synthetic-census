import os
from syn_census.utils.config2 import ParserBuilder
from syn_census.synthetic_data_generation.sample_from_dist import aggregate_shards

parser_builder = ParserBuilder({
    'micro_file': True,
    'block_clean_file': True,
    'synthetic_output_dir': True,
    'task_name': False,
    })

if __name__ == '__main__':
    parser_builder.parse_args()
    print(parser_builder.args)
    args = parser_builder.args
    if args.task_name != '':
        task_name = args.task_name + '_'
    else:
        task_name = ''
    out_df = aggregate_shards(args.micro_file, args.block_clean_file, args.synthetic_output_dir, task_name)
    out_fname = os.path.join(args.synthetic_output_dir, task_name + 'microdata.csv')
    with open(out_fname, 'w') as f:
        print('Writing to', out_fname)
        out_df.to_csv(f, index=False)
