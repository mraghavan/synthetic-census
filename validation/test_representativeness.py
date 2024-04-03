import sys
import os
sys.path.append('../')
from syn_census.utils.config2 import ParserBuilder
from syn_census.synthetic_analysis.representativeness import print_results, write_dist_adjustment

parser_builder = ParserBuilder({
    'state': True,
    'micro_file': True,
    'task_name': True,
    'synthetic_output_dir': True,
    'dist_adjustment': True,
    })

if __name__ == '__main__':
    parser_builder.parse_args()
    print(parser_builder.args, file=sys.stderr)
    args = parser_builder.args
    task_name = args.task_name
    if task_name != '':
        task_name += '_'
    synthetic_file = os.path.join(args.synthetic_output_dir, task_name + 'microdata.csv')
    print_results(args.state, synthetic_file, args.micro_file)
    dist_adjustment_file = os.path.join(args.synthetic_output_dir, task_name + args.dist_adjustment)
    write_dist_adjustment(args.state, synthetic_file, args.micro_file, dist_adjustment_file)
