import sys
sys.path.append('../')
from syn_census.utils.config2 import ParserBuilder
from syn_census.synthetic_analysis.sampling_evaluation import print_results

parser_builder = ParserBuilder(
        {'state': True,
         'synthetic_output_dir': False,
         'num_sols': True,
         'task_name': False,
         })

if __name__ == '__main__':
    parser_builder.parse_args()
    print(parser_builder.args, file=sys.stderr)
    args = parser_builder.args
    print_results(args.state, args.synthetic_output_dir, args.num_sols, args.task_name)
