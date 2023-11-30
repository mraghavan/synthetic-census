import sys
sys.path.append('../')
from syn_census.utils.config2 import ParserBuilder
from syn_census.synthetic_analysis.representativeness import print_results

parser_builder = ParserBuilder({
    'state': True,
    'micro_file': True,
    'synthetic_data': True,
    })

if __name__ == '__main__':
    parser_builder.parse_args()
    print(parser_builder.args, file=sys.stderr)
    args = parser_builder.args
    print_results(args.state, args.synthetic_data, args.micro_file)
