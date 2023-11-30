from syn_census.utils.config2 import ParserBuilder
from syn_census.synthetic_data_generation.partition_blocks import generate_data
import sys
import os
import pickle

parser_builder = ParserBuilder({
    'state': True,
    'micro_file': True,
    'block_clean_file': True,
    'synthetic_output_dir': True,
    'num_sols': False,
    'task': False,
    'num_tasks': False,
    'task_name': False,
    })

    # out_file = synthetic_output_dir + task_name + '%d_%d.pkl' % (task, num_tasks)

if __name__ == '__main__':
    parser_builder.parse_args()
    print(parser_builder.args)
    args = parser_builder.args
    task = args.task
    num_tasks = args.num_tasks
    if args.task_name != '':
        task_name = args.task_name + '_'
    else:
        task_name = ''
    out_file = args.synthetic_output_dir + task_name + '%d_%d.pkl' % (task, num_tasks)
    if os.path.exists(out_file):
        print(out_file, 'already exists')
        sys.exit(0)

    output, errors = generate_data(args.state, args.micro_file, args.block_clean_file, args.num_sols, task, num_tasks)

    print('errors', errors, file=sys.stderr)
    print('Writing to', out_file)
    with open(out_file, 'wb') as f:
        pickle.dump(output, f)
    print(len(errors), 'errors')
