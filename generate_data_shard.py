import re
import os
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
    'include_probs': False,
    })

def get_tmp_file(fname_re: str, directory: str):
    pat = re.compile(fname_re)
    all_tmps = {}
    for f in os.listdir(directory):
        m = pat.match(f)
        if m:
            all_tmps[int(m.group(1))] = f
    if all_tmps == {}:
        return ''
    return os.path.join(directory, all_tmps[max(all_tmps.keys())])

def remove_all_tmps(fname_re: str, directory: str):
    pat = re.compile(fname_re)
    for f in os.listdir(directory):
        if pat.match(f):
            print('Removing tmp file', f)
            os.remove(os.path.join(directory, f))

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
    tmp_file = args.synthetic_output_dir + task_name + '%d_%d_tmp.pkl' % (task, num_tasks)
    if os.path.exists(out_file):
        print(out_file, 'already exists')
        sys.exit(0)

    output, errors = generate_data(
            args.micro_file,
            args.block_clean_file,
            args.num_sols,
            task,
            num_tasks,
            include_probs=args.include_probs,
            tmp_file=tmp_file,
            )

    print('errors', errors, file=sys.stderr)
    print('Writing to', out_file)
    with open(out_file, 'wb') as f:
        pickle.dump(output, f)
    print(len(errors), 'errors')
    os.remove(tmp_file)
