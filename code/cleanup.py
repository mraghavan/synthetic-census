from census_utils import *
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("No job specified")
        sys.exit(1)
    job_id = sys.argv[1] + '_'
    d = get_dist_dir()
    for fname in os.listdir(d):
        if re.match(job_id + '[0-9]+_[0-9]+.pkl', fname):
            os.remove(d + fname)
    out = 'out_files/'
    if os.path.exists(out):
        for fname in os.listdir(out):
            if re.match('census.' + job_id + r'[0-9]+.out', fname):
                # print(fname)
                os.remove(out + fname)
            elif re.match('census.' + job_id + r'[0-9]+.err', fname):
                # print(fname)
                os.remove(out + fname)
