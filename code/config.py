import os
import json

params = None
with open('params.json') as f:
    params = json.load(f)
STATE = params['state']
DATA_DIR = os.path.expandvars(params['data']) + STATE + '/'
MICRO_FILE = DATA_DIR + STATE.lower() + '.2010.pums.01.txt'
BLOCK_FILE = DATA_DIR + 'block_data.csv'
BLOCK_OUTPUT_FILE = DATA_DIR + 'block_data_cleaned.csv'
OUTPUT_DIR = os.path.expandvars(params['output']) + STATE + '/'
SWAPPED_FILE = OUTPUT_DIR + 'swapped.csv'
SHAPE_FILE = DATA_DIR + STATE + '_block_2010.shp'
GROUP_SHAPE_FILE = DATA_DIR + STATE + '_blck_grp_2010.shp'
NUM_SOLS = params['num_sols']
WRITE = params['write'] == 1

flag = True

def print_config():
    print(params)

def check_file(fname):
    global flag
    if not os.path.exists(fname):
        flag = False
        print('Error: file not found (%s)' % fname)
    else:
        print('File found')

if __name__ == '__main__':
    print('Data located in', DATA_DIR)
    print('Checking microdata file (%s)' % MICRO_FILE)
    check_file(MICRO_FILE)
    print('Checking block data file (%s)' % BLOCK_FILE)
    check_file(BLOCK_FILE)
    print('Checking output directory (%s)' % OUTPUT_DIR)
    check_file(OUTPUT_DIR)
    if flag:
        print('All files found')
    else:
        print('Some files were missing')
