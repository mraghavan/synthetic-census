import os
import json
import argparse

class ParserBuilder():
    parser_options = {
        'state': ('', str),
        'micro_file': ('', str),
        'block_file': ('', str),
        'block_clean_file': ('', str),
        'synthetic_output_dir': ('', str),
        'synthetic_data': ('', str),
        'num_sols': (100, int),
        'task': (1, int),
        'num_tasks': (1, int),
        'task_name': ('', str),
        }
    file_paths = {
            'micro_file',
            'block_file',
            'block_clean_file',
            'synthetic_output_dir',
            'synthetic_data',
            }
    def __init__(self, requirements):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--from_params', default='', type=str)
        for req in requirements:
            if req not in self.parser_options:
                raise ValueError('Invalid requirement %s' % req)
            self.parser.add_argument('--' + req, default=self.parser_options[req][0], type=self.parser_options[req][1])
        self.required_args = requirements.copy()
        self.verify_required_args()

    def parse_args(self):
        self.args = self.parser.parse_args()
        if self.args.from_params != '':
            with open(self.args.from_params, 'rb') as f:
                params = json.load(f)
                for req in self.required_args:
                    if req in params:
                        # Only override if it's a default value
                        if self.args.__dict__[req] == self.parser_options[req][0]:
                            self.args.__dict__[req] = params[req]
        for f in self.file_paths:
            if f in self.required_args:
                self.args.__dict__[f] = os.path.expandvars(self.args.__dict__[f])

    def verify_required_args(self):
        for req in self.required_args:
            if self.required_args[req]:
                if self.args.__dict__[req] == self.parser_options[req][0]:
                    raise ValueError('Must specify %s' % req)
                # make sure it's a valid file path
                elif req in self.file_paths:
                    if not os.path.exists(self.args.__dict__[req]):
                        raise ValueError('%s does not exist' % self.args.__dict__[req])

class Config():
    def __init__(self):
        self.state = None
        self.in_dir = None
        self.out_dir = None
        self.num_sols = None

    def set_state(self, state):
        self.state = state

    def set_in_dir(self, in_dir):
        self.in_dir = in_dir

    def set_out_dir(self, out_dir):
        self.out_dir = out_dir

    def set_num_sols(self, num_sols):
        self.num_sols = num_sols

    @property
    def data_dir(self):
        return os.path.expandvars(self.in_dir) + self.state + '/'

    @property
    def micro_file(self):
        return self.data_dir + self.state.lower() + '.2010.pums.01.txt'

    @property
    def block_file(self):
        return self.data_dir + 'block_data.csv'

    @property
    def block_output_file(self):
        return self.data_dir + 'block_data_cleaned.csv'

    @property
    def output_dir(self):
        if self.write:
            return os.path.expandvars(self.out_dir) + self.state + '/'
        return ''

    @property
    def swapped_file(self):
        if self.write:
            return self.out_dir + 'swapped.csv'
        return ''

    @property
    def shape_file(self):
        return self.data_dir + 'shapefiles/' + self.state + '_block_2010.shp'

    @property
    def group_shape_file(self):
        return self.data_dir + 'shapefiles/' + self.state + '_blck_grp_2010.shp'

    @property
    def us_dir(self):
        return os.path.expandvars(self.in_dir) + 'US/shapefiles/'

    @property
    def up_leg_shape_file(self):
        return self.us_dir + 'US_stleg_up_2010.shp'

    @property
    def low_leg_shape_file(self):
        return self.us_dir + 'US_stleg_low_2010.shp'

    @property
    def county_shape_file(self):
        return self.us_dir + 'US_county_2010.shp'

    @property
    def tract_shape_file(self):
        return self.us_dir + 'US_tract_2010.shp'

    @property
    def cong_shape_file(self):
        return self.us_dir + 'US_cong_2010.shp'

    @property
    def write(self):
        return self.out_dir != ''

    def print_config(self):
        print('State:', self.state)
        print('Input files in', self.data_dir)
        if self.write:
            print('Writing to output directory', self.output_dir)
        else:
            print('Not writing to output directory')

# params = None
# with open('params.json') as f:
    # params = json.load(f)
# STATE = params['state']
# DATA_DIR = os.path.expandvars(params['data']) + STATE + '/'
# MICRO_FILE = DATA_DIR + STATE.lower() + '.2010.pums.01.txt'
# BLOCK_FILE = DATA_DIR + 'block_data.csv'
# BLOCK_OUTPUT_FILE = DATA_DIR + 'block_data_cleaned.csv'
# OUTPUT_DIR = os.path.expandvars(params['output']) + STATE + '/'
# SWAPPED_FILE = OUTPUT_DIR + 'swapped.csv'
# SHAPE_FILE = DATA_DIR + 'shapefiles/' + STATE + '_block_2010.shp'
# GROUP_SHAPE_FILE = DATA_DIR + 'shapefiles/' + STATE + '_blck_grp_2010.shp'

# US_DIR = os.path.expandvars(params['data']) + 'US/shapefiles/'
# UP_LEG_SHAPE_FILE = US_DIR + 'US_stleg_up_2010.shp'
# LOW_LEG_SHAPE_FILE = US_DIR + 'US_stleg_lo_2010.shp'
# COUNTY_SHAPE_FILE = US_DIR + 'US_county_2010.shp'
# TRACT_SHAPE_FILE = US_DIR + 'US_tract_2010.shp'
# CONG_SHAPE_FILE = US_DIR + 'US_cd111th_2010.shp'

flag = True

# def print_config():
    # print(params)

def check_file(fname):
    global flag
    if not os.path.exists(fname):
        flag = False
        print('Error: file not found (%s)' % fname)
    else:
        print('File found')

# if __name__ == '__main__':
    # print('Data located in', DATA_DIR)
    # print('Checking microdata file (%s)' % MICRO_FILE)
    # check_file(MICRO_FILE)
    # print('Checking block data file (%s)' % BLOCK_FILE)
    # check_file(BLOCK_FILE)
    # print('Checking output directory (%s)' % OUTPUT_DIR)
    # check_file(OUTPUT_DIR)
    # if flag:
        # print('All files found')
    # else:
        # print('Some files were missing')
