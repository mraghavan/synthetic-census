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

# May need these later
# US_DIR = os.path.expandvars(params['data']) + 'US/shapefiles/'
# UP_LEG_SHAPE_FILE = US_DIR + 'US_stleg_up_2010.shp'
# LOW_LEG_SHAPE_FILE = US_DIR + 'US_stleg_lo_2010.shp'
# COUNTY_SHAPE_FILE = US_DIR + 'US_county_2010.shp'
# TRACT_SHAPE_FILE = US_DIR + 'US_tract_2010.shp'
# CONG_SHAPE_FILE = US_DIR + 'US_cd111th_2010.shp'
