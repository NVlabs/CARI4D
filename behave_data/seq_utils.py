# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
utils to get sequence information
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
import os, sys
sys.path.append(os.getcwd())
import json
from os.path import join, basename, dirname


class SeqInfo:
    "a simple class to handle information of a sequence"
    def __init__(self, seq_path):
        self.info = self.get_seq_info_data(seq_path)

    def get_obj_name(self, convert=False):
        if convert: # for using detectron
            if 'chair' in self.info['cat']:
                return 'chair'
            if 'ball' in self.info['cat']:
                return 'sports ball'
        return self.info['cat']

    def get_gender(self):
        return self.info['gender']

    def get_config(self):
        return self.info['config']

    def get_intrinsic(self):
        return self.info['intrinsic']

    def get_empty_dir(self):
        return self.info['empty']

    def beta_init(self):
        return self.info['beta']

    def kinect_count(self):
        if 'kinects' in self.info:
            return len(self.info['kinects'])
        else:
            return 3

    @property
    def kids(self):
        count = self.kinect_count()
        return [i for i in range(count)]

    def get_seq_info_data(self, seq):
        info_file = join(seq, 'info.json')
        data = json.load(open(info_file))
        # all paths are relative to the sequence path
        path_names = ['config', 'empty', 'intrinsic']
        for name in path_names:
            if data[name] is not None:
                data[name] = join(seq, data[name])
        return data


def save_seq_info(seq_folder, config, intrinsic, cat,
                  gender, empty, beta,
                  kids=[0, 1, 2, 3]):
    outfile = join(seq_folder, 'info.json')
    info = {
        "config":config,
        "intrinsic":intrinsic,
        'cat':cat,
        'gender':gender,
        'empty':empty,
        'kinects':kids,
        'beta':beta
    }
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("{} saved.".format(outfile))
    print("{}: {}, {}, {}, {}, {}".format(seq_folder, config, intrinsic, cat, beta, gender))




if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('seq_folder')
    parser.add_argument('config')
    parser.add_argument('cat')
    parser.add_argument('face_dir')
    parser.add_argument('gender')
    parser.add_argument('beta')
    parser.add_argument('--empty', default=None)
    parser.add_argument('-c', '--color', default=True, help='generated pc in color coordinate or not',)
    parser.add_argument('--intrinsic')
    parser.add_argument('-k', '--kids', default=[0, 1, 2], nargs='+', type=int)

    args = parser.parse_args()
    save_seq_info(args.seq_folder, args.config, args.intrinsic, args.cat, args.face_dir,
                  args.gender, args.empty, args.color, args.beta, args.kids)

