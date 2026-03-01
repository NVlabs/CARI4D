# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys, os
sys.path.append(os.getcwd())
from prep.fp_behave import merge_pickles
from glob import glob
import time 
from prep.fp_filter_2dir import FPFilterTwoDirProcessor
import json
import os.path as osp


class BehaveHy3D2DirFPRunner(FPFilterTwoDirProcessor):
    def get_template_file(self):
        "load from hy3d"
        obj_name = self.video_prefix.split('_')[2]
        files = sorted(glob(f'{self.args.hy3d_root}/{self.video_prefix}*/*{self.video_prefix}*_align.obj'))
        if len(files) == 0:
            raise ValueError(f'no aligned hy3d template found for {self.video_prefix}')
        mesh_file = files[0]
        print('using object template from file:', mesh_file)
        return mesh_file

def _child_run_kid(kid, args):
    """Run one view in an isolated child process.
    Creates CUDA/context-heavy objects inside the child to avoid pickling issues.
    """
    import torch

    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass

    processor_child = BehaveHy3D2DirFPRunner(args) # this holds some thread lock 
    processor_child.process_video(kid)
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

def process_video(args):
    # use multiple processes to process different views of this video 
    import multiprocessing as mp

    mp.set_start_method('spawn', force=True)
    ctx = mp.get_context('spawn')

    procs = []  # run 4 processes in parallel, around 7s/frame

    # run only on the selected views 
    selected_views = json.load(open('splits/selected-views-map.json'))
    video_prefix = osp.basename(args.video).split('.')[0]
    kids = [int(selected_views[video_prefix][1])] if video_prefix in selected_views else args.cameras
    args.cameras = kids 
    print(f"running kids {kids} for seq {video_prefix}")
    for k in kids:
        p = ctx.Process(target=_child_run_kid, args=(k, args))
        p.start()
        time.sleep(5) # to avoid race condition
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == '__main__':
    parser = FPFilterTwoDirProcessor.get_parser()
    args = parser.parse_args()
    import traceback

    try:
        if osp.isfile(args.video):
            videos = [args.video]
        else:
            videos = sorted(glob(args.video))
        print(f"In total {len(videos)} video files")
        selected_views = json.load(open('splits/selected-views-map.json'))
        video_prefix = osp.basename(args.video).split('.')[0]
        if args.index is not None:
            chunk_size = 1  # for easy paralle 
            videos = videos[args.index * chunk_size:(args.index + 1) * chunk_size]
        print(f"Processing {len(videos)} video files, first video: {videos[0]}, last video: {videos[-1]}")
        for video in videos:
            args.video = video
            processor = BehaveHy3D2DirFPRunner(args)
            kid_to_run = int(selected_views[video_prefix][1]) if video_prefix in selected_views else args.kid
            processor.process_video(kid_to_run)

        # now collect results from different cameras into one file
        merge_pickles(videos, args)
    except Exception as e:
        print(args.video, 'failed')
        traceback.print_exc()
    


        
