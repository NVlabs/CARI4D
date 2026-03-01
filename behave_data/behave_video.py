# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
a base class to handle behave video data
"""
import json
import joblib
import sys, os
import os.path as osp
from typing import Any
import numpy as np
import imageio
from .utils import availabe_kindata
from .video_reader import VideoController, ColorDepthController


def load_masks(video_prefix, frame_time: str, k, h5_file, data_source='behave') -> tuple[int | Any, int | Any]:
    """
    load human, object mask from the packed h5 file
    the original BEHAVE dataset does not provide mask for every single frame, hence I pack the mask into h5 files for easy access.

    :param video_prefix: basename of the video
    :param frame_time: the frame time of the mask to be read, used to identify the mask key string
    :param k: camera/kinect id, used to identify the mask key string
    :param h5_file: h5.File object, stores masks as key-value pairs
    :param data_source: name of the dataset used
    :return: human, object mask of shape (H, W), np.uint8 format, or None if such mask is not found in the h5 file
    """
    mname_h = f'{video_prefix}/{frame_time}-k{k}.person_mask.png'
    mname_o = f'{video_prefix}/{frame_time}-k{k}.obj_rend_mask.png'
    try:
        mask_h = h5_file[mname_h][:]  
        mask_o = h5_file[mname_o][:]  
        mask_h = mask_h.astype(np.uint8) * 255
        mask_o = mask_o.astype(np.uint8) * 255
        return mask_h, mask_o
    except Exception as e:
        print(e)
        return None, None

class BaseBehaveVideoData(object):
    def __init__(self, args=None):
        # automatic mask path selection
        self.args = args
        self.data_source = args.data_source
        self.prepare_video_loader(args)
        input_color = args.video
        video_prefix = osp.basename(input_color).split('.')[0]
        self.video_prefix = video_prefix
        seq_date = video_prefix.split('_')[0].lower()
        mask_files = ['masks-date01-02.tar', 'masks-date03.tar', 'masks-date04-06.tar', 'masks-date07.tar']
        if seq_date in ['date01', 'date02']:
            tar_path = mask_files[0]
        elif seq_date in ['date03']:
            tar_path = mask_files[1]
        elif seq_date in ['date04', 'date05', 'date06']:
            tar_path = mask_files[2]
        else:
            tar_path = mask_files[3]
        self.tar_path = osp.join(osp.dirname(input_color), tar_path)
        if 'dpro' in input_color:
            self.tar_path = osp.join(osp.dirname(input_color.replace('/dpro/', '/videos/')), tar_path)
        self.video = args.video
        self.wild_video = args.wild_video

        # faster access using h5 file
        self.tar_path = f'{args.masks_root}/{video_prefix}_masks_k0.h5'

        self.camera_K = self.init_camera_K()
        self.scale_ratio = 2 if self.camera_K[0, 2] > 1000 else 1  # scale down ratio for 2k image
        print(self.camera_K, 'scale ratio', self.scale_ratio)
        self.camera_K[:2] /= self.scale_ratio
        self.init_others()
        self.model = None

    def get_time_str(self, t):
        "get the time string used for unique identification"
        if self.data_source == 'behave' and not self.wild_video:
            frame_time = 't{:08.3f}'.format(t) 
        else:
            frame_time = f'{t:06d}'
        return frame_time

    def time_str_to_float(self, time_str):
        "convert the time string to a float"
        if self.data_source == 'behave' and not self.wild_video:
            t = float(time_str[1:])
        else:
            t = float(time_str)
        return t
        

    def get_chunk_num(self):
        return 500

    def prepare_video_loader(self, args):
        input_color = args.video
        video_prefix = osp.basename(input_color).split('.')[0]
        output_h5_path = osp.join(args.outpath, video_prefix + f'_all.pkl')
        os.makedirs(osp.dirname(output_h5_path), exist_ok=True)
        self.output_path = output_h5_path

        kids, comb = availabe_kindata(input_color, kinect_count=4 if args.data_source != 'intercap' else 6)
        print("Available kinects for sequence {}: {}".format(osp.basename(input_color), kids))
        self.kids = kids
        kinect_count = len(kids)
        # load videos
        video_folder = osp.dirname(input_color)
        if args.nodepth:
            controllers = [VideoController(os.path.join(video_folder, f'{video_prefix}.{k}.color.mp4')) for k in kids]
        else:
            controllers = [ColorDepthController(os.path.join(video_folder, video_prefix), k) for k in kids]
        end_time = np.min([controllers[x].end_time() for x in range(kinect_count)]) if args.tend is None else args.tend
        start_time = args.tstart
        fps = args.fps
        if args.wild_video:
            self.times = np.arange(0, len(controllers[0]))
        else:
            if args.data_source == 'behave':
                self.times = np.arange(start_time, end_time - 1. / fps, 1. / fps).tolist()
            else:
                self.times = np.arange(0, len(controllers[0]))
        self.video_prefix = video_prefix
        self.controllers = controllers
        # cut into chunks
        if len(self.times) > self.get_chunk_num():
            # do chunks
            end = len(self.times) if args.end == -1 else args.end
            output_h5_path = osp.join(args.outpath, self.video_prefix + f'_{args.start:06d}-{end:06d}.pkl')
            if osp.isfile(output_h5_path) and not args.redo:
                print("Already exists {}, all done".format(output_h5_path))
                exit(0)
            self.output_path = output_h5_path
            self.times = self.times[args.start:end]

    def init_others(self):
        pass

    def init_camera_K(self):
        if not self.args.wild_video:
            if self.args.data_source == 'behave':
                if self.video is not None:
                    assert 'Date' in osp.basename(self.video), f'invalid video path {osp.basename(self.video)}'
                fx, fy = 979.784, 979.840  # for original kinect coordinate system
                cx, cy = 1018.952, 779.486
                self.image_size = (1536, 2048)
            elif self.args.data_source == 'hodome':
                # from camera 1 intrinsics 
                assert '2022' in osp.basename(self.video), f'invalid video path {osp.basename(self.video)}'
                K = np.array([[1.02583559e+03, 0.00000000e+00, 6.36561061e+02],
                                [0.00000000e+00, 1.02583559e+03, 3.57313754e+02],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
                fx, fy = K[0,0], K[1,1]
                cx, cy = K[0,2], K[1,2]
                self.image_size = (720, 1280)
            elif self.args.data_source == 'intercap':
                if self.video is not None:
                    assert 'ICapS' in osp.basename(self.video), f'invalid video path {osp.basename(self.video)}'
                from .const import ICAP_CENTERs
                from .const import ICAP_FOCALs
                kid = 0 # use default camera 0 intrinsics
                fx, fy = ICAP_FOCALs[kid, 0], ICAP_FOCALs[kid, 1]
                cx, cy = ICAP_CENTERs[kid, 0], ICAP_CENTERs[kid, 1]
                self.image_size = (1080, 1920)
                
            elif self.args.data_source == 'imhd':
                assert '2023' in osp.basename(self.video), f'invalid video path {osp.basename(self.video)}'
                from .const import IMHD_VIEW_IDS
                from .const import get_IMHD_camera_K
                intrinsics = get_IMHD_camera_K(osp.basename(self.video).split(".")[0], IMHD_VIEW_IDS[0])
                fx, fy = intrinsics[0,0], intrinsics[1,1]
                cx, cy = intrinsics[0,2], intrinsics[1,2]
                self.image_size = (1080, 1920)
            elif self.args.data_source == 'procigen':
                assert 'Subxx' in osp.basename(self.video), f'invalid video path {osp.basename(self.video)}'
                K = np.array([[979.784, 0, 1018.952],
                          [0, 979.840, 779.486],
                          [0, 0, 1]])
                K[:2] /= 2. 
                intrinsics = K 
                fx, fy = K[0,0], K[1,1]
                cx, cy = K[0,2], K[1,2]
                self.image_size = (768, 1024)
            else:
                raise ValueError(f'Invalid data source: {self.args.data_source}')
            camera_K = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]]).astype(np.float32)
        else:
            pkl_file = self.args.video.replace('.mp4', '.pkl')
            d = joblib.load(pkl_file)
            fx, fy = d['fx'], d['fy']
            cx, cy = d['cx'], d['cy']
            camera_K = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]]).astype(np.float32)
            # get the image size info
            img = imageio.get_reader(self.args.video).get_data(0)
            self.image_size = img.shape[:2]
        return camera_K

    def load_color_depth(self, enum_idx, kids, t):
        if self.args.wild_video:
            actual_time = t
        else:
            actual_times = np.array([self.controllers[x].get_closest_time(t) for x, _ in enumerate(kids)])
            best_kid = np.argmin(np.abs(actual_times - t))
            actual_time = actual_times[best_kid]
        if self.args.nodepth:
            return self.controllers[enum_idx].get_closest_frame(actual_time), None
        else:
            color, depth = self.controllers[enum_idx].get_closest_frame(actual_time)
            return color, depth

    @staticmethod
    def get_parser():
        from argparse import ArgumentParser

        parser = ArgumentParser()

        # common setup for all video input processes
        parser.add_argument('-v', '--video', help='path to a video file')
        parser.add_argument('-o', '--outpath', default='/home/xianghuix/datasets/behave/fp')
        parser.add_argument('--masks_root', default='/home/xianghuix/datasets/behave/masks-h5-my')
        parser.add_argument('-fps', type=int, default=30, help='generate frames at which fps')
        parser.add_argument('-tstart', type=float, default=3.0, help='first frame time')
        parser.add_argument('-tend', type=float, default=None, help='last frame time')
        parser.add_argument('--redo', default=False, action='store_true')
        parser.add_argument('-k', '--kid', default=1, type=int)
        parser.add_argument('-fs', '--start', type=int, default=0, help='start frame index')
        parser.add_argument('--end', type=int, default=-1, help='end frame index')
        parser.add_argument('--nodepth', default=False, action='store_true')
        parser.add_argument('--cameras', default=[0, 1, 2, 3], nargs='+', type=int)
        parser.add_argument('--wild_video', default=False, action='store_true')
        parser.add_argument('--data_source', default='behave', choices=['behave', 'hodome', 'intercap', 'imhd', 'procigen'])

        # for debug
        parser.add_argument('--viz_path', default=None)
        parser.add_argument('--debug', default=0, type=int)

        # for parallel processing, the index
        parser.add_argument('--index', default=None, type=int)
        parser.add_argument('--chunk_start', default=None, type=int)
        parser.add_argument('--chunk_end', default=None, type=int)

        # additional data path
        parser.add_argument('--nlf_path', default='/home/xianghuix/datasets/behave/nlf-smplh-genmo-incam-z/', type=str)

        # additional path for hy3d
        parser.add_argument('--hy3d_root', default='/home/xianghuix/datasets/behave/selected-views/hy3d-aligned', type=str)


        return parser
