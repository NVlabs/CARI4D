"""
original kinect frame reader that only loads multi-view color and depth images

author: Xianghui Xie
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""

import numpy as np
import os, argparse
from os.path import join, isfile, basename, isdir
from PIL import Image
import cv2


class KinectFrameReader:
    def __init__(self, seq, empty=None, kinect_count=4, ext='jpg', check_image=False):
        # prepare depth and color paths
        if seq.endswith('/'):
            seq = seq[:-1]
        self.seq_path = seq
        self.ext = ext  # file extension for color image
        self.kinect_count = kinect_count
        self.frames = self.prepare_paths(check_image=check_image)
        self.empty = empty
        self.bkgs = self.prepare_bkgs()
        self.seq_name = basename(seq)
        self.kids = [i for i in range(kinect_count)]

    def prepare_paths(self, check_image=True):
        "find out which frames contain complete color and depth images"
        frames = sorted(os.listdir(self.seq_path))
        valid_frames = []
        for frame in frames:
            frame_folder = join(self.seq_path, frame)
            if check_image:
                if self.check_frames(frame_folder, check_image):
                    valid_frames.append(frame)
            else:
                if isdir(frame_folder):
                    valid_frames.append(frame)

        return valid_frames

    def check_frames(self, frame_folder, check_depth=False):
        for k in range(self.kinect_count):
            color_file = join(frame_folder, 'k{}.color.{}'.format(k, self.ext))
            depth_file = join(frame_folder, 'k{}.depth.png'.format(k))
            if not isfile(color_file) or (check_depth and not isfile(depth_file)):
                return False
        return True

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx):
        " return the i-th frame of color and depth images"
        frame_folder = join(self.seq_path, self.frames[idx])
        color_files = [join(frame_folder, f'k{k}.color.{self.ext}') for k in range(self.kinect_count)]
        depth_files = [join(frame_folder, f'k{k}.depth.png') for k in range(self.kinect_count)]

        colors = [Image.open(c).convert('RGB') for c in color_files]
        colors = [np.array(c) for c in colors]
        depths = [cv2.imread(d, cv2.IMREAD_ANYDEPTH) for d in depth_files]

        if self.bkgs is not None:
            depths_filtered = []
            for d, bkg in zip(depths, self.bkgs):
                df = remove_background(d, bkg, tol=30)
                depths_filtered.append(df)
            depths = depths_filtered

        return colors, depths

    def get_color_images(self, idx, kids, bgr=False):
        color_files = self.get_color_files(idx, kids)
        if bgr:
            colors = [cv2.imread(x) for x in color_files]
        else:
            colors = [Image.open(c).convert('RGB') for c in color_files]
            colors = [np.array(c) for c in colors]
        return colors

    def get_color_files(self, idx, kids):
        frame_folder = self.get_frame_folder(idx)
        color_files = [join(frame_folder, f'k{k}.color.{self.ext}') for k in kids]
        return color_files

    def get_depth_images(self, idx, kids):
        frame_folder = join(self.seq_path, self.frames[idx])
        depth_files = [join(frame_folder, f'k{k}.depth.png') for k in kids]
        depths = [cv2.imread(d, cv2.IMREAD_ANYDEPTH) for d in depth_files]
        return depths

    def get_frame_folder(self, idx):
        if isinstance(idx, int):
            assert idx < len(self)
            return join(self.seq_path, self.frames[idx])
        elif isinstance(idx, str):
            return join(self.seq_path, idx)
        else:
            raise NotImplemented

    def prepare_bkgs(self):
        if self.empty is None:
            return None
        else:
            bkgs = [get_seq_bkg(self.empty, x) for x in range(self.kinect_count)]
            return bkgs

    def remove_background(self, depth, bkg, tol=100):
        diff = np.abs(depth - bkg)
        mask = ~(diff >= tol)
        depth[mask] = 0
        return depth

    def frame_time(self, idx):
        return self.frames[idx]

    def get_timestamps(self):
        "timestamp list for all frames"
        times = [float(x[1:]) for x in self.frames]
        return times

    def get_frame_idx(self, timestr):
        try:
            idx = self.frames.index(timestr)
            return idx
        except ValueError:
            return -1


def get_seq_bkg(seq, kid, start=0):
    "get the average depth for this sequence of one kinect"
    frames = sorted(os.listdir(seq))
    depths = []
    for frame in frames[start:]:
        depth_file = join(seq, frame, f'k{kid}.depth.png')
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        if depth is not None:
            depths.append(depth)
    avg = np.stack(depths, axis=-1).mean(axis=-1)
    return avg


def remove_background(depth, bkg, tol=100):
    diff = np.abs(depth - bkg)
    mask = ~(diff >= tol)
    depth[mask] = 0
    return depth








