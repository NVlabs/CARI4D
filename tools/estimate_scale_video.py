
"""
the image and masks are loaded from the video data format
"""

import sys, os
sys.path.append(os.getcwd())
import torch, os, json 
import trimesh
import numpy as np
from tqdm import tqdm
import cv2 
import os.path as osp
from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV
import h5py

from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
import nvdiffrast.torch as dr
from glob import glob
import Utils

from tools.estimate_scale import estimate_metric_scale
from behave_data.behave_video import BaseBehaveVideoData

def get_specific_frame(video_prefix, frame_time, kid=1):
    from behave_data.video_reader import ColorDepthController
    ctrl = ColorDepthController(video_prefix, kid)
    color, depth = ctrl.get_closest_frame(float(frame_time[1:]))
    return color, depth

class MetricScaleEstimator(BaseBehaveVideoData):
    def estimate_scale(self, args):
        "estimate the metric scale for the video, and save to a json file"
        video_prefix = osp.basename(args.video).split('.')[0]
        out_file = osp.join(args.outpath, f'{video_prefix}_scale.json')
        if osp.isfile(out_file) and not args.redo:
            print(f'{out_file} already exists, skipping...')
            return 
        
        # get the frame index used to reconstruct the mesh, filename format: <video_prefix>*_<frame_index>_rgba.obj
        hy_file = sorted(glob(f'{self.args.hy3d_root}/{video_prefix}*/*{video_prefix}*.obj'))[0]
        frame_time = '000' + osp.basename(hy_file).split('_')[-2]

        # Get RGB and depth 
        color, depth = get_specific_frame(f'{osp.dirname(args.video)}/{video_prefix}', frame_time, kid=0)
        # Get mask

        assert self.scale_ratio == 1.0, "the camera should not be rescaled"
        camera_K = self.camera_K.copy() 

        h5_file = f'{args.masks_root}/{video_prefix}_masks_k0.h5'
        h5_data = h5py.File(h5_file, 'r')
        mname_o = f'{video_prefix}/{frame_time}-k0.obj_rend_mask.png'
        mask_o = h5_data[mname_o][:] 
        mask_o = mask_o.astype(np.uint8) * 255

        # Init foundationpose
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()

        estimate_metric_scale(scorer, refiner, glctx, args.outpath, hy_file, color, depth, mask_o, camera_K)



        
    
    

if __name__ == '__main__':
    import argparse
    args = BaseBehaveVideoData.get_parser().parse_args()

    estimator = MetricScaleEstimator(args)
    estimator.estimate_scale(args)