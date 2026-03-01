# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import sys, os

sys.path.append(os.getcwd())
import cv2
import numpy as np
import glob
from tqdm import tqdm
import joblib
import os.path as osp
import imageio
import Utils
import h5py
from behave_data.behave_video import BaseBehaveVideoData, load_masks
from videoio import Uint16Reader, Uint16Writer
import json
import prep.align_utils as align_utils



class MonodepthAligner(BaseBehaveVideoData):
    def align(self, args, kid_to_run=None):
        ""
        kids = self.kids
        tars = [h5py.File(self.tar_path.replace('_masks_k0.h5', f'_masks_k{k}.h5'), 'r') for k in kids]

        self.scale_ratio = 1 
        self.camera_K[:2] *= self.scale_ratio # do not do any resize to avoid artifacts in dmap resizing 
        from behave_data.utils import get_intrinsics_unified
        if not args.wild_video:
            K_all = [get_intrinsics_unified(self.args.data_source, self.video_prefix, kid, self.args.wild_video) for kid in self.kids]
            K_all = np.stack(K_all)
            K_all[:, :2] /= self.scale_ratio # make sure the resolution matches 
        else:
            K_all = self.camera_K[None].copy()

        video_root = osp.dirname(args.video)
        if 'aligned' in video_root:
            video_root = video_root.replace('-aligned', '')
            # to remove the -aligned from the video_root
            print(f'removing -aligned from {video_root}')
        method_name = osp.basename(video_root)

        scales, shifts = [], []
        outdir = video_root + '-aligned'
        os.makedirs(outdir, exist_ok=True)
        viz_res = False 

        for k in kids:
            if kid_to_run is not None and k != kid_to_run:
                continue
            outfile_pkl = osp.join(outdir, f'{self.video_prefix}.{k}.depth-reg_aligned.pkl')
            if osp.isfile(outfile_pkl):
                print(f'{outfile_pkl} already exists, skipping')
                continue 
            
            # get the color and depth 
            dreader_mono = Uint16Reader(osp.join(video_root, f'{self.video_prefix}.{k}.depth-reg.mp4'))

            dfile_gt = self.get_target_dfile(k, method_name, video_root)
            dreader_gt = Uint16Reader(dfile_gt)
            time_file = osp.join(video_root, f'{self.video_prefix}.{k}.time.json')
            video_len = dreader_mono.video_params['length']
            assert video_len == dreader_gt.video_params['length'], f'length of depth videos does not match: {video_len} != {dreader_gt.video_params["length"]}'
            print("loaded videos from Mono:", osp.join(video_root, f'{self.video_prefix}.{k}.depth-reg.mp4'), 'and GT:', osp.join(video_root.replace(f'/{method_name}', '/videos'), f'{self.video_prefix}.{k}.depth-reg.mp4'))
            if not osp.isfile(time_file):
                frame_times_actual = np.arange(0, video_len)
                frame_times_file = frame_times_actual.copy() 
            else:
                frame_times_actual = np.array(json.load(open(time_file))['depth'], dtype=float) / 1e6
                frame_times_file = np.arange(0, frame_times_actual[-1], 1. / 30.).tolist()
            assert len(frame_times_actual) == video_len, f'length of frame times does not match: {len(frame_times_actual)} != {video_len}'
            iter_mono = iter(dreader_mono)
            iter_gt = iter(dreader_gt)

            outfile = osp.join(outdir, f'{self.video_prefix}.{k}.depth-reg.mp4')
            # add synlink for color file
            os.symlink(f'../{method_name}/{self.video_prefix}.{k}.color.mp4', outfile.replace('.depth-reg.mp4', '.color.mp4'))
            if osp.isfile(f'{video_root}/{self.video_prefix}.{k}.color.pkl'):
                # the pkl for in the wild videos where intrinsics are estimated.
                os.symlink(f'../{method_name}/{self.video_prefix}.{k}.color.pkl', outfile.replace('.depth-reg.mp4', '.color.pkl'))

            depth_writer = None 
            if viz_res:
                file = f'outputs/viz-dmap-align/{self.video_prefix}.{k}.depth-align_{method_name}.mp4'
                vw = imageio.get_writer(file, fps=30)

            scales, shifts = [], []
            for i in tqdm(range(video_len)):
                dmap_mono = np.array(next(iter_mono))
                dmap_gt = np.array(next(iter_gt))
                # get the masks
                time_actual = frame_times_actual[i]
                # get closest frame time 
                time_frame = frame_times_file[np.argmin(np.abs(frame_times_file - time_actual))]
                frame_time = self.get_time_str(time_frame)

                mask_h, mask_o = load_masks(self.video_prefix, frame_time, k, tars[k])
                if mask_h is None:
                    mask_h = np.ones_like(dmap_mono).astype(bool)
                    mask_o = np.ones_like(dmap_mono).astype(bool)
                    print(f'no mask found for frame {self.video_prefix}/{frame_time}, kinect {k}, using all valid pixels')
                else:
                    mask_h = mask_h.astype(bool)
                    mask_o = mask_o.astype(bool)

                # process the dmap
                dmap_mono_orig = dmap_mono.copy()/1000.
                device = 'cuda' #TODO: save the original dmap + scale and shift 
                dmap_mono = Utils.erode_depth(dmap_mono/1000., radius=2, device=device)
                dmap_mono = Utils.bilateral_filter_depth(dmap_mono, radius=2, device=device)
                dmap_gt = Utils.erode_depth(dmap_gt/1000., radius=2, device=device)
                dmap_gt = Utils.bilateral_filter_depth(dmap_gt, radius=2, device=device)

                human_only = args.wild_video 
                if not human_only:
                    mask_gt = (mask_h | mask_o ) & (dmap_gt > 0)
                    mask_mono = (mask_h | mask_o ) & (dmap_mono > 0)
                else:
                    # use only human mask to get GT, but for alignment, apply to both GT and human mask 
                    mask_gt = mask_h & (dmap_gt > 0)
                    mask_mono = (mask_h | mask_o ) & (dmap_mono > 0)
                scale, shift = align_utils.compute_scale_and_shift_robust(dmap_mono, dmap_gt, mask_mono&mask_gt)
                depth_aligned = dmap_mono_orig.copy() # do not do filtering here
                # scale the whole thing 
                depth_aligned[mask_mono] = depth_aligned[mask_mono] * scale + shift

                if depth_writer is None:
                    H, W = dmap_mono.shape[:2]
                    depth_writer = Uint16Writer(outfile, (W, H), fps=30)
                dmap_aligned = (depth_aligned * 1000).astype(np.uint16)
                depth_writer.write(dmap_aligned)
                scales.append(scale)
                shifts.append(shift)

                # visualize the depth as rgb image 
                if viz_res:
                    vis_aligned = cv2.applyColorMap((depth_aligned*100).astype(np.uint8), cv2.COLORMAP_JET)
                    vis_mono = cv2.applyColorMap((dmap_mono*100).astype(np.uint8), cv2.COLORMAP_JET)
                    vis_gt = cv2.applyColorMap((dmap_gt*100).astype(np.uint8), cv2.COLORMAP_JET)
                    vis_diff = cv2.applyColorMap((np.abs(dmap_mono-dmap_gt)*100).astype(np.uint8), cv2.COLORMAP_JET)
                    vis_diff2 = cv2.applyColorMap((np.abs(depth_aligned-dmap_gt)*100).astype(np.uint8), cv2.COLORMAP_JET)
                    comb = np.concatenate((vis_mono, vis_gt, vis_aligned, vis_diff, vis_diff2), axis=1)
                    comb = cv2.resize(comb, (comb.shape[1]//4, comb.shape[0]//4))
                    vw.append_data(comb)
            depth_writer.close()
            # close readers
            dreader_mono.close()
            dreader_gt.close()
            print(f'{outfile} done')

            # save the scales and shifts
            scales = np.array(scales)
            shifts = np.array(shifts)
            joblib.dump({'scales': scales, 'shifts': shifts}, outfile_pkl)

    def get_target_dfile(self, k, method_name, video_root) -> str:
        "get the target depth file that will be aligned to"
        dfile_gt = osp.join(video_root.replace(f'/{method_name}', '/videos'), f'{self.video_prefix}.{k}.depth-reg.mp4')
        return dfile_gt


def child_run(args, kid_to_run):
    if 'aligned' in args.video:
        args.video = args.video.replace('-aligned', '')
        # to remove the -aligned from the video_root
        print(f'removing -aligned from {args.video}')
    aligner = MonodepthAligner(args)
    aligner.align(args, kid_to_run)
            

if __name__ == '__main__':
    args = MonodepthAligner.get_parser().parse_args()
    import multiprocessing as mp
    import time 
    from copy import deepcopy
    mp.set_start_method('spawn', force=True)
    ctx = mp.get_context('spawn')

    if osp.isfile(args.video):
        videos = [args.video]
    else:
        videos = sorted(glob.glob(args.video))

    chunk_size = len(videos) // 8 + 1 
    if args.index is not None:
        videos = videos[args.index * chunk_size:(args.index + 1) * chunk_size]
    max_procs_per_gpu = 2
    procs = []
    print(f'running {len(videos)} videos, first video: {videos[0]}, last video: {videos[-1]}')
    args_orig = args  
    kids = [0, 1, 2, 3] if args.data_source != 'intercap' else [0, 1, 2, 3, 4, 5]
    if args.wild_video:
        kids = [0] 
    for i in range(len(videos)):
        args = deepcopy(args_orig)   
        args.video = videos[i]
        for k in kids:
            # check if the 
            proc = ctx.Process(target=child_run, args=(args, k))
            proc.start()
            time.sleep(2) # to avoid race condition
            procs.append(proc)
        for p in procs:
            p.join()
    print('all done')
    