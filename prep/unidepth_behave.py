# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import joblib
import sys, os
import os.path as osp
from glob import glob

import cv2
import time

sys.path.append(os.getcwd())
import torch
import numpy as np
import torch
from tqdm import tqdm
import numpy as np
from videoio import Uint16Writer
from unidepth.utils.camera import Pinhole
from unidepth.models import UniDepthV2

from behave_data.video_reader import VideoController
from behave_data.behave_video import BaseBehaveVideoData

def get_intrinsics(kid):
    assert kid in [0, 1, 2, 3], f'invalid kinect index {kid}!'
    if kid == 0:
        fx, fy = 976.212, 976.047
        cx, cy = 1017.958, 787.313
    elif kid == 1:
        fx, fy = 979.784, 979.840  # for original kinect coordinate system
        cx, cy = 1018.952, 779.486
    elif kid == 2:
        fx, fy = 974.899, 974.337
        cx, cy = 1018.747, 786.176
    else:
        fx, fy = 972.873, 972.790
        cx, cy = 1022.0565, 770.397
    return fx, fy, cx, cy

def get_intrinsics_unified(data_source, seq_name, kid, wild_video=False):
    if data_source == 'behave':
        # fx, fy, cx, cy = get_intrinsics(kid)
        if not wild_video:
            if kid == 0:
                fx, fy = 976.212, 976.047
                cx, cy = 1017.958, 787.313
            elif kid == 1:
                fx, fy = 979.784, 979.840  # for original kinect coordinate system
                cx, cy = 1018.952, 779.486
            elif kid == 2:
                fx, fy = 974.899, 974.337
                cx, cy = 1018.747, 786.176
            else:
                fx, fy = 972.873, 972.790
                cx, cy = 1022.0565, 770.397
        else:
            raise ValueError(f'Invalid wild video: {wild_video}')
        # Run inference. TODO: check if we can speed this up by batching
        intrinsics = np.array([[fx, 0, cx],
                            [0., fy, cy],
                            [0, 0., 1]])
    elif data_source == 'hodome':
        from behave_data.const import get_camera_K_hodome, HODOME_VIEW_IDS
        intrinsics = get_camera_K_hodome(seq_name, HODOME_VIEW_IDS[kid])
    elif data_source == 'intercap':
        from behave_data.const import ICAP_CENTERs
        from behave_data.const import ICAP_FOCALs
        intrinsics = ICAP_FOCALs[kid]
        intrinsics = np.array([[intrinsics[0], 0, ICAP_CENTERs[kid][0]],
                            [0, intrinsics[1], ICAP_CENTERs[kid][1]],
                            [0, 0, 1]])
    elif data_source == 'imhd':
        from behave_data.const import IMHD_VIEW_IDS
        from behave_data.const import get_IMHD_camera_K
        intrinsics = get_IMHD_camera_K(seq_name, IMHD_VIEW_IDS[kid])
    elif data_source == 'procigen':
        K = np.array([[979.784, 0, 1018.952],
                    [0, 979.840, 779.486],
                    [0, 0, 1]])
        K[:2] /= 2. 
        intrinsics = K 
    else:
        raise ValueError(f'Invalid data source: {data_source}')

    return intrinsics


class UniDepthBehaveProcessor(BaseBehaveVideoData):
    def init_camera_K(self):
        if self.wild_video:
            return np.zeros((3, 3))
        return super().init_camera_K()

    def process_video(self, kid_to_run=None):
        ""
        if self.model is None:
            model = self.init_model()
            model.eval()
            self.model = model
        model = self.model

        args = self.args
        for kid, ctrl in enumerate(self.controllers):
            if kid_to_run is not None and kid != kid_to_run:
                print('skipping kinect', kid)
                continue
            
            outfile = osp.join(self.args.outpath, osp.basename(ctrl.video_path).replace('.color.', '.depth-reg.'))
            print(f'{osp.basename(ctrl.video_path)} -> {outfile}')
            if osp.isfile(outfile):
                print("Already exists {}, all done".format(outfile))
                continue
            depth_writer = None

            video_iter = ctrl.video_iter
            focals = []
            if args.data_source == 'behave':
                fx, fy, cx, cy = get_intrinsics(kid)
                if not args.wild_video:
                    intrinsics = np.array([[fx, 0, cx],
                                    [0., fy, cy],
                                    [0, 0., 1]])
                else:
                    intrinsics = None
                
            elif args.data_source == 'hodome':
                from behave_data.const import get_camera_K_hodome, HODOME_VIEW_IDS
                intrinsics = get_camera_K_hodome(osp.basename(ctrl.video_path), HODOME_VIEW_IDS[kid])
            elif args.data_source == 'intercap':
                from behave_data.const import ICAP_CENTERs
                from behave_data.const import ICAP_FOCALs
                intrinsics = ICAP_FOCALs[kid]
                intrinsics = np.array([[intrinsics[0], 0, ICAP_CENTERs[kid][0]],
                                    [0, intrinsics[1], ICAP_CENTERs[kid][1]],
                                    [0, 0, 1]])
            elif args.data_source == 'imhd':
                from behave_data.const import IMHD_VIEW_IDS
                from behave_data.const import get_IMHD_camera_K
                intrinsics = get_IMHD_camera_K(osp.basename(ctrl.video_path).split(".")[0], IMHD_VIEW_IDS[kid])
            elif args.data_source == 'procigen':
                K = np.array([[979.784, 0, 1018.952],
                          [0, 979.840, 779.486],
                          [0, 0, 1]])
                K[:2] /= 2. 
                intrinsics = K 
            else:
                raise ValueError(f'Invalid data source: {args.data_source}')
            print("camera intrinsics:", intrinsics)
            try:
                for fidx, img in enumerate(tqdm(video_iter)):
                    image = np.array(img)
                    rgb = torch.from_numpy(image).permute(2, 0, 1)  # C, H, W
                    # prepare intrinsics based on data source
                    camera = Pinhole(K=torch.from_numpy(intrinsics)) if intrinsics is not None else None
                    prediction = model.infer(rgb, camera)
                    depth = prediction["depth"][0, 0] # return B, 1, H, W  # Depth in [m].
                    focals.append(prediction['intrinsics'][0, 0, 0].cpu().numpy()) # (B, 3, 3)
                    if args.wild_video and intrinsics is None:
                        # update intrinsics
                        intrinsics = prediction['intrinsics'].cpu().numpy()[0]
                        print("Intrinsics updated!", intrinsics)
                    dmap = depth.cpu().numpy()  # (H, W), meter
                    dmap = (dmap * 1000).astype(np.uint16)
                    if depth_writer is None:
                        H, W = image.shape[:2]  # dynamic size
                        print('using image reso:', H, W)
                        depth_writer = Uint16Writer(outfile, (W, H), fps=args.fps)
                    depth_writer.write(dmap)
                        
            except StopIteration:
                # video done, save dmap
                depth_writer.close()
                print(f'{outfile} done')
                continue
            if len(focals) > 0:
                print('average focal length:', np.mean(focals))
                # save as file
                if args.wild_video:
                    assert len(focals) > 0
                    # fx, fy = np.mean(focals), np.mean(focals)
                    # cx, cy = W/2, H/2
                    pkl_file = self.args.video.replace('.mp4', '.pkl')
                    # use the 1st frame intrinsics
                    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
                    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
                    intrinsics = {'fx':fx, 'fy':fy, 'cx':cx, 'cy':cy, 'focals':focals, 'H': H, 'W': W}
                    print('saving intrinsics:', intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy'], 'to', pkl_file)
                    joblib.dump(intrinsics, pkl_file)
                    print('intrinsics saved to {}'.format(pkl_file))


    def init_model(self):
        name = 'unidepth-v2-vitl14'
        model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}").cuda()
        setattr(model, 'resolution_level', 9)  # set to maximum resolution
        return model

def _child_run_kid(kid, args):
    proc = UniDepthBehaveProcessor(args)
    proc.process_video(kid)

def process_procigen(args, kid_to_run):
    "because each video is very small, to reduce loading time, load model only once"
    name = 'unidepth-v2-vitl14'
    model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}").cuda()
    setattr(model, 'resolution_level', 9)  # set to maximum resolution
    
    videos = sorted(glob(args.video))
    # cut the video into chunks 
    chunk_size = len(videos) // 8 
    if args.index is not None:
        videos = videos[args.index * chunk_size:(args.index + 1) * chunk_size]
    print(f'processing {len(videos)} videos, first video: {videos[0]}, last video: {videos[-1]}')
    for video in tqdm(videos):
        assert '0.color.' in video, f'invalid video path {video}'
        video_k = video.replace('.0.color.', f'.{kid_to_run}.color.')
        outfile = osp.join(args.outpath, osp.basename(video_k).replace('.color.', '.depth-reg.'))
        if osp.isfile(outfile):
            print("Already exists {}, all done".format(outfile))
            continue
        controller = VideoController(video_k)
        K = np.array([[979.784, 0, 1018.952],
                          [0, 979.840, 779.486],
                          [0, 0, 1]])
        intrinsics = K 
        print("Using Procigen intrinsics:", intrinsics)
        video_iter = controller.video_iter
        depth_writer = None
        
            
        for img in video_iter:
            image = np.array(img)
            # resize image to double the reso
            h, w = image.shape[:2]
            image = cv2.resize(image, (w*2, h*2), interpolation=cv2.INTER_LINEAR)
            rgb = torch.from_numpy(image).permute(2, 0, 1)  # C, H, W
            # prepare intrinsics based on data source
            camera = Pinhole(K=torch.from_numpy(intrinsics)) 
            prediction = model.infer(rgb, camera)
            depth = prediction["depth"][0, 0] # return B, 1, H, W  # Depth in [m].
            dmap = depth.cpu().numpy()  # (H, W), meter
            dmap = (dmap * 1000).astype(np.uint16)
            if depth_writer is None:
                # now resize dmap by half
                dmap = cv2.resize(dmap, (w, h), interpolation=cv2.INTER_NEAREST)
                H, W = dmap.shape[:2]
                # H, W = image.shape[:2]  # dynamic size
                print('using image reso:', H, W)
                depth_writer = Uint16Writer(outfile, (W, H), fps=args.fps)
            depth_writer.write(dmap)
        depth_writer.close()
        print(f'{outfile} done')


if __name__ == '__main__':
    from multiprocessing import Process
    import multiprocessing as mp

    mp.set_start_method('spawn', force=True)
    ctx = mp.get_context('spawn')

    parser = BaseBehaveVideoData.get_parser()

    args = parser.parse_args()

    args.nodepth = True

    if osp.isfile(args.video):
        videos = [args.video]
    else:
        videos = sorted(glob(args.video))
    print(f'in total {len(videos)} videos')
    os.makedirs(args.outpath, exist_ok=True)

    if args.data_source == 'procigen':
        procs = []  # run 4 processes in parallel, can be actually faster! 
        kids = args.cameras
        for k in kids:
            p = ctx.Process(target=process_procigen, args=(args, k))
            p.start()
            time.sleep(10)   # to avoid race condition
            procs.append(p)
        for p in procs:
            p.join()
    else:
        for file in videos:
            args.video = file
            if 'empty' in args.video:
                continue
            obj_name = osp.basename(args.video).split('_')[2]
            if obj_name in ['boxtiny', 'boxsmall', 'basketball', 'keyboard', 'toolbox']:
                continue # these objects are too small to run inf 
            proc = UniDepthBehaveProcessor(args)
            proc.process_video()
    print('all done')