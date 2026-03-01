# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
align the NLF predictions to the unidepth predictions
"""

import sys, os
sys.path.append(os.getcwd())
import cv2
import numpy as np
import glob
from tqdm import tqdm
import trimesh
import os.path as osp
import Utils
import joblib
import h5py
import torch
from behave_data.behave_video import BaseBehaveVideoData, load_masks
from lib_smpl import get_smpl, SMPL_MODEL_ROOT
from tools import icp_utils
import open3d as o3d 
from smplfitter.pt import BodyModel, BodyFitter
import time


class NLF2Unidepth(BaseBehaveVideoData):
    "align the NLF predictions to the unidepth predictions"
    def align(self, args, kid_to_run=None):
        outfile = f'{args.outpath}/{self.video_prefix}_params.pkl'
        if osp.isfile(outfile) and not args.redo:
            print(f'{outfile} already exists, skipping...all done')
            return
        self.device = device = 'cuda'
        loop = tqdm(self.times)
        loop.set_description(f"processing {self.video_prefix}")
        kids = self.kids

        tars = [h5py.File(self.tar_path.replace('_masks_k0.h5', f'_masks_k{k}.h5'), 'r') for k in kids]

        # Load NLF prediction and check 
        nlf_file = f'{args.nlf_path}/{self.video_prefix}_params.pkl'
        if not osp.isfile(nlf_file):
            print(f'{nlf_file} does not exist, skipping')
            return 
        nlf_data = joblib.load(nlf_file)
        nlf_transls = nlf_data['transls'] 
        nlf_gender = nlf_data['gender']
        # now get the SMPL verts from NLF
        smpl_model = get_smpl(nlf_gender, hands=True).to(self.device)
        nlf_verts_all = []
        for k, _ in enumerate(kids):
            nlf_smpl_verts = smpl_model(torch.from_numpy(nlf_data['poses'][:, k]).to(self.device).float(),
                                        torch.from_numpy(nlf_data['betas'][:, k]).to(self.device).float(),
                                        torch.from_numpy(nlf_transls[:, k]).to(self.device).float())[0].cpu().numpy()
            nlf_verts_all.append(nlf_smpl_verts)
        nlf_verts_all = np.stack(nlf_verts_all, axis=1)

        if not self.args.wild_video:
            from behave_data.utils import get_intrinsics_unified
            K_all = [get_intrinsics_unified(self.args.data_source, self.video_prefix, kid, self.args.wild_video) for kid in self.kids]
            K_all = np.stack(K_all)
            K_all[:, :2] /= self.scale_ratio # make sure the resolution matches 
        else:
            K_all = [self.camera_K] # this already take into account the scale ratio 

        last_frame = float(nlf_data['frames'][-1][1:])
        first_frame = float(nlf_data['frames'][0][1:])
        times_cut = [t for t in self.times if round(t, 3) <= last_frame and round(t, 3) >= first_frame]
        print(f'Sequence {self.video_prefix} Cutting times to {len(self.times)} frames, last frame: {times_cut[-1]}, original last frame: {self.times[-1]}, packed data last frame: {nlf_data["frames"][-1]}')
        # print out first frame info
        print(f'Sequence {self.video_prefix} First frame: {first_frame}, original first frame: {self.times[0]}, after cut: {times_cut[0]}, packed data first frame: {nlf_data["frames"][0]}')
        if len(times_cut) != len(nlf_verts_all):
            if len(times_cut) < len(nlf_verts_all):
                gt_data = joblib.load(f'/home/xianghuix/datasets/behave/behave-packed/{self.video_prefix}_GT-packed.pkl')
                frames_gt = gt_data['frames']
                times_cut = [self.time_str_to_float(t) for t in frames_gt]
            else:
                # simply append the last frame of nlf_verts_all
                L1 = len(times_cut)
                L2 = len(nlf_verts_all)
                assert np.abs(L2 - L1) < 5, f'inconsistent number of frames {L2}!={L1} on {self.video_prefix}!'
                nlf_verts_all = np.concatenate([nlf_verts_all, nlf_verts_all[-1:].repeat(L2 - L1, axis=0)], axis=0)
                print(f'Sequence {self.video_prefix} Padding nlf_verts_all to {len(nlf_verts_all)} frames')
        self.times = times_cut
        assert len(nlf_verts_all) == len(self.times), f'inconsistent number of frames {len(nlf_verts_all)}!={len(self.times)} on {self.video_prefix}!'

        fitter = BodyFitter(BodyModel('smplh', nlf_gender, model_root=SMPL_MODEL_ROOT).to('cuda')).to(device)
        param_names = ['poses', 'betas', 'transls', 'center_pts', 'center_verts']
        params_all = {name: [] for name in param_names}

        # load exclude frames
        
        for enum_idx, k in enumerate(kids):
            if kid_to_run is not None and k != kid_to_run:
                continue
            outfile_kid = f'{args.outpath}/{self.video_prefix}_params_k{k}.pkl'
            if osp.isfile(outfile_kid):
                print(f'{outfile_kid} already exists, skipping')
                # directly load the results without reruning 
                params_k = joblib.load(outfile_kid)
                params_all['poses'].append(params_k['poses'])
                params_all['betas'].append(params_k['betas'])
                params_all['transls'].append(params_k['transls'])
                params_all['center_pts'].append(params_k['center_pts'])
                params_all['center_verts'].append(params_k['center_verts'])
                continue

            verts_aligned = [] 
            frame_times = []
            center_pts = []
            center_verts = []
            for i, t in enumerate(tqdm(self.times)):
                frame_time = self.get_time_str(t)
                frame_times.append(frame_time)

                # skip if this frame will be ignored in training data. 
                _, depth = self.load_color_depth(enum_idx, self.kids, t)

                mask_h, mask_o = load_masks(self.video_prefix, frame_time, k, tars[enum_idx])

                # resize depth and mask by the scale ratio
                h, w = depth.shape[:2]
                depth = cv2.resize(depth, (int(w / self.scale_ratio), int(h / self.scale_ratio)), cv2.INTER_NEAREST) 
                mask_h = cv2.resize(mask_h, (int(w / self.scale_ratio), int(h / self.scale_ratio))) 

                # resize depth to the same resolution as the color image
                # count the time for each step 
                t0 = time.time() # note: when additing gaussian  noise, the erosion step will create a lot of 0s!
                depth = Utils.erode_depth(depth/1000., radius=2, device='cuda')
                depth = Utils.bilateral_filter_depth(depth, radius=2, device='cuda')
                dmap_xyz = Utils.depth2xyzmap(depth, K_all[enum_idx])
                t1 = time.time() # this takes 0.206s, ICP: 0.134s, Rescale: 0.000s, ICP2: 0.072s!
                # resize to half: Dmap filter: 0.093s, ICP: 0.037s, Rescale: 0.000s, ICP2: 0.046s 
                # ICP becomes super slow when multiple (4) processes are running: Dmap filter: 0.071s, ICP: 0.936s, Rescale: 0.000s, ICP2: 0.705s

                # now align NLF to the point cloud 
                verts_nlf = nlf_verts_all[i, enum_idx] 
                pts_hum = dmap_xyz[mask_h>0].reshape((-1, 3))
                pts_nlf_sample = trimesh.Trimesh(verts_nlf, smpl_model.faces, process=False).sample(8000)
                # align mini z first: this is quite unreliable! 
                z_pts_nlf = np.median(pts_nlf_sample[:, 2])
                z_pts_hum_median = np.median(pts_hum[:, 2])
                mat = np.eye(4)
                mat[:3, 3] = [0, 0, z_pts_hum_median - z_pts_nlf]
                pts_nlf = np.matmul(pts_nlf_sample, mat[:3, :3].T) + mat[:3, 3]
                # now run icp 
                src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_nlf))
                tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_hum))
                t2 = time.time()
                t3 = time.time()
                mat_icp = icp_utils.translation_only_icp_torch(src, tgt, voxel_size=0.01, max_iters=[25, 10, 5])
                t4 = time.time()
                # torch is 10 times faster than open3d! 
                mat = np.matmul(mat_icp, mat)
                z_nlf_orig = np.mean(verts_nlf[:, 2])
                z_nlf_new = mat[:3, 3][2] + z_nlf_orig
                scale = z_nlf_new / z_nlf_orig 
                mat = np.eye(4)
                np.fill_diagonal(mat, scale)
                mat[3, 3] = 1
                mat[:3, 3] = [0, 0, z_pts_hum_median - z_pts_nlf*scale]
                # redo the alignment with the new scale
                t4 = time.time()
                src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.matmul(pts_nlf_sample, mat[:3, :3].T) + mat[:3, 3]))
                mat_icp = icp_utils.translation_only_icp_torch(src, tgt, voxel_size=0.01)
                t5 = time.time()
                mat = np.matmul(mat_icp, mat)
                verts_nlf_align = np.matmul(verts_nlf, mat[:3, :3].T) + mat[:3, 3] # TODO: implement rescaling, and further align. 
                # check if this alignment is reasonable
                z = np.mean(verts_nlf_align[:, 2])
                if z < 1.0:
                    print(f'warning: z is too small {z} on frame {frame_time} for kid {enum_idx}, skipping')
                    verts_nlf_align = verts_nlf # do not do any alignment, just raw nlf 
                verts_aligned.append(verts_nlf_align) 
                center_pts.append(np.mean(pts_hum, axis=0))
                center_verts.append(np.mean(verts_nlf, axis=0))
                ## End of aligning NLF to depth
            # now run SMPLH fitter 
            verts_aligned = np.stack(verts_aligned, 0)
            fit_res = fitter.fit(torch.from_numpy(verts_aligned).to(device).float(), num_iter=3,
                             requested_keys=['shape_betas', 'trans', 'vertices', 'pose_rotvecs'])

            # save the result for this kid
            params_k = {
                'poses': fit_res['pose_rotvecs'].cpu().numpy(),
                'betas': fit_res['shape_betas'].cpu().numpy(),
                'transls': fit_res['trans'].cpu().numpy(),
                'center_pts': np.stack(center_pts, axis=0),
                'center_verts': np.stack(center_verts, axis=0),
                'frames': frame_times,
                'gender': nlf_gender,
                'kids': kids,
            }
            joblib.dump(params_k, outfile_kid)
            print(f'saved to {outfile_kid}, kinect {k} done')
            params_all['poses'].append(fit_res['pose_rotvecs'].cpu().numpy())
            params_all['betas'].append(fit_res['shape_betas'].cpu().numpy())
            params_all['transls'].append(fit_res['trans'].cpu().numpy())
            params_all['center_pts'].append(np.stack(center_pts, axis=0))
            params_all['center_verts'].append(np.stack(center_verts, axis=0))
        
def child_run(args, kid_to_run):
    renderer = NLF2Unidepth(args)
    renderer.align(args, kid_to_run)



if __name__ == '__main__':
    import multiprocessing as mp
    from copy import deepcopy

    mp.set_start_method('spawn', force=True)
    ctx = mp.get_context('spawn')
    parser = NLF2Unidepth.get_parser()
    args = parser.parse_args()

    if osp.isfile(args.video):
        videos = [args.video]
    else:
        videos = sorted(glob.glob(args.video)) # process multiple video
    if args.index is not None:
        chunk_size = len(videos) // 100 + 1 
        videos = videos[args.index * chunk_size:(args.index + 1) * chunk_size]
    print(f"In total {len(videos)} video files")
    kids = args.cameras 
    if args.wild_video:
        kids = [0] 
    args_orig = args
    for video in tqdm(videos):
        video_prefix = osp.basename(video).split('.')[0]
        outfile = f'{args.outpath}/{video_prefix}_params.pkl'
        if osp.isfile(outfile) and not args.redo:
            print(f'{outfile} already exists, skipping')
            continue

        # multiplrocessing 
        args = deepcopy(args_orig)
        args.video = video
        procs = []
        for k in kids:
            p = ctx.Process(target=child_run, args=(args, k))
            p.start()
            time.sleep(3) # to avoid race condition
            procs.append(p)
        for p in procs:
            p.join()
        
        # now collect the results 
        params_all = {name: [] for name in ['poses', 'betas', 'transls', 'center_pts', 'center_verts']}
        for enum_idx, kid in enumerate(kids):
            outfile_kid = f'{args.outpath}/{video_prefix}_params_k{kid}.pkl'
            if not osp.isfile(outfile_kid):
                print(f"Warning: file {outfile_kid} not found, skipping")
                continue
            params_k = joblib.load(outfile_kid)
            params_all['poses'].append(params_k['poses'])
            params_all['betas'].append(params_k['betas'])
            params_all['transls'].append(params_k['transls'])
            params_all['center_pts'].append(params_k['center_pts'])
            params_all['center_verts'].append(params_k['center_verts'])
            
        params_all = {name: np.stack(params_all[name], axis=1) for name in params_all.keys()}
        params_all['frames']= params_k['frames']
        params_all['gender'] = params_k['gender']
        params_all['kids'] = params_k['kids']
        joblib.dump(params_all, outfile)
        
        print(f'aligned SMPLH params saved to {outfile}, all done')
        
