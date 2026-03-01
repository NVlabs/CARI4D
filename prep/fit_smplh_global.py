# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
optimize SMPLH parameters to have globally consistent pose and translations
"""

import sys, os
sys.path.append(os.getcwd())
import cv2
import numpy as np
import glob
from tqdm import tqdm
import joblib
import os.path as osp
import torch 
from transformers import get_scheduler
import torch.nn.functional as F
import smplx
import imageio
from lib_smpl import get_smpl, SMPL_MODEL_ROOT, SMPL_ASSETS_ROOT
from behave_data.behave_video import BaseBehaveVideoData
from lib_smpl.body_landmark import BodyLandmarks


class GlobalSMPLHOptimizer(BaseBehaveVideoData):
    def fit(self, args):
        "fit SMPLH parameters, optionally with global camera poses"
        assert args.wild_video, 'only wild video is supported'
        seq_name = osp.basename(args.video).split('.')[0]
        self.device = 'cuda'
        view_id = 0 

        # load the SMPLH parameters, and compute average shape 
        nlf_file = f'{args.nlf_path}/{seq_name}_params.pkl'
        if not osp.isfile(nlf_file):
            print(f'{nlf_file} does not exist, skipping')
            return 
        outfile = osp.join(osp.dirname(nlf_file)+'-opt', f'{seq_name}_params.pkl')
        os.makedirs(osp.dirname(outfile), exist_ok=True)
        if osp.isfile(outfile):
            print(f'{outfile} already exists, skipping')
            return 
        nlf_data = joblib.load(open(nlf_file, 'rb'))
        nlf_transls = nlf_data['transls'][:, view_id] 
        nlf_gender = nlf_data['gender']
        nlf_poses = nlf_data['poses'][:, view_id]
        nlf_betas = nlf_data['betas'][:, view_id] # (T, 10), compute an average
        nlf_betas = np.mean(nlf_betas, axis=0)[None].repeat(len(nlf_poses), axis=0) 
        # to torch tensors 
        smpl_global_pose = torch.from_numpy(nlf_poses[:, :3]).float().to(self.device).requires_grad_(True)
        smpl_body_pose = torch.from_numpy(nlf_poses[:, 3:66]).float().to(self.device).requires_grad_(True) # translation opt only will lead to very strange translation, prob. bc. all gradients are injected there 
        smpl_hands = torch.from_numpy(nlf_poses[:, 66:]).float().to(self.device) 
        smpl_trans = torch.from_numpy(nlf_transls).float().to(self.device).requires_grad_(True)
        smpl_betas = torch.from_numpy(nlf_betas).float().to(self.device)

        # for body landmarks 
        self.landmark = BodyLandmarks(SMPL_ASSETS_ROOT)
       
        batch_size = min(256, len(smpl_global_pose))
        total_steps = 1000 
        smplh_model = smplx.create(
            model_path=SMPL_MODEL_ROOT,
            model_type='smplh',
            gender=nlf_gender,
            use_pca=False,
            batch_size=batch_size,
            flat_hand_mean=True # the given hands are directly in the axis angle format
        ).to(self.device)

        # Load the 2D keypoints 
        pack_file = f'{args.packed_root}/{seq_name}_GT-packed.pkl'
        pack_data = joblib.load(pack_file)
        joints_2d = pack_data['joints2d'][:, view_id] # (L, 25, 3) 
        joints_2d = torch.from_numpy(joints_2d).float().to(self.device)
        op_thres = 0.4 
        assert len(joints_2d) == len(smpl_global_pose), f'the number of 2d joints does not match the number of frames: {len(joints_2d)} != {len(smpl_global_pose)}'

        # do optimization 
        lw_temp, lw_j2d = 10000.0, 0.01 # better pixel alignment, but still not perfect. 
        seq_len = len(smpl_global_pose) 

        lr = 1e-3 
        opt_params = [
            {'params': smpl_global_pose, 'lr': lr},
            {'params': smpl_body_pose, 'lr': lr},
            # {'params': smpl_hands, 'lr': lr},
            {'params': smpl_trans, 'lr': lr},
            {'params': smpl_betas, 'lr': lr},
        ]
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = get_scheduler(optimizer=optimizer, name='cosine',
                                              num_warmup_steps=total_steps//10,
                                              num_training_steps=int(total_steps * 1.5))

        K_full = self.camera_K.copy() 
        K_full_th = torch.from_numpy(K_full).float().to(self.device)
        loop = tqdm(range(total_steps))
        for step in loop: 
            # sample a batch of data 
            rid = np.random.randint(0, seq_len - batch_size+1)
            start_ind, end_ind = rid, min(seq_len, rid + batch_size)

            smplh_output = smplh_model(
                betas=smpl_betas[start_ind:end_ind],
                body_pose=smpl_body_pose[start_ind:end_ind],
                global_orient=smpl_global_pose[start_ind:end_ind],
                transl=smpl_trans[start_ind:end_ind],
                left_hand_pose=smpl_hands[start_ind:end_ind, :45],
                right_hand_pose=smpl_hands[start_ind:end_ind, 45:],
                return_verts=True,
                return_full_pose=True
            )

            joints_25_3d = self.landmark.get_body_kpts_batch_torch(smplh_output.vertices)
            joints_25_proj = joints_25_3d @ K_full_th.T
            joints_25_proj_xy = joints_25_proj[:, :, :2] / joints_25_proj[:, :, 2:3]  
            j2d_conf = joints_2d[start_ind:end_ind, :, 2:3]
            j2d_conf[j2d_conf < op_thres] = 0.0
            j2d_mask = j2d_conf 
            loss_j2d = F.mse_loss(joints_25_proj_xy, joints_2d[start_ind:end_ind, :, :2], reduction='none') * j2d_mask.float() 
            loss_j2d = loss_j2d.mean() * lw_j2d

            # now temporal smoothness 
            acc_h = smplh_output.joints[:-2] -2 * smplh_output.joints[1:-1] + smplh_output.joints[2:] # focus on joints, especially hands should be stable 
            loss_temp = (acc_h**2).sum(-1).mean() * lw_temp
            loss_total = loss_j2d + loss_temp 
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            scheduler.step()
            if step % 200 == 0:
                desc = f'step {step}, j2d: {loss_j2d.item():.2f}, temp: {loss_temp.item():.2f}, total: {loss_total.item():.2f}'
                loop.set_description(desc)

                # visualize the 2d points 
                with torch.no_grad():
                    smplh_output = smplh_model(
                        betas=smpl_betas,
                        body_pose=smpl_body_pose,
                        global_orient=smpl_global_pose,
                        transl=smpl_trans,
                        left_hand_pose=smpl_hands[:, :45],
                        right_hand_pose=smpl_hands[:, 45:],
                    )
                    joints_25_3d = self.landmark.get_body_kpts_batch_torch(smplh_output.vertices)
                    joints_25_proj = joints_25_3d @ K_full_th.T
                    joints_25_proj_xy = joints_25_proj[:, :, :2] / joints_25_proj[:, :, 2:3]

                    video_reader = imageio.get_reader(args.video)
                    video_writer = imageio.get_writer(outfile.replace('.pkl', f'_step{step:04d}.mp4'), 'FFMPEG', fps=30)
                    for i in range(seq_len):
                        img = video_reader.get_data(i)
                        img_orig = img.copy()
                        j2d_pr, j2d_gt = joints_25_proj_xy[i].cpu().numpy().astype(int), (joints_2d[i, :, :2]).cpu().numpy().astype(int)
                        j2d_mask = joints_2d[i, :, 2:3].cpu().numpy() > op_thres
                        j2d_gt[~j2d_mask.repeat(2, -1)] = 0 # set to -1 to avoid drawing 
                        for j1, j2 in zip(j2d_pr, j2d_gt):
                            cv2.circle(img, (j1[0], j1[1]), 3, (0, 255, 255), -1)
                            cv2.circle(img, (j2[0], j2[1]), 3, (255, 0, 0), -1)
                        img = np.concatenate([img_orig, img], 1)
                        video_writer.append_data(img)


        
        # save the results, same format as the nlf file 
        smpl_result = {
            "transls": smpl_trans.detach().cpu().numpy()[:, None],
            "poses": torch.cat([smpl_global_pose, smpl_body_pose, smpl_hands], 1).detach().cpu().numpy()[:, None],
            "betas": smpl_betas.detach().cpu().numpy()[:, None],
            "gender": nlf_gender,
            "frames": nlf_data['frames'],
        }
        joblib.dump(smpl_result, open(outfile, 'wb'))
        print(f'saved to {outfile}, all done')
    
    @staticmethod
    def get_parser():
        parser = BaseBehaveVideoData.get_parser()
        parser.add_argument('--packed_root', type=str, default='data/cari4d-demo/wild/packed/')
        return parser

def main():
    args = GlobalSMPLHOptimizer.get_parser().parse_args()
    if osp.isfile(args.video):
        videos = [args.video]
    else:
        videos = sorted(glob.glob(args.video)) # process multiple video
    if args.index is not None:
        chunk_size = len(videos) // 100 + 1 
        videos = videos[args.index * chunk_size:(args.index + 1) * chunk_size]
    print(f"In total {len(videos)} video files")
    
    for video in tqdm(videos):
        args.video = video
        optimizer = GlobalSMPLHOptimizer(args)
        optimizer.fit(args)

if __name__ == '__main__':
    main()


